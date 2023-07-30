# Adapted from YOLO-NAS tutorial by Harpreet:
# https://colab.research.google.com/drive/10N6NmSMCiRnFlKV9kaIS_z3pk0OI1xKC?usp=sharing
# https://colab.research.google.com/drive/1q0RmeVRzLwRXW-h9dPFSOchwJkThUy6d#scrollTo=m0SkK3bjMOqH

import argparse
import logging
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from PIL import Image
from psd_tools import PSDImage
from super_gradients.common.object_names import Models
from super_gradients.training import models
from super_gradients.training.models.detection_models.pp_yolo_e.post_prediction_callback import (
    PPYoloEPostPredictionCallback,  # noqa: E501
)
from super_gradients.training.utils.predict import (  # noqa: E501
    DetectionPrediction,
    ImageDetectionPrediction,
)
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from sliding_window_metrics import Metrics

SWLOG = logging.getLogger("SlidingWindow")

os.environ["CUDA_LAUNCH_BLOCKINGs"] = "1"

MAX_IMAGE_SIZE = 10000


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=str, help="Path to data directory to test")
    parser.add_argument("model_path", type=str, help="Path to checkpoint")
    parser.add_argument("num_classes", type=int, help="Number of classes")
    parser.add_argument(
        "--window_size", "-s", type=int, help="Size of the sliding window"
    )
    parser.add_argument(
        "--window_overlap", "-o", type=float, help="Overlap between sliding windows"
    )
    parser.add_argument(
        "--conf", "-c", type=float, help="Confidence threshold for predictions"
    )
    parser.add_argument(
        "--iou",
        "-i",
        type=float,
        help="Max overlap between bboxes before one is removed during NMS",
    )
    parser.add_argument(
        "--on_edge_penalty",
        "-p",
        type=float,
        help="Penalty for bboxes on the edge of the image",
    )
    parser.add_argument(
        "--edge_threshold",
        "-e",
        type=float,
        help="Threshold for what is considered an edge of the window",
    )
    parser.add_argument(
        "--ground_truth_path", "-g", type=str, help="Path to ground truth file"
    )
    parser.add_argument(
        "--dataset_yaml", "-y", type=str, help="Path to dataset yaml file"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="Be verbose",
        action="store_const",
        dest="loglevel",
        const=logging.DEBUG,
        default=logging.INFO,
    )
    return parser.parse_args()


@dataclass
class SlidingWindowDetect:
    DEFAULT_CONF = {
        "window_size": 640,
        "window_overlap": 0.3,
        "conf": 0.4,
        "iou": 0.7,
        "on_edge_penalty": 0.3,
        "edge_threshold": 0.05,
    }

    model_path: str
    num_classes: int
    window_size: Optional[int] = None
    window_overlap: Optional[float] = None

    ground_truth_path: Optional[str] = None
    dataset_yaml_path: Optional[str] = None

    # Confidence threshold for predictions, below which predictions are discarded
    conf: Optional[float] = None

    # Max overlap between bboxes before one is removed during NMS
    iou: Optional[float] = None

    # Penalty for bboxes on the edge of the image (in confidence is removed, 0.3 = 30%).
    # Scores after penalty don't go below conf.
    on_edge_penalty: Optional[float] = None

    # Threshold for what is considered an edge of the window (0.05 = 5%)
    edge_threshold: Optional[float] = None

    def __post_init__(self):
        self._set_defaults()
        self._validate_input(vars(self))

        self.model_path = os.path.abspath(self.model_path)
        self.model_name = Models.YOLO_NAS_L
        self._edge_threshold = self.edge_threshold * self.window_size
        self._overlap_size = int(self.window_size * self.window_overlap)

        if self.ground_truth_path is not None:
            self.ground_truth_path = os.path.abspath(self.ground_truth_path)
            self._load_ground_truth()
        if self.dataset_yaml_path is not None:
            self.dataset_yaml_path = os.path.abspath(self.dataset_yaml_path)
            self._load_dataset_yaml()

        # Print out the all local variables
        SWLOG.info("[Configuration]")
        for key, value in vars(self).items():
            if not key.startswith("_"):
                SWLOG.info(f"{key}: {value}")

        self._get_model()

    def _set_defaults(self):
        for key, value in self.DEFAULT_CONF.items():
            if getattr(self, key, None) is None:
                setattr(self, key, value)

    def _validate_input(self, vars):
        assert os.path.exists(vars["model_path"])
        assert vars["num_classes"] > 0
        assert 0 <= vars["conf"] <= 1
        assert 0 <= vars["iou"] <= 1
        assert 0 <= vars["on_edge_penalty"] <= 1
        assert 0 <= vars["edge_threshold"] <= 1
        assert vars["window_size"] > 0
        assert 0 <= vars["window_overlap"] < 1
        if (
            vars["ground_truth_path"] is not None
            or vars["dataset_yaml_path"] is not None
        ):
            assert vars["ground_truth_path"] is not None
            assert os.path.exists(vars["ground_truth_path"])
            assert vars["dataset_yaml_path"] is not None
            assert os.path.exists(vars["dataset_yaml_path"])

    def _get_model(self):
        SWLOG.debug("Getting model...")
        self.model = models.get(
            self.model_name,
            checkpoint_path=self.model_path,
            num_classes=self.num_classes,
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)

    def _load_dataset_yaml(self):
        # Load the dataset yaml file
        SWLOG.debug("Loading dataset yaml...")
        import yaml

        with open(self.dataset_yaml_path, "r") as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
            self._classes = data["names"]
            SWLOG.debug(f"Classes ({type(self._classes)}): {self._classes}")

    def _load_ground_truth(self):
        SWLOG.debug("Loading ground truth...")
        # yolo format: [class, x, y, w, h], where x,y are the center of the bbox
        # and w,h are the width and height. All values are relative to the image size.
        # one line per bbox
        boxes = []
        labels = []
        with open(self.ground_truth_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(" ")
                labels.append(int(line[0]))
                boxes.append([float(x) for x in line[1:]])
        # Convert to NumPy arrays
        self._ground_truth_boxes = np.array(boxes)
        self._ground_truth_labels = np.array(labels)

    def _convert_yolo_to_xyxy(self, boxes, img_shape):
        for i, box in enumerate(boxes):
            # Convert to x1,y1,x2,y2
            x, y, w, h = box
            x1 = int((x - w / 2) * img_shape[1])
            x2 = int((x + w / 2) * img_shape[1])
            y1 = int((y - h / 2) * img_shape[0])
            y2 = int((y + h / 2) * img_shape[0])
            boxes[i] = np.array([x1, y1, x2, y2])

    def _nms(
        self,
        bboxes: list[np.ndarray],
        scores: list[float],
        labels: list[int],
        num_classes: int,
    ) -> tuple[list[np.ndarray], list[float], list[int]]:
        """Performs non-maximum suppression on the given bboxes, scores, and labels.

        Args:
            bboxes (list[np.ndarray]):
            A list of NumPy arrays with the shape [x1,y1,x2,y2].
            scores (list[float]): A list of confidence scores for each bbox,
            in the same order as bboxes.
            labels (list[int]): A list of label indexes for each bbox,
            in the same order as bboxes.
            num_classes (int): The number of classes

        Returns:
            tuple[list[np.ndarray], list[float], list[int]]:
            A tuple (bboxes, scores, labels) with the same format as the input.
        """

        nms = PPYoloEPostPredictionCallback(
            score_threshold=self.conf,
            nms_top_k=1000,
            max_predictions=300,
            nms_threshold=self.iou,
        )
        # convert to tensor first
        bboxes = torch.from_numpy(np.asarray(bboxes))  # [N, 4]
        scores = torch.from_numpy(np.asarray(scores))  # [N]
        labels = torch.from_numpy(np.asarray(labels))  # [N]

        # convert boxes to torch tensor shaped [B, N, 4]
        # (B is one, only one BBox per anchor)
        bboxes = bboxes.unsqueeze(0)  # [1, N, 4]

        # convert scores and labels to torch tensor shaped [B, N, C]
        ## Make labels into a one-hot vector
        labels = torch.nn.functional.one_hot(labels, num_classes=num_classes)  # [N, C]
        ## By multiplying every 1 in labels with the confidence score
        ## Get get a single one-hot vector to represent both labels and scores
        scores = (labels.T * scores).T
        ## Finally, make B = 1 explicit
        scores = scores.unsqueeze(0)  # [1, N, C]

        # Get results formatted as list[tensor[N, 6]] with len = B
        # namely [x1,y1,x2,y2, conf, label_idx] x N
        result = nms.forward(((bboxes, scores), None), "")[0]

        SWLOG.info(f"NMS Removed {bboxes.shape[1] - result.shape[0]} bboxes")

        # split the results into bboxes, scores, and labels
        bboxes = list(result[:, :4].numpy())
        scores = result[:, 4].numpy().tolist()
        labels = result[:, 5].numpy().astype(int).tolist()

        return bboxes, scores, labels

    def _norm_bbox(
        self, bbox: np.ndarray, i: float, j: float, shape, stride
    ) -> np.ndarray:
        x1 = j * stride
        y1 = i * stride

        bbox = bbox.copy()
        bbox[0] = max(bbox[0] + x1, 0)
        bbox[1] = max(bbox[1] + y1, 0)
        bbox[2] = min(bbox[2] + x1, shape[0])
        bbox[3] = min(bbox[3] + y1, shape[1])

        return bbox

    def _bbox_on_edge(self, bbox) -> bool:
        # check if bbox is on the overlap
        left = bbox[0] <= self._edge_threshold
        top = bbox[1] <= self._edge_threshold
        right = bbox[2] >= self.window_size - self._edge_threshold
        bottom = bbox[3] >= self.window_size - self._edge_threshold

        SWLOG.debug(
            f"bbox is at top: {top}, bottom: {bottom}, left: {left}, right: {right}"
        )
        return (top, bottom, left, right)

    def _get_shared_edges(self, i, j, shape):
        # chek which side _ground_truth_labelswindow shares with other windows
        # top when i != 0 and bottom when i != shape[0] - 1
        # left when j != 0 and right when j != shape[1] - 1
        top = i > 0
        bottom = i < shape[0] - 1
        left = j > 0
        right = j < shape[1] - 1

        SWLOG.debug(
            f"window shares top: {top}, bottom: {bottom}, left: {left}, right: {right}"
        )
        return (top, bottom, left, right)

    def _tuple_and(self, a: tuple[bool], b: tuple[bool]):
        """computes the and of two tuples of bools.

        Args:
            a (tuple[bool]): tuple of bools of length n
            b (tuple[bool]): tuple of bools of length n

        Returns:
            tuple[bool]: tuple where each element is the and of the corresponding elements in a and b

        Raises:
            AssertionError: if the length of a and b are not the same
        """
        assert len(a) == len(b)
        return tuple([a[i] and b[i] for i in range(len(a))])

    def _process_window(
        self, windows, class_names, labels, confs, bboxes, i, j, image, stride
    ):
        SWLOG.debug(f"-----------Processing window at ({i}, {j})")
        win: ImageDetectionPrediction = windows[i, j]
        class_names.update(win.class_names)
        pred = win.prediction

        if self.ground_truth_path is not None:
            # Get metrics for the window
            metrics, values = Metrics.get_metrics(
                pred, self._gt_windows[i][j], self._gt_labels[i][j]
            )

            SWLOG.debug(f"Window {i}, {j} has {len(pred.labels)} predictions")
            SWLOG.debug(
                f"Window {i}, {j} has {len(self._gt_labels[i][j])} ground truth boxes"
            )
            tp, fp, fn = values
            tp, fp, fn = tp[0.5]["all"], fp[0.5]["all"], fn[0.5]["all"]
            SWLOG.debug(
                f"Window {i}, {j} has {len(tp)} true positives, {len(fp)} false positives, and {len(fn)} false negatives"
            )

            Metrics.print_metrics(metrics)

        SWLOG.debug("-" * 20)
        SWLOG.debug(f"Window {i}, {j} has {len(pred.labels)} predictions")

        shared_edges = self._get_shared_edges(i, j, windows.shape)

        for k in range(len(pred.labels)):
            labels.append(win.class_names[int(pred.labels[k])])
            bbox = pred.bboxes_xyxy[k]
            # Check if the bbox is on the edge of the window
            conf = pred.confidence[k]

            SWLOG.debug("-" * 10)
            SWLOG.debug(
                f"k: {k} ({win.class_names[int(pred.labels[k])]}), bbox: {bbox.tolist()}, conf: {conf}"
            )
            if any(self._tuple_and(shared_edges, self._bbox_on_edge(bbox))):
                conf = conf - self.on_edge_penalty * conf
                conf = max(conf, self.conf)
                SWLOG.debug(f"On Edge! New after penalty: {conf}")
            confs.append(np.float32(conf))

            bboxes.append(
                self._norm_bbox(
                    bbox,
                    i,
                    j,
                    image.shape,
                    stride,
                )
            )

    def _combine_windows(
        self,
        image: np.ndarray,
        windows: np.ndarray,  # [N, N, ImageDetectionPrediction]
        stride: int,
        shape=None,
    ) -> ImageDetectionPrediction:
        SWLOG.debug("Combining windows...")
        labels = []
        confs = []
        bboxes = []
        class_names = set()

        if shape and shape != image.shape:
            image = self._reshape_image(image, shape)

        SWLOG.debug(f"shape: {windows.shape}")
        t = tqdm(total=windows.size, desc="Combining windows")
        for i, j in np.ndindex(windows.shape):
            t.update(1)
            t.set_postfix_str(f"Window {i}, {j}")
            self._process_window(
                windows,
                class_names,
                labels,
                confs,
                bboxes,
                i,
                j,
                image,
                stride,
            )

        class_names = self._update_labels(class_names, labels)

        bboxes, confs, labels = self._nms(bboxes, confs, labels, self.num_classes)

        pred = self._create_prediction(bboxes, confs, labels, image)
        return ImageDetectionPrediction(
            image=image, prediction=pred, class_names=class_names
        )

    def _reshape_image(self, image, shape):
        SWLOG.debug(f"Reshaping image from {image.shape} to {shape}")
        return image[: shape[0], : shape[1], :]

    def _update_labels(self, class_names, labels):
        SWLOG.debug("Updating labels to their index")
        class_names = list(class_names)
        for i in range(len(labels)):
            labels[i] = (
                self._classes.index(labels[i])
                if self.ground_truth_path
                else class_names.index(labels[i])
            )
        return class_names

    def _create_prediction(self, img_bboxes, img_confidence, img_labels, image):
        return DetectionPrediction(
            bboxes=np.asarray(img_bboxes),
            bbox_format="xyxy",
            confidence=np.asarray(img_confidence),
            labels=np.asarray(img_labels),
            image_shape=image.shape,
        )

    def _predict_windows(self, windows: np.ndarray) -> np.ndarray:
        # In order to avoid re-initializing the model pipeline every time,
        # we pass a list of images instead of a single image at a time

        # Turn it from 2D array of images into 1D list of images
        w = list(np.reshape(windows, (-1, *windows.shape[2:])))

        preds = self.model.predict(w, conf=self.conf, iou=self.iou)
        preds.save("test_crops.png")

        # Reshape back to 2D array of ImagePredictions
        # TODO: no need to reshape, just add i and j to the array
        preds = preds._images_prediction_lst
        preds = np.reshape(np.asarray(preds), windows.shape[:2])
        return preds

    def _get_windows(self, image: np.ndarray) -> np.ndarray:
        img_shape = image.shape

        if img_shape[0] <= self.window_size and img_shape[1] <= self.window_size:
            # Reshape the array to (1, 1, window_size, window_size, 3)
            SWLOG.info("Image is smaller than window size, using whole image")
            return np.reshape(image, (1, 1, *img_shape))

        stride = self.window_size - self._overlap_size
        window_shape = (self.window_size, self.window_size, 3)

        SWLOG.debug(
            f"Getting windows with size {self.window_size}px,"
            f"overlapping by {stride}px"
        )

        pad_x = self.window_size - (img_shape[0] % stride)
        pad_y = self.window_size - (img_shape[1] % stride)
        SWLOG.debug(f"Padding image by {pad_x}x{pad_y} pixels")
        img = np.pad(
            image, ((0, pad_x), (0, pad_y), (0, 0)), mode="constant", constant_values=0
        )

        windows = np.lib.stride_tricks.sliding_window_view(
            img, window_shape, writeable=True
        )

        # Remove windows so that remaining ones overlap only by [overlap]
        windows = windows[::stride, ::stride, :, :, :, :]
        windows = windows.squeeze()

        # If the windows at the edges are only 30% of the window size, remove them
        if img_shape[0] % stride < self.window_size * self.window_overlap:
            SWLOG.debug("Removing last row of windows")
            windows = windows[:-1, :, :, :, :]
        if img_shape[1] % stride < self.window_size * self.window_overlap:
            SWLOG.debug("Removing last column of windows")
            windows = windows[:, :-1, :, :, :]

        if self.ground_truth_path:
            SWLOG.debug("Getting ground truth windows")
            # If ground truth is provided, make ground truth windows
            # gt_windows [num_windows_x, num_windows_y, num_bboxes, x1, y1, x2, y2].
            # We still don't know which BB belongs to which window, so we'll do that
            self._gt_windows = [
                [[] for _ in range(windows.shape[1])] for _ in range(windows.shape[0])
            ]
            # gt_labels [num_windows_x, num_windows_y, num_bboxes, label]
            self._gt_labels = [
                [[] for _ in range(windows.shape[1])] for _ in range(windows.shape[0])
            ]
            for k, (bbox, label) in enumerate(
                zip(self._ground_truth_boxes, self._ground_truth_labels)
            ):
                num_windows = 0
                for i, j in np.ndindex(windows.shape[:2]):
                    # for every window, get the relative window version of the bbox
                    rel_bbox = self._image_to_window_bbox(bbox, i, j, stride)
                    window_bbox = np.asarray([0, 0, self.window_size, self.window_size])
                    # If bbox is not out of bounds of the window with minimum 10% overlap
                    if self._box_in_window(rel_bbox, window_bbox):
                        rel_bbox = np.asarray([
                            max(rel_bbox[0], 0),
                            max(rel_bbox[1], 0),
                            min(rel_bbox[2], self.window_size),
                            min(rel_bbox[3], self.window_size),
                        ])
                        self._gt_windows[i][j].append(rel_bbox)
                        self._gt_labels[i][j].append(label)
                        num_windows += 1
                assert num_windows > 0, f"Could not find window for bbox {k}"

        return windows, stride

    def _box_in_window(self, bbox, window, vis=0.1):
        # Return true if the intersection of the bbox and window is at least vis of the bbox area
        return Metrics.intersection(bbox, window) > vis * Metrics.box_area(bbox)

    def _image_to_window_bbox(self, bbox, i, j, stride):
        # Get the relative window version of the bbox
        window_bbox = np.asarray([
            bbox[0] - i * stride,
            bbox[1] - j * stride,
            bbox[2] - i * stride,
            bbox[3] - j * stride,
        ])
        return window_bbox

    def _image_from_path(self, image: str) -> np.ndarray:
        SWLOG.debug(f"Loading {image}")
        if image.endswith(".psd") or image.endswith(".psb"):
            return np.asarray(PSDImage.open(image).composite().convert("RGB"))
        else:
            return np.asarray(Image.open(image).convert("RGB"))

    @staticmethod
    def _resize_image(img: ImageDetectionPrediction, size: int):
        SWLOG.debug(f"Resizing image to {size}px")
        # see which dimension is larger, and resize it to size and scale the other
        # dimension accordingly
        import cv2

        # Convert all bounding boxes to percentage of the image sizes
        img.prediction.bboxes_xyxy = [
            np.asarray([
                bbox[0] / img.image.shape[1],
                bbox[1] / img.image.shape[0],
                bbox[2] / img.image.shape[1],
                bbox[3] / img.image.shape[0],
            ])
            for bbox in list(img.prediction.bboxes_xyxy)
        ]

        max_dim = np.argmax(img.image.shape[:2])
        scale = size / img.image.shape[max_dim]
        new_dims = [int(dim * scale) for dim in img.image.shape[:2]]
        img.image = cv2.resize(img.image, tuple(new_dims[::-1]))

        # reconvert bboxes
        img.prediction.bboxes_xyxy = np.asarray(
            [
                np.asarray([
                    bbox[0] * img.image.shape[1],
                    bbox[1] * img.image.shape[0],
                    bbox[2] * img.image.shape[1],
                    bbox[3] * img.image.shape[0],
                ])
                for bbox in list(img.prediction.bboxes_xyxy)
            ]
        )

        return img

    @staticmethod
    def save_visualisation(img: ImageDetectionPrediction):
        SWLOG.debug("Visualizing prediction")
        # if the maximum dimension is greater than 1000px, resize it
        if max(img.image.shape) > MAX_IMAGE_SIZE:
            img = SlidingWindowDetect._resize_image(img, MAX_IMAGE_SIZE)
        img.save("test.png")

    def print_metrics(self, pred):
        pred = pred.prediction
        metrics, values = Metrics.get_metrics(
            pred, self._ground_truth_boxes, self._ground_truth_labels
        )
        SWLOG.debug("-" * 40)
        SWLOG.debug("Metrics")
        SWLOG.debug("-" * 40)
        Metrics.print_metrics(metrics)

    def detect(self, image: str):
        SWLOG.debug(f"Detecting {image}")
        image = self._image_from_path(image)
        SWLOG.debug(f"Image shape: {image.shape}")
        SWLOG.debug(f"Image size: {image.size}")
        SWLOG.debug(f"Image dtype: {image.dtype}")
        SWLOG.debug(f"Image max: {image.max()}")
        SWLOG.debug(f"Image min: {image.min()}")
        if self.ground_truth_path:
            self._convert_yolo_to_xyxy(self._ground_truth_boxes, image.shape)
        windows, stride = self._get_windows(image)
        SWLOG.debug(
            f"Number of windows: {windows.shape[0] * windows.shape[1]}"
            f"({windows.shape[0]}x{windows.shape[1]})"
        )
        predictions = self._predict_windows(windows)
        img_prediction = self._combine_windows(
            image, predictions, stride, shape=image.shape
        )
        return img_prediction


def main():
    args = parse_args()
    SWLOG.setLevel(args.loglevel)

    sw = SlidingWindowDetect(
        model_path=args.model_path,
        num_classes=args.num_classes,
        window_size=args.window_size,
        window_overlap=args.window_overlap,
        conf=args.conf,
        iou=args.iou,
        on_edge_penalty=args.on_edge_penalty,
        edge_threshold=args.edge_threshold,
        ground_truth_path=args.ground_truth_path,
        dataset_yaml_path=args.dataset_yaml,
    )

    pred = sw.detect(args.image_path)

    sw.print_metrics(pred)

    SlidingWindowDetect.save_visualisation(pred)


if __name__ == "__main__":
    with logging_redirect_tqdm():
        main()
