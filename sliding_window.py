# Adapted from YOLO-NAS tutorial by Harpreet:
# https://colab.research.google.com/drive/10N6NmSMCiRnFlKV9kaIS_z3pk0OI1xKC?usp=sharing
# https://colab.research.google.com/drive/1q0RmeVRzLwRXW-h9dPFSOchwJkThUy6d#scrollTo=m0SkK3bjMOqH

import argparse
import logging
import os
from dataclasses import dataclass
from typing import Optional
import copy

import coco_evaluator as coco
import numpy as np
import torch
from PIL import Image
from psd_tools import PSDImage
from super_gradients.common.object_names import Models
from super_gradients.training import models
from super_gradients.training.models.detection_models.pp_yolo_e.post_prediction_callback import PPYoloEPostPredictionCallback  # noqa: E501; noqa: E501
from super_gradients.training.utils.predict import (  # noqa: E501
    DetectionPrediction,
    ImageDetectionPrediction,
)

SWLOG = logging.getLogger("SlidingWindow")
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
SWLOG.addHandler(handler)
SWLOG.propagate = False

os.environ["CUDA_LAUNCH_BLOCKINGs"] = "1"

MAX_IMAGE_SIZE = 10000


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

    debug: bool = False

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
            self._gt_classes = data["names"]
            SWLOG.debug(f"Classes ({type(self._gt_classes)}): {self._gt_classes}")

    def _load_ground_truth(self):
        """Load the ground truth file and set the
        ground truth boxes to _rel_gt_boxes and labels to _gt_labels
        """
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
        self._rel_gt_boxes = np.array(boxes)
        self._gt_labels = np.array(labels)

    def _set_coco_gt(self, image):
        self._set_coco_bboxes(image.shape)
        self._set_coco_img(image)

    def _set_coco_img(self, image: np.ndarray):
        SWLOG.debug("Setting coco image")
        bboxes = [
            box.get_absolute_bounding_box(format=coco.BBFormat.XYX2Y2)
            for box in self._coco_gt_bboxes
        ]
        bboxes = np.asarray(bboxes)
        confs = np.ones(len(bboxes))
        pred = DetectionPrediction(
            bboxes, "xyxy", confs, self._gt_labels, image.shape[:2]
        )
        self._gt_image_det = ImageDetectionPrediction(image, pred, self._gt_classes)

    def _set_coco_bboxes(self, img_shape):
        SWLOG.debug("Setting COCO bboxes for ground truth given image shape")
        self._coco_gt_bboxes = coco.BoundingBox.from_list(
            "img",
            self._gt_labels,
            self._rel_gt_boxes,
            type_coordinates=coco.CoordinatesType.RELATIVE,
            img_size=img_shape[:2][::-1],
            bb_type=coco.BBType.GROUND_TRUTH,
            bb_format=coco.BBFormat.YOLO,
        )

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
            multi_label_per_box=False,
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
        return (top, bottom, left, right)

    def _get_shared_edges(self, i, j, shape):
        # check which side the window shares with other windows
        # top when i != 0 and bottom when i != shape[0] - 1
        # left when j != 0 and right when j != shape[1] - 1
        top = i > 0
        bottom = i < shape[0] - 1
        left = j > 0
        right = j < shape[1] - 1
        return (top, bottom, left, right)

    def _tuple_and(self, a: tuple[bool], b: tuple[bool]):
        """computes the and of two tuples of bools.

        Args:
            a (tuple[bool]): tuple of bools of length n
            b (tuple[bool]): tuple of bools of length n

        Returns:
            tuple[bool]: tuple where each element is the AND of the
            corresponding elements in a and b

        Raises:
            AssertionError: if the length of a and b are not the same
        """
        assert len(a) == len(b)
        return tuple([a[i] and b[i] for i in range(len(a))])

    def _process_window(
        self, windows, class_names: set, labels, confs, bboxes, i, j, image, stride
    ):
        win: ImageDetectionPrediction = windows[i, j]
        class_names.update(win.class_names)
        pred = win.prediction

        if self.ground_truth_path is not None:
            metrics = self.get_metrics_imgpred(
                win, name=f"{i}_{j}", gt=self._gt_windows_boxes[i][j]
            )
            ap = metrics["summary"]["AP"]
            if not np.isnan(ap):
                SWLOG.debug(f"AP for window {i}, {j}: {ap}")

        shared_edges = self._get_shared_edges(i, j, windows.shape)

        for k in range(len(pred.labels)):
            labels.append(win.class_names[int(pred.labels[k])])
            bbox = pred.bboxes_xyxy[k]
            conf = pred.confidence[k]

            # Check if the bbox is on the edge of the window
            if any(self._tuple_and(shared_edges, self._bbox_on_edge(bbox))):
                conf = conf - self.on_edge_penalty * conf
                conf = max(conf, self.conf)
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
        for i, j in np.ndindex(windows.shape):
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

    def _update_labels(self, class_names: set, labels):
        SWLOG.debug("Updating labels to their index")

        if self.ground_truth_path is not None:
            assert (
                len(class_names.difference(self._gt_classes)) == 0
            ), "Some classes in the ground truth are not in the model's classes"
            class_names = self._gt_classes
        else:
            class_names = list(class_names)
        for i in range(len(labels)):
            labels[i] = class_names.index(labels[i])

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

        if self.debug:
            preds.save("windows_preds")

        # Get internal list of ImageDetectionPredictions
        preds = preds._images_prediction_lst

        # Reshape back to 2D array of ImagePredictions
        preds = np.reshape(np.asarray(preds), windows.shape[:2])

        # Compute metrics across all windows
        if self.ground_truth_path is not None:
            # If there is a ground truth, make sure the labels indexes match the ground truth

            pred_coco_boxes = []
            for i, j in np.ndindex(preds.shape[:2]):
                # Get the prediction label_idxs and classes
                p: ImageDetectionPrediction = preds[i, j]
                p_label_idxs = p.prediction.labels
                p_classes = p.class_names
                # Save the order of the classes
                p_label_classes = [p_classes[int(l_idx)] for l_idx in p_label_idxs]
                # update the class names to match the ground truth
                p.class_names = self._gt_classes
                # update the labels to match the ground truth
                p_label_idxs = [self._gt_classes.index(c) for c in p_label_classes]
                p.prediction.labels = np.asarray(p_label_idxs)

                img_coco_boxes = coco.BoundingBox.from_image_detection_prediction(
                    f"{i}_{j}", p
                )
                pred_coco_boxes.extend(img_coco_boxes)
            gt_coco_boxes = []
            for i in self._gt_windows_boxes:
                for j in i:
                    gt_coco_boxes.extend(j)
            metrics = self.get_metrics(gt_coco_boxes, pred_coco_boxes)

            SWLOG.debug(f"---METRICS FOR ALL WINDOWS---")
            self.print_metrics(metrics)

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

        if self.ground_truth_path is not None:
            SWLOG.debug("Getting ground truth for windows")
            # If ground truth is provided, make ground truth windows
            # gt_windows [num_windows_x, num_windows_y, num_bboxes, x1, y1, x2, y2].
            # We still don't know which BB belongs to which window, so we'll do that
            self._gt_windows_boxes = [
                [[] for _ in range(windows.shape[1])] for _ in range(windows.shape[0])
            ]
            # gt_labels [num_windows_x, num_windows_y, num_bboxes, label]
            # self._gt_windows_boxes_labels = [
            #     [[] for _ in range(windows.shape[1])] for _ in range(windows.shape[0])
            # ]

            for k, box in enumerate(self._coco_gt_bboxes):
                box = coco.BoundingBox.clone(box)
                bbox: tuple = box.get_absolute_bounding_box(format=coco.BBFormat.XYX2Y2)

                num_windows = 0
                for i, j in np.ndindex(windows.shape[:2]):
                    # for every window, get the relative window version of the bbox
                    rel_bbox = self._image_to_window_bbox(bbox, i, j, stride)
                    window_bbox = np.asarray([0, 0, self.window_size, self.window_size])
                    # If bbox is not out of bounds of the
                    # window with minimum 10% overlap
                    if self._box_in_window(rel_bbox, window_bbox):
                        rel_bbox = (
                            max(rel_bbox[0], 0),
                            max(rel_bbox[1], 0),
                            min(rel_bbox[2], self.window_size),
                            min(rel_bbox[3], self.window_size),
                        )
                        box.set_image_name(f"{i}_{j}")
                        box.set_image_size((self.window_size, self.window_size))
                        box.set_coordinates(
                            rel_bbox, type_coordinates=coco.CoordinatesType.ABSOLUTE
                        )
                        self._gt_windows_boxes[i][j].append(box)
                        # self._gt_windows_boxes_labels[i][j].append(label)
                        num_windows += 1
                assert num_windows > 0, "Could not find window for bbox"

        return windows, stride

    def _intersection(self, box1, box2):
        x1, y1 = np.maximum(box1[:2], box2[:2])
        x2, y2 = np.minimum(box1[2:], box2[2:])
        return np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    def _box_area(self, box):
        return (box[2] - box[0]) * (box[3] - box[1])

    def _box_in_window(self, bbox, window, vis=0.1):
        # Return true if the intersection of the
        # bbox and window is at least vis of the bbox area
        return self._intersection(bbox, window) > vis * self._box_area(bbox)

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

        max_dim = np.argmax(img.image.shape[:2])
        scale = size / img.image.shape[max_dim]
        new_dims = [round(dim * scale) for dim in img.image.shape[:2]]
        img.image = cv2.resize(img.image, tuple(new_dims[::-1]))

        img.prediction = copy.deepcopy(img.prediction)
        img.prediction.bboxes_xyxy *= scale

        return img

    def save_visualisation(self, img):
        self._save_visualisation(img, title="prediction")
        if self.ground_truth_path is not None:
            self._save_visualisation(self._gt_image_det, title="ground_truth")

    def _save_visualisation(
        self, img: ImageDetectionPrediction, title: str = "prediction"
    ):
        SWLOG.debug(f"Saving visualisation: {title}")
        # if the maximum dimension is greater than 1000px, resize it
        if max(img.image.shape) > MAX_IMAGE_SIZE:
            img = SlidingWindowDetect._resize_image(img, MAX_IMAGE_SIZE)
        img.save(f"{title}.png")

    def _process_metrics(self, metrics):
        # add some extra metrics
        for _, metric in metrics.items():
            if metric["TP"] is None:
                continue
            num = len(metric["precision"])
            metric["num_predictions"] = num
            # total positives is the number of ground truth bboxes
            metric["num_ground_truths"] = metric["total positives"]
            metric["FN"] = metric["total positives"] - metric["TP"]
            metric["accuracy"] = metric["TP"] / num if num > 0 else None
            metric["f1"] = (
                (
                    2
                    * metric["precision"][-1]
                    * metric["recall"][-1]
                    / (metric["precision"][-1] + metric["recall"][-1])
                )
                if num > 0 and metric["precision"][-1] + metric["recall"][-1] > 0
                else None
            )

    def get_metrics(self, gt, pred_bboxes):
        metrics = coco.get_coco_metrics(gt, pred_bboxes)
        # { class_id : { class, precision, recall, AP, interpolated precision,
        #               interpolated recall, total positives, TP, FP }
        # }
        self._process_metrics(metrics)
        summary = coco.get_coco_summary(gt, pred_bboxes)
        # { AP, AP50, AP75, APsmall, APmedium, APlarge, AR1,
        #   AR10, AR100, ARsmall, ARmedium, ARlarge }
        return {
            "by_class": metrics,
            "summary": summary,
        }

    def get_metrics_imgpred(self, img_pred, name="img", gt=None):
        """Get metrics for a single image prediction

        Args:
            img_pred (ImageDetectionPrediction): The image prediction

        Returns:
            dict: A dictionary with the following format:
                    'by_class':
                        (for every class)
                        {Class ID}:
                            'class', 'precision', 'recall', 'AP', s'interpolated precision','interpolated recall', 'total positives', 'TP', 'FP', 'f1', 'accuracy'
                    'summary':
                        'AP', 'AP50', 'AP75', 'APsmall', 'APmedium', 'APlarge', 'AR1', 'AR10', 'AR100', 'ARsmall', 'ARmedium', 'ARlarge'}

        """  # noqa: E501
        if gt is None:
            gt = self._coco_gt_bboxes
        assert gt is not None, "No coco ground truth set! Are they set?"

        pred_bboxes = coco.BoundingBox.from_image_detection_prediction(name, img_pred)

        return self.get_metrics(gt, pred_bboxes)

    def print_metrics(self, metrics):
        metrics_to_skip = [
            "interpolated precision",
            "interpolated recall",
            "total positives",
            "APsmall",
            "APmedium",
            "APlarge",
            "ARsmall",
            "ARmedium",
            "ARlarge",
            "class",
        ]

        def print_value(k, v):
            if k not in metrics_to_skip:
                print(
                    f"| {k}: {v[-1] if isinstance(v, np.ndarray) and len(v) > 0 else v}"
                )

        def print_values(items):
            nones = []
            for key, value in items:
                if value is None:
                    nones.append(key)
                else:
                    print_value(key, value)
            if len(nones) > 0:
                print(f"| None for {', '.join(nones)}")

        def print_title(title):
            print("-" * 50)
            print(f"| {title}")
            print("-" * 50)

        print_title("Summary")
        print_values(metrics["summary"].items())

        print_title("Metrics by Class")
        for class_id, class_metrics in metrics["by_class"].items():
            print(f"| Class ID: {class_id}")
            print_values(class_metrics.items())
            print("-" * 50)

    def detect(self, image: str):
        SWLOG.debug(f"Detecting {image}")
        image = self._image_from_path(image)
        if self.ground_truth_path is not None:
            self._set_coco_gt(image)
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
    parser.add_argument("-d", "--debug", type=bool, help="Debug mode", default=False)
    return parser.parse_args()


def main():
    args = parse_args()
    SWLOG.setLevel(logging.DEBUG)

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
        debug=args.debug,
    )

    img_pred = sw.detect(args.image_path)
    sw.print_metrics(sw.get_metrics_imgpred(img_pred))
    sw.save_visualisation(img_pred)


if __name__ == "__main__":
    main()
