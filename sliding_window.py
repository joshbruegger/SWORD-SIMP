# Adapted from YOLO-NAS tutorial by Harpreet:
# https://colab.research.google.com/drive/10N6NmSMCiRnFlKV9kaIS_z3pk0OI1xKC?usp=sharing
# https://colab.research.google.com/drive/1q0RmeVRzLwRXW-h9dPFSOchwJkThUy6d#scrollTo=m0SkK3bjMOqH

import argparse
import os
import numpy as np
from psd_tools import PSDImage
from PIL import Image


import torch

from super_gradients.common.object_names import Models
from super_gradients.training import models
from super_gradients.training.models.detection_models.pp_yolo_e.post_prediction_callback import PPYoloEPostPredictionCallback  # noqa: E501
from super_gradients.training.utils.predict import ImageDetectionPrediction, DetectionPrediction  # noqa: E501

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=str, help="Path to data directory to test")
    parser.add_argument(
        "--model_path",
        "-m",
        type=str,
        required=True,
        help="Path to checkpoint",
    )
    parser.add_argument(
        "--num_classes", "-n", type=int, required=True, help="Number of classes"
    )
    return parser.parse_args()


def print_welcome(args):
    padding = 140
    print("\n\n")
    print(" SWORD-SIMP SLIDING WINDOW ".center(padding, "8"))
    print(f" Image path: {args.image_path} ".center(padding, "8"))
    print(f" Model path: {args.model_path} ".center(padding, "8"))
    print(f" Number of classes: {args.num_classes} ".center(padding, "8"))
    print("\n\n")


class SlidingWindowDetect:

    def __init__(
        self,
        *,
        model_path,
        num_classes,
        window_size=640,
        window_overlap=0.5,
        conf=0.6,  # Confidence threshold for predictions, below which predictions are discarded
        iou=0.3,  # Max overlap between bboxes before one is removed during NMS
        on_edge_penalty=0.3,  # Penalty for bboxes on the edge of the image (in confidence is removed, 0.3 = 30%)
    ):
        self.model_path = os.path.abspath(model_path)
        self.model_name = Models.YOLO_NAS_L
        self.window_size = window_size
        self.window_overlap = window_overlap
        self.conf = conf
        self.iou = iou
        self.num_classes = num_classes
        self.on_edge_penalty = on_edge_penalty

        assert 0 <= self.window_overlap < 1

        # Print out the all local variables
        print("Configuration:")
        for key, value in vars(self).items():
            print(f"{key}: {value}")

        self._get_model()

    def _get_model(self):
        self.model = models.get(
            self.model_name,
            checkpoint_path=self.model_path,
            num_classes=self.num_classes,
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)

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

        # #############################
        # # FOR TESTING, MAKE FAUX BBOXES, SCORES, AND LABELS
        # # 5 bboxes, 6 classes, random scores

        # # Generate fake bboxes making sure they're valid (x1 < x2, y1 < y2)
        # # bboxes should be floats
        # bboxes = []
        # for i in range(5):
        #     x1 = np.random.rand() * 100
        #     x2 = np.random.rand() * 100
        #     y1 = np.random.rand() * 100
        #     y2 = np.random.rand() * 100
        #     bboxes.append(np.asarray([x1, y1, x2, y2]))

        # # Generate fake scores between 0 and 1 (list)
        # scores = [np.random.rand() for i in range(5)]
        # # Generate fake labels
        # labels = [np.random.randint(0, 6) for i in range(5)]
        # #############################

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

        print(bboxes)
        print(scores)

        # Get results formatted as list[tensor[N, 6]] with len = B
        # namely [x1,y1,x2,y2, conf, label_idx] x N
        result = nms.forward(((bboxes, scores), None), "")[0]

        # for testing, calculate which bboxes were removed
        print(f"Removed {bboxes.shape[1] - result.shape[0]} bboxes")

        # split the results into bboxes, scores, and labels
        bboxes = list(result[:, :4].numpy())
        scores = result[:, 4].numpy().tolist()
        labels = result[:, 5].numpy().astype(int).tolist()

        return bboxes, scores, labels

    def _normalize_bbox(
        self, bbox: np.ndarray, i: float, j: float, shape, stride
    ) -> np.ndarray:
        x1 = j * stride
        y1 = i * stride

        bbox = bbox.copy()
        on_edge = False
        bbox[0] += x1
        bbox[1] += y1
        bbox[2] += x1
        bbox[3] += y1

        if bbox[0] <= 0 or bbox[1] <= 0 or bbox[2] >= shape[0] or bbox[3] >= shape[1]:
            on_edge = True

        bbox[0] = max(bbox[0], 0)
        bbox[1] = max(bbox[1], 0)
        bbox[2] = min(bbox[2], shape[0])
        bbox[3] = min(bbox[3], shape[1])

        return bbox, on_edge

    def _process_window(
        self, w, class_names, labels, confs, bboxes, i, j, image, stride
    ):
        class_names.update(w.class_names)
        p = w.prediction

        for k in range(len(p.labels)):
            labels.append(w.class_names[int(p.labels[k])])
            norm, on_edge = self._normalize_bbox(
                p.bboxes_xyxy[k],
                i,
                j,
                image.shape,
                stride,
            )
            conf = p.confidence[k]
            if on_edge:
                print(f"On edge: {w.class_names[int(p.labels[k])]}")
                print(f"Confidence: {conf}")
                conf = conf - self.on_edge_penalty * conf
                conf = max(conf, self.conf)
                print(f"New confidence: {conf}")
            confs.append(np.float32(conf))
            bboxes.append(norm)

    def _combine_windows(
        self,
        image: np.ndarray,
        windows: np.ndarray,
        stride: int,
        shape=None,
    ) -> ImageDetectionPrediction:
        labels = []
        confs = []
        bboxes = []
        class_names = set()

        if shape and shape != image.shape:
            image = self._reshape_image(image, shape)

        for i, j in np.ndindex(windows.shape[:2]):
            self._process_window(
                windows[i][j],
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

        # bboxes, confs, labels = self._nms(bboxes, confs, labels, self.num_classes)

        pred = self._create_prediction(bboxes, confs, labels, image)
        return ImageDetectionPrediction(
            image=image, prediction=pred, class_names=class_names
        )

    def _reshape_image(self, image, shape):
        return image[: shape[0], : shape[1], :]

    def _update_labels(self, img_class_names, img_labels):
        img_class_names = list(img_class_names)
        for i in range(len(img_labels)):
            img_labels[i] = img_class_names.index(img_labels[i])
        return img_class_names

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
        preds = preds._images_prediction_lst
        preds = np.reshape(np.asarray(preds), windows.shape[:2])
        return preds

    def _get_windows(self, image: np.ndarray) -> np.ndarray:
        img_shape = image.shape

        if img_shape[0] <= self.window_size and img_shape[1] <= self.window_size:
            # Reshape the array to (1, 1, window_size, window_size, 3)
            return np.reshape(image, (1, 1, *img_shape))

        stride = int(self.window_size * (1 - self.window_overlap))

        print(
            f"Getting windows  with size {self.window_size}px,"
            f"overlapping by this many pixels: {stride}"
        )

        pad_x = self.window_size - (img_shape[0] % stride)
        pad_y = self.window_size - (img_shape[1] % stride)
        img = np.pad(
            image, ((0, pad_x), (0, pad_y), (0, 0)), mode="constant", constant_values=0
        )

        window_shape = (self.window_size, self.window_size, 3)
        windows = np.lib.stride_tricks.sliding_window_view(
            img, window_shape, writeable=True
        )

        # Remove windows so that remaining ones overlap only by [overlap]
        windows = windows[::stride, ::stride, :, :, :, :]
        windows = windows.squeeze()

        # If the windows at the edges are only 30% of the window size, remove them
        if img_shape[0] % stride < self.window_size * self.window_overlap:
            windows = windows[:-1, :, :, :, :]
        if img_shape[1] % stride < self.window_size * self.window_overlap:
            windows = windows[:, :-1, :, :, :]

        return windows, stride

    def _image_from_path(self, image: str) -> np.ndarray:
        if image.endswith(".psd") or image.endswith(".psb"):
            return PSDImage.open(image).numpy(channel="color")
        else:
            return np.asarray(Image.open(image).convert("RGB"))

    def detect(self, image: str):
        print(f"Detecting {image}")
        image = self._image_from_path(image)
        windows, stride = self._get_windows(image)
        print(
            f"Number of windows: {windows.shape[0] * windows.shape[1]}"
            f"({windows.shape[0]}x{windows.shape[1]})"
        )
        predictions = self._predict_windows(windows)
        img_prediction = self._combine_windows(
            image, predictions, stride, shape=image.shape
        )
        img_prediction.save("test.png")


def main():
    args = parse_args()
    print_welcome(args)

    SlidingWindowDetect(
        model_path=args.model_path,
        num_classes=args.num_classes,
    ).detect(args.image_path)


if __name__ == "__main__":
    main()
