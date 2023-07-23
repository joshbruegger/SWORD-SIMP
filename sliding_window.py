# Adapted from YOLO-NAS tutorial by Harpreet:
# https://colab.research.google.com/drive/10N6NmSMCiRnFlKV9kaIS_z3pk0OI1xKC?usp=sharing
# https://colab.research.google.com/drive/1q0RmeVRzLwRXW-h9dPFSOchwJkThUy6d#scrollTo=m0SkK3bjMOqH

import argparse
import os
import numpy as np
from psd_tools import PSDImage
from PIL import Image
from collections import defaultdict
from dataclasses import dataclass


from torch import cuda
from super_gradients.common.object_names import Models
from super_gradients.training import models
from super_gradients.training.utils.predict import ImageDetectionPrediction, DetectionPrediction

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
    print(" SWORD-SIMP SLIDING WINDOW".center(padding, "8"))
    print(f" Image path: {args.image_path} ".center(padding, "8"))
    print(f" Model path: {args.model_path} ".center(padding, "8"))
    print(f" Number of classes: {args.num_classes} ".center(padding, "8"))
    print("\n\n")


@dataclass
class Window:
    bbox: np.ndarray
    x1: int
    y1: int
    x2: int
    y2: int
    x_idx: int
    y_idx: int
    pred: DetectionPrediction


class SlidingWindowDetect:

    def __init__(
        self,
        *,
        model_path,
        num_classes,
        window_size=1080,
        overlap=0.5,
        conf=0.25,
        iou=0.5,
    ):
        self.model_path = os.path.abspath(model_path)
        self.model_name = Models.YOLO_NAS_L
        self.window_size = window_size
        self.overlap = overlap
        self.conf = conf
        self.iou = iou
        self.num_classes = num_classes

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
        self.model = self.model.to("cuda" if cuda.is_available() else "cpu")

    def _get_sections(self, image: Image):
        width, height = image.size
        windows = []
        stride = int(self.window_size * (1 - self.overlap))
        for i in range(0, width, stride):
            for j in range(0, height, stride):
                crop = image.crop((
                    i,
                    j,
                    min(i + self.window_size, width),
                    min(j + self.window_size, height),
                ))
                windows.append(
                    Window(
                        x1=i,
                        y1=j,
                        x2=i + self.window_size,
                        y2=j + self.window_size,
                        x_idx=i // stride,
                        y_idx=j // stride,
                        pred=self.model.predict(crop, conf=self.conf, iou=self.iou)[0],
                    )
                )
        return windows

    def _normalize_bbox(self, bbox: np.ndarray, x1, y1, x2, y2):
        bbox = bbox.copy()
        bbox[0] += x1
        bbox[1] += y1
        bbox[2] += x2
        bbox[3] += y2
        return bbox

    def _combine_sections(
        self, image: np.ndarray, windows: list[Window]
    ) -> ImageDetectionPrediction:
        img_labels = []
        img_confidence = []
        img_bboxes = []
        img_class_names = None

        for w in windows:
            if img_class_names is None:
                img_class_names = w.pred.class_names
            else:
                assert img_class_names == w.pred.class_names

            for i in range(len(w.pred.labels)):
                img_labels.append(w.pred.class_names[i])
                img_confidence.append(w.pred.confidence[i])
                img_bboxes.append(
                    self._normalize_bbox(
                        w.pred.bboxes_xyxy[i],
                        w.x1,
                        w.y1,
                        w.x2,
                        w.y2,
                    )
                )

        prediction = DetectionPrediction(
            bboxes=img_bboxes,
            bbox_format="xyxy",
            confidence=img_confidence,
            labels=img_labels,
            image_shape=image.shape,
        )
        return ImageDetectionPrediction(image=image, predictions=prediction)

    def _image_from_path(self, image: str):
        if image.endswith(".psd") or image.endswith(".psb"):
            return PSDImage.open(image).composite()
        else:
            return Image.open(image).convert("RGB")

    def detect(self, image: str):
        print(f"Detecting {image}")
        image = self._image_from_path(image)
        test = self.model.predict(image, conf=self.conf, iou=self.iou)[0]
        print(test)
        # self.model.predict(image, conf=self.conf, iou=self.iou)
        # sections = self._get_sections(image)
        # img_prediction = self._combine_sections(image, sections)


def main():
    args = parse_args()
    print_welcome(args)

    SlidingWindowDetect(
        model_path=args.model_path,
        num_classes=args.num_classes,
    ).detect(args.image_path)


if __name__ == "__main__":
    main()
