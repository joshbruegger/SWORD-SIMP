## TAKEN FROM https://github.com/rafaelpadilla/review_object_detection_metrics/blob/2efe66d2c4b89e4fcc64e490d4caced2096f03aa/src/bounding_box.py

from math import isclose
from typing import Optional

from .general_utils import (
    xywh_to_xyxy,
    convert_to_relative_values,
    rel_to_abs,
)

from .enumerators import BBFormat, BBType, CoordinatesType


class BoundingBox:
    """Class representing a bounding box."""

    def __init__(
        self,
        image_name,
        class_id=None,
        coordinates=None,
        type_coordinates=CoordinatesType.ABSOLUTE,
        img_size=None,
        bb_type=BBType.GROUND_TRUTH,
        confidence=None,
        format=BBFormat.XYWH,
    ):
        """ Constructor.

        Parameters
        ----------
            image_name : str
                String representing the name of the image.
            class_id : str
                String value representing class id.
            coordinates : tuple
                Tuple with 4 elements whose values (float) represent coordinates of the bounding \\
                    box.
                The coordinates can be (x, y, w, h)=>(float,float,float,float) or(x1, y1, x2, y2)\\
                    =>(float,float,float,float).
                See parameter `format`.
            type_coordinates : Enum (optional)
                Enum representing if the bounding box coordinates (x,y,w,h) are absolute or relative
                to size of the image. Default:'Absolute'.
            img_size : tuple (optional)
                Image size in the format (width, height)=>(int, int) representinh the size of the
                image of the bounding box. If type_coordinates is 'Relative', img_size is required.
            bb_type : Enum (optional)
                Enum identifying if the bounding box is a ground truth or a detection. If it is a
                detection, the confidence must be informed.
            confidence : float (optional)
                Value representing the confidence of the detected object. If detectionType is
                Detection, confidence needs to be informed.
            format : Enum
                Enum (BBFormat.XYWH or BBFormat.XYX2Y2) indicating the format of the coordinates of
                the bounding boxes.
                BBFormat.XYWH: <left> <top> <width> <height>
                BBFomat.XYX2Y2: <left> <top> <right> <bottom>.
                BBFomat.YOLO: <x_center> <y_center> <width> <height>. (relative)
        """

        self._image_name = image_name
        # self._type_coordinates = type_coordinates
        self._confidence = confidence
        self._class_id = class_id
        # self._format = format
        if bb_type == BBType.DETECTED and confidence is None:
            raise ValueError(
                "For bb_type='Detected', it is necessary to inform the confidence value."
            )
        self._bb_type = bb_type

        if img_size is None:
            self._width_img = None
            self._height_img = None
        else:
            self._width_img = img_size[0]
            self._height_img = img_size[1]

        if format == BBFormat.YOLO:
            assert self._width_img is not None and self._height_img is not None
            # self._format = BBFormat.XYWH
            format = BBFormat.XYWH
            # self._type_coordinates = CoordinatesType.RELATIVE
            type_coordinates = CoordinatesType.RELATIVE

        self.set_coordinates(coordinates, type_coordinates, format, img_size=img_size)

    # Fixed by Josh Bruegger
    def set_coordinates(self, coordinates, type_coordinates, format, img_size=None):
        if type_coordinates == CoordinatesType.RELATIVE and img_size is None:
            raise IOError(
                "Parameter 'img_size' is required. It is necessary to inform the image size."
            )

        # If relative coordinates, convert to absolute values
        if type_coordinates == CoordinatesType.RELATIVE:
            self._width_img = img_size[0]
            self._height_img = img_size[1]

            # assert self._format in [
            assert format in [
                BBFormat.XYWH,
                BBFormat.XYX2Y2,
            ], "for relative coordinates, the format must be XYWH or XYXY"

            # Convert to absolute values
            coordinates = rel_to_abs(coordinates, img_size)

            # if self._format == BBFormat.XYWH:
            if format == BBFormat.XYWH:
                coordinates = xywh_to_xyxy(coordinates)

            self._x, self._y, self._x2, self._y2 = coordinates
            self._w = self._x2 - self._x
            self._h = self._y2 - self._y

        elif type_coordinates == CoordinatesType.ABSOLUTE:
            # if self._format == BBFormat.XYWH:
            if format == BBFormat.XYWH:
                self._h = coordinates[3]
                self._w = coordinates[2]
                self._y = coordinates[1] - self._h / 2.0
                self._x = coordinates[0] - self._w / 2.0
                self._x2 = self._x + self._w
                self._y2 = self._y + self._h
            # elif self._format == BBFormat.XYX2Y2:  # <left> <top> <right> <bottom>.
            elif format == BBFormat.XYX2Y2:
                self._x = coordinates[0]
                self._y = coordinates[1]
                self._x2 = coordinates[2]
                self._y2 = coordinates[3]
                self._w = self._x2 - self._x
                self._h = self._y2 - self._y

        else:
            raise ValueError(
                "Invalid type_coordinates."
                "It must be CoordinatesType.ABSOLUTE or CoordinatesType.RELATIVE"
            )

        # Convert all values to float
        self._x = float(self._x)
        self._y = float(self._y)
        self._w = float(self._w)
        self._h = float(self._h)
        self._x2 = float(self._x2)
        self._y2 = float(self._y2)

    def get_absolute_bounding_box(self, format=BBFormat.XYWH):
        """Get bounding box in its absolute format.

        Parameters
        ----------
        format : Enum
            Format of the bounding box (BBFormat.XYWH or BBFormat.XYX2Y2) to be retreived.

        Returns
        -------
        tuple
            Four coordinates representing the absolute values of the bounding box.
            If specified format is BBFormat.XYWH, the coordinates are (upper-left-X, upper-left-Y,
            width, height).
            If format is BBFormat.XYX2Y2, the coordinates are (upper-left-X, upper-left-Y,
            bottom-right-X, bottom-right-Y).
        """
        if format == BBFormat.XYWH:
            return (self._x, self._y, self._w, self._h)
        elif format == BBFormat.XYX2Y2:
            return (self._x, self._y, self._x2, self._y2)

    def get_relative_bounding_box(self, img_size=None):
        """Get bounding box in its relative format.

        Parameters
        ----------
        img_size : tuple
            Image size in the format (width, height)=>(int, int)

        Returns
        -------
        tuple
            Four coordinates representing the relative values of the bounding box (x,y,w,h) where:
                x,y : bounding_box_center/width_of_the_image
                w   : bounding_box_width/width_of_the_image
                h   : bounding_box_height/height_of_the_image
        """
        if img_size is None and self._width_img is None and self._height_img is None:
            raise IOError(
                "Parameter 'img_size' is required. It is necessary to inform the image size."
            )
        if img_size is not None:
            return convert_to_relative_values(
                (img_size[0], img_size[1]), (self._x, self._x2, self._y, self._y2)
            )
        else:
            return convert_to_relative_values(
                (self._width_img, self._height_img),
                (self._x, self._x2, self._y, self._y2),
            )

    def set_image_name(self, name):
        self._image_name = name

    def get_image_name(self) -> str:
        """Get the string that represents the image.

        Returns
        -------
        string
            Name of the image.
        """
        return self._image_name

    def get_confidence(self):
        """Get the confidence level of the detection. If bounding box type is BBType.GROUND_TRUTH,
        the confidence is None.

        Returns
        -------
        float
            Value between 0 and 1 representing the confidence of the detection.
        """
        return self._confidence

    # def get_format(self):
    # """Get the format of the bounding box (BBFormat.XYWH or BBFormat.XYX2Y2).

    # Returns
    # -------
    # Enum
    #     Format of the bounding box. It can be either:
    #         BBFormat.XYWH: <left> <top> <width> <height>
    #         BBFomat.XYX2Y2: <left> <top> <right> <bottom>.
    # """
    # return self._format

    def set_class_id(self, class_id):
        self._class_id = class_id

    def set_bb_type(self, bb_type):
        self._bb_type = bb_type

    def set_image_size(self, size: tuple):
        """Set the size of the image where the bounding box is represented.

        Parameters
        ----------
        size : tuple
            Image size in pixels in the format (width, height)

        Added by Josh Bruegger 2023-07-30
        """
        self._width_img = size[0]
        self._height_img = size[1]

    def get_class_id(self):
        """Get the class of the object the bounding box represents.

        Returns
        -------
        string
            Class of the detected object (e.g. 'cat', 'dog', 'person', etc)
        """
        return self._class_id

    def get_image_size(self):
        """Get the size of the image where the bounding box is represented.

        Returns
        -------
        tupe
            Image size in pixels in the format (width, height)=>(int, int)
        """
        return (self._width_img, self._height_img)

    # Modified by Josh Bruegger 2023-07-30
    def get_area(self):
        # assert isclose(self._w * self._h, (self._x2 - self._x) * (self._y2 - self._y))
        assert self._x2 >= self._x
        assert self._y2 >= self._y
        return max(self._x2 - self._x, 0) * max(self._y2 - self._y, 0)
        # return (self._x2 - self._x + 1) * (self._y2 - self._y + 1)

    # def get_coordinates_type(self):
    #     """Get type of the coordinates (CoordinatesType.RELATIVE or CoordinatesType.ABSOLUTE).

    #     Returns
    #     -------
    #     Enum
    #         Enum representing if the bounding box coordinates (x,y,w,h) are absolute or relative
    #             to size of the image (CoordinatesType.RELATIVE or CoordinatesType.ABSOLUTE).
    #     """
    #     return self._type_coordinates

    def get_bb_type(self):
        """Get type of the bounding box that represents if it is a ground-truth or detected box.

        Returns
        -------
        Enum
            Enum representing the type of the bounding box (BBType.GROUND_TRUTH or BBType.DETECTED)
        """
        return self._bb_type

    def __str__(self):
        abs_bb_xywh = self.get_absolute_bounding_box(format=BBFormat.XYWH)
        abs_bb_xyx2y2 = self.get_absolute_bounding_box(format=BBFormat.XYX2Y2)
        area = self.get_area()
        return f"image name: {self._image_name}\nclass: {self._class_id}\nbb (XYWH): {abs_bb_xywh}\nbb (X1Y1X2Y2): {abs_bb_xyx2y2}\narea: {area}\nbb_type: {self._bb_type}"

    def __eq__(self, other):
        if not isinstance(other, BoundingBox):
            # unrelated types
            return False
        return str(self) == str(other)

    @staticmethod
    def compare(det1, det2):
        """Static function to compare if two bounding boxes represent the same area in the image,
            regardless the format of their boxes.

        Parameters
        ----------
        det1 : BoundingBox
            BoundingBox object representing one bounding box.
        dete2 : BoundingBox
            BoundingBox object representing another bounding box.

        Returns
        -------
        bool
            True if both bounding boxes have the same coordinates, otherwise False.
        """
        det1BB = det1.getAbsoluteBoundingBox()
        det1img_size = det1.get_image_size()
        det2BB = det2.getAbsoluteBoundingBox()
        det2img_size = det2.get_image_size()

        if (
            det1.get_class_id() == det2.get_class_id()
            and det1.get_confidence() == det2.get_confidence()
            and det1BB[0] == det2BB[0]
            and det1BB[1] == det2BB[1]
            and det1BB[2] == det2BB[2]
            and det1BB[3] == det2BB[3]
            and det1img_size[0] == det1img_size[0]
            and det2img_size[1] == det2img_size[1]
        ):
            return True
        return False

    @classmethod
    def clone(cls, bounding_box: "BoundingBox"):
        """Static function to clone a given bounding box.

        Parameters
        ----------
        bounding_box : BoundingBox
            Bounding box object to be cloned.

        Returns
        -------
        BoundingBox
            Cloned BoundingBox object.

        Fixed by Josh Bruegger 2023-07-30
        """
        absBB = bounding_box.get_absolute_bounding_box(format=BBFormat.XYWH)
        return cls(
            bounding_box.get_image_name(),
            class_id=bounding_box.get_class_id(),
            coordinates=absBB,
            type_coordinates=CoordinatesType.ABSOLUTE,
            img_size=bounding_box.get_image_size(),
            bb_type=bounding_box.get_bb_type(),
            confidence=bounding_box.get_confidence(),
            format=BBFormat.XYWH,
        )

    @staticmethod
    def iou(boxA, boxB):
        coords_A = boxA.get_absolute_bounding_box(format=BBFormat.XYX2Y2)
        coords_B = boxB.get_absolute_bounding_box(format=BBFormat.XYX2Y2)
        # if boxes do not intersect
        if BoundingBox.have_intersection(coords_A, coords_B) is False:
            return 0
        interArea = BoundingBox.get_intersection_area(coords_A, coords_B)
        union = BoundingBox.get_union_areas(boxA, boxB, interArea=interArea)
        # intersection over union
        iou = interArea / union
        assert iou >= 0
        return iou

    # boxA = (Ax1,Ay1,Ax2,Ay2)
    # boxB = (Bx1,By1,Bx2,By2)
    @staticmethod
    def have_intersection(boxA, boxB):
        if isinstance(boxA, BoundingBox):
            boxA = boxA.get_absolute_bounding_box(BBFormat.XYX2Y2)
        if isinstance(boxB, BoundingBox):
            boxB = boxB.get_absolute_bounding_box(BBFormat.XYX2Y2)
        if boxA[0] > boxB[2]:
            return False  # boxA is right of boxB
        if boxB[0] > boxA[2]:
            return False  # boxA is left of boxB
        if boxA[3] < boxB[1]:
            return False  # boxA is above boxB
        if boxA[1] > boxB[3]:
            return False  # boxA is below boxB
        return True

    @staticmethod
    def get_intersection_area(boxA, boxB):
        if isinstance(boxA, BoundingBox):
            boxA = boxA.get_absolute_bounding_box(BBFormat.XYX2Y2)
        if isinstance(boxB, BoundingBox):
            boxB = boxB.get_absolute_bounding_box(BBFormat.XYX2Y2)
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        # intersection area
        return (xB - xA + 1) * (yB - yA + 1)

    @staticmethod
    def get_union_areas(boxA, boxB, interArea=None):
        area_A = boxA.get_area()
        area_B = boxB.get_area()
        if interArea is None:
            interArea = BoundingBox.get_intersection_area(boxA, boxB)
        return float(area_A + area_B - interArea)

    @staticmethod
    def get_amount_bounding_box_all_classes(bounding_boxes, reverse=False):
        classes = list(set([bb._class_id for bb in bounding_boxes]))
        ret = {}
        for c in classes:
            ret[c] = len(BoundingBox.get_bounding_box_by_class(bounding_boxes, c))
        # Sort dictionary by the amount of bounding boxes
        ret = {
            k: v
            for k, v in sorted(ret.items(), key=lambda item: item[1], reverse=reverse)
        }
        return ret

    @staticmethod
    def get_bounding_box_by_class(bounding_boxes, class_id):
        # get only specified bounding box type
        return [bb for bb in bounding_boxes if bb.get_class_id() == class_id]

    @staticmethod
    def get_bounding_boxes_by_image_name(bounding_boxes, image_name):
        # get only specified bounding box type
        return [bb for bb in bounding_boxes if bb.get_image_name() == image_name]

    @staticmethod
    def get_total_images(bounding_boxes):
        return len(list(set([bb.get_image_name() for bb in bounding_boxes])))

    @staticmethod
    def get_average_area(bounding_boxes):
        areas = [bb.get_area() for bb in bounding_boxes]
        return sum(areas) / len(areas)

    @classmethod
    def from_list(
        cls,
        image_name: str,
        list_class_id: list[str],
        list_coordinates: list[tuple[float, float, float, float]],
        list_confidence: Optional[list[float]] = None,
        bb_type: BBType = BBType.GROUND_TRUTH,
        bb_format: BBFormat = BBFormat.XYWH,
        type_coordinates: CoordinatesType = CoordinatesType.ABSOLUTE,
        img_size: Optional[tuple[int, int]] = None,
    ) -> list["BoundingBox"]:
        """ Make a list of BoundingBox objects for the same image from listsof class_id, coordinates, confidence.
        All items in the list must have the same length. All bounding boxes are of the same type and format.

        Parameters
        ----------
            image_name : str
                list of Strings representing the name of the image.
            list_class_id : str
                list of Strings value representing class id.
            list_coordinates : tuple
                list of Tuples with 4 elements whose values (float) represent coordinates of the bounding \\
                    box.
                The coordinates can be (x, y, w, h)=>(float,float,float,float) or(x1, y1, x2, y2)\\
                    =>(float,float,float,float).
                See parameter `format`.
            list_confidence : float (optional)
                Value representing the confidence of the detected object. If detectionType is
                Detection, confidence needs to be informed.
            bb_type : Enum (optional)
                Enum identifying if the bounding box is a ground truth or a detection. If it is a
                detection, the confidence must be informed.
            bb_format : Enum
                Enum (BBFormat.XYWH or BBFormat.XYX2Y2) indicating the format of the coordinates of
                the bounding boxes.
                BBFormat.XYWH: <left> <top> <width> <height>
                BBFomat.XYX2Y2: <left> <top> <right> <bottom>.
                BBFomat.YOLO: <x_center> <y_center> <width> <height>. (relative)
            type_coordinates : Enum (optional)
                Enum representing if the bounding box coordinates (x,y,w,h) are absolute or relative
                to size of the image. Default:'Absolute'.
            img_size : tuple (optional)
                Image size in the format (width, height)=>(int, int) representinh the size of the
                image of the bounding box. If type_coordinates is 'Relative', img_size is required.

            By Josh Bruegger 2023-07-30
        """
        assert len(list_class_id) == len(
            list_coordinates
        ), "All lists must have the same length"

        has_conf = list_confidence is not None
        if has_conf:
            assert len(list_class_id) == len(
                list_confidence
            ), "All lists must have the same length"

        bounding_boxes = []
        for i, (id, coord) in enumerate(zip(list_class_id, list_coordinates)):
            bounding_boxes.append(
                cls(
                    image_name,
                    class_id=id,
                    coordinates=coord,
                    confidence=list_confidence[i] if has_conf else None,
                    bb_type=bb_type,
                    format=bb_format,
                    type_coordinates=type_coordinates,
                    img_size=img_size,
                )
            )
        return bounding_boxes

    @classmethod
    def from_image_detection_prediction(
        cls, image_name: str, image_detection_prediction
    ) -> list["BoundingBox"]:
        pred = image_detection_prediction.prediction
        shape = image_detection_prediction.image.shape[:2][::-1]

        return cls.from_list(
            image_name,
            pred.labels,
            pred.bboxes_xyxy,
            pred.confidence,
            type_coordinates=CoordinatesType.ABSOLUTE,
            img_size=shape,
            bb_type=BBType.DETECTED,
            bb_format=BBFormat.XYX2Y2,
        )
