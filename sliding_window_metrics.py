import numpy as np
from collections import defaultdict
from super_gradients.training.utils.predict import DetectionPrediction
import logging

MLOG = logging.getLogger("SlidingWindowMetrics")


class Metrics:
    IOU_05_95 = np.linspace(0.5, 0.95, 10)

    @staticmethod
    def intersection(box1, box2):
        x1, y1 = np.maximum(box1[:2], box2[:2])
        x2, y2 = np.minimum(box1[2:], box2[2:])
        return np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    @staticmethod
    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    @staticmethod
    def iou(box1, box2):
        intersection = Metrics.intersection(box1, box2)
        union = Metrics.box_area(box1) + Metrics.box_area(box2) - intersection
        return intersection / union if union > 0 else 0

    def precision(tp, fp):
        return tp / (tp + fp) if (tp + fp) > 0 else 0

    def recall(tp, fn):
        return tp / (tp + fn) if (tp + fn) > 0 else 0

    def accuracy(tp, fp, fn):
        return tp / (tp + fp + fn) if (tp + fp) + fn > 0 else 0

    def f1(precision, recall):
        return (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0
        )

    @staticmethod
    def get_metrics(
        preds: DetectionPrediction,
        target_bboxes: list[np.ndarray],
        target_labels: list[int],
        iou_threshold=0.5,
        use_05_95=False,
    ):
        thresholds = Metrics.IOU_05_95 if use_05_95 else [iou_threshold]
        if iou_threshold not in thresholds:
            thresholds.append(iou_threshold)

        MLOG.debug(f"Using thresholds: {thresholds}")

        tp, fp, fn = Metrics._get_matches(
            preds, target_bboxes, target_labels, thresholds
        )

        # Dictionary formatted as follows:
        #     threshold:
        #         class:
        #             metric: value
        # The "all" class is used to store metrics for all classes
        metrics = defaultdict(lambda: defaultdict(dict))

        for t in thresholds:
            # Get metrics for all classes
            metrics[t]["all"] = Metrics._compute_metrics(
                tp[t]["all"], fp[t]["all"], fn[t]["all"], preds
            )
            # Get metrics for each class
            for c in set(target_labels):
                metrics[t][c] = Metrics._compute_metrics(
                    tp[t][c], fp[t][c], fn[t][c], preds
                )

        return metrics, (tp, fp, fn)

    @staticmethod
    def _compute_metrics(tp_set, fp_set, fn_set, preds):
        tp = len(tp_set)
        fp = len(fp_set)
        fn = len(fn_set)

        metrics = {}
        metrics["precision"] = Metrics.precision(tp, fp)
        metrics["accuracy"] = Metrics.accuracy(tp, fp, fn)
        metrics["recall"] = Metrics.recall(tp, fn)
        metrics["f1"] = Metrics.f1(metrics["precision"], metrics["recall"])
        metrics["ap"] = Metrics._compute_ap(tp_set, fp_set, fn_set, preds)
        return metrics

    def _ap_get_prs(boxes, tp_set, fp_set, fn_set):
        num = len(boxes)
        precisions = np.zeros(num)
        recalls = np.zeros(num)
        tp, fp, fn = 0, 0, 0
        for i in range(num):
            itp, ifp, ifn = i in tp_set, i in fp_set, i in fn_set
            if not any([itp, ifp, ifn]):
                continue
            tp, fp, fn = tp + itp, fp + ifp, fn + ifn
            precisions[i] = Metrics.precision(tp, fp)
            recalls[i] = Metrics.recall(tp, fn)
            MLOG.debug(
                f"{i}: tp={tp}, fp={fp}, fn={fn}, Precision: {precisions[i]}, Recall: {recalls[i]}"
            )
        return precisions, recalls

    @staticmethod
    def _compute_ap(tp_set, fp_set, fn_set, preds: DetectionPrediction):
        boxes = preds.bboxes_xyxy
        labels = preds.labels
        confs = preds.confidence

        # Sort predictions by confidence
        sorted_idxs = np.argsort(confs)[::-1]
        boxes = boxes[sorted_idxs]
        labels = labels[sorted_idxs]
        confs = confs[sorted_idxs]

        precisions, recalls = Metrics._ap_get_prs(boxes, tp_set, fp_set, fn_set)

        # Compute the area under the precision-recall curve
        indices = np.argsort(recalls)
        sorted_recalls = recalls[indices]
        sorted_precisions = precisions[indices]
        return np.trapz(sorted_precisions, sorted_recalls)

    @staticmethod
    def _get_matches(
        preds: DetectionPrediction, target_bboxes, target_labels, thresholds
    ):
        MLOG.debug("Getting matches...")
        tp = defaultdict(lambda: defaultdict(set[int]))
        fp = defaultdict(lambda: defaultdict(set[int]))
        fn = defaultdict(lambda: defaultdict(set[int]))
        matched_targets = defaultdict(set[int])

        for i, (p_box, p_label) in enumerate(zip(preds.bboxes_xyxy, preds.labels)):
            ious = defaultdict(list[tuple])
            for j, (t_box, t_label) in enumerate(zip(target_bboxes, target_labels)):
                if p_label != t_label or j in matched_targets:
                    continue
                iou = Metrics.iou(p_box, t_box)
                for t in thresholds:
                    if iou > t:
                        ious[t].append((iou, j))

            for t in thresholds:
                if len(ious[t]) == 0:
                    fp[t]["all"].add(i)
                    fp[t][p_label].add(i)
                else:
                    iou, j = max(ious[t], key=lambda x: x[0])
                    tp[t]["all"].add(i)
                    tp[t][p_label].add(i)
                    matched_targets[t].add(j)
        for t in thresholds:
            fn[t]["all"] = set(range(len(target_bboxes))) - matched_targets[t]
            for i in fn[t]["all"]:
                fn[t][target_labels[i]].add(i)
        return tp, fp, fn

    @staticmethod
    def print_metrics(metrics):
        for threshold, metrics_dict in metrics.items():
            MLOG.debug(f"Metrics for IOU threshold {threshold}:")
            for class_id, class_metrics in metrics_dict.items():
                MLOG.debug(f"  Class {class_id}:")
                for metric, value in class_metrics.items():
                    MLOG.debug(f"    {metric}: {value}")
