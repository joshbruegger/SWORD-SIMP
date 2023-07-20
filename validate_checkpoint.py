import argparse
import os
from super_gradients.training import Trainer, models
from super_gradients.training.dataloaders.dataloaders import (
    coco_detection_yolo_format_train,
)
from super_gradients.training.metrics import DetectionMetrics_050
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback
from super_gradients.common.object_names import Models


def main():
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "checkpoint", type=str, help="path to the checkpoint to validate"
    )
    parser.add_argument(
        "validation_dir", type=str, help="path to the validation directory"
    )

    args = parser.parse_args()

    trainer = Trainer(experiment_name="test", ckpt_root_dir="testcheckpoints")
    best_model = models.get(
        Models.YOLO_NAS_L,
        num_classes=28,
        checkpoint_path=os.path.abspath(args.checkpoint),
    )

    test_data = coco_detection_yolo_format_train(
        dataset_params={
            "data_dir": args.validation_dir,
            "images_dir": os.path.join(args.validation_dir, "images"),
            "labels_dir": os.path.join(args.validation_dir, "labels"),
            "classes": [
                2,
                47,
                69,
                138,
                183,
                196,
                233,
                333,
                360,
                376,
                377,
                380,
                388,
                392,
                403,
                405,
                418,
                537,
                558,
                599,
                606,
                631,
                644,
                649,
                652,
                680,
                690,
                693,
            ],
        },
        dataloader_params={
            "batch_size": 32,
            "num_workers": 2,
        },
    )

    r = trainer.test(
        model=best_model,
        test_loader=test_data,
        test_metrics_list=[
            DetectionMetrics_050(
                score_thres=0.1,
                top_k_predictions=300,
                num_cls=28,
                normalize_targets=True,
                post_prediction_callback=PPYoloEPostPredictionCallback(
                    score_threshold=0.01,
                    nms_top_k=1000,
                    max_predictions=300,
                    nms_threshold=0.7,
                ),
            )
        ],
    )

    print(r)


if __name__ == "__main__":
    main()
