# Adapted from YOLO-NAS tutorial by Harpreet:
# https://colab.research.google.com/drive/10N6NmSMCiRnFlKV9kaIS_z3pk0OI1xKC?usp=sharing

import os
import argparse

from super_gradients.training import Trainer, models
from super_gradients.training.dataloaders.dataloaders import (
    coco_detection_yolo_format_train, coco_detection_yolo_format_val
)
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import DetectionMetrics_050
from super_gradients.training.models.detection_models.pp_yolo_e import (
    PPYoloEPostPredictionCallback
)

class config:
    args = None

    # constructor
    def __init__(self, args):
        self.args = args

    #trainer params
    CHECKPOINT_DIR = 'checkpoints'
    EXPERIMENT_NAME = 'SWORD_YOLO_NAS'
    NUM_EPOCHS = args.num_epochs

    #dataset params
    DATA_DIR = args.dir

    TRAIN_IMAGES_DIR = 'train/images' #child dir of DATA_DIR where train images are
    TRAIN_LABELS_DIR = 'train/labels' #child dir of DATA_DIR where train labels are

    VAL_IMAGES_DIR = 'val/images' #child dir of DATA_DIR where validation images are
    VAL_LABELS_DIR = 'val/labels' #child dir of DATA_DIR where validation labels are

    # if you have a test set
    TEST_IMAGES_DIR = 'test/images' #child dir of DATA_DIR where test images are
    TEST_LABELS_DIR = 'test/labels' #child dir of DATA_DIR where test labels are

    # model params
    MODEL_NAME = 'yolo_nas_l' # choose from yolo_nas_s, yolo_nas_m, yolo_nas_l
    PRETRAINED_WEIGHTS = 'coco' #only one option here: coco


    CLASSES = []
    with open(args.classes, 'r') as f:
        for line in f:
            CLASSES.append(line.strip())

    NUM_CLASSES = len(CLASSES)

    #dataloader params - you can add whatever PyTorch dataloader params you have
    #could be different across train, val, and test
    DATALOADER_PARAMS={
    'batch_size':args.batch_size,
    'num_workers':args.num_workers,
    }



def train(config):
    trainer = Trainer(experiment_name=config.EXPERIMENT_NAME, ckpt_root_dir=config.CHECKPOINT_DIR)

    train_data = coco_detection_yolo_format_train(
        dataset_params={
            'data_dir': config.DATA_DIR,
            'images_dir': config.TRAIN_IMAGES_DIR,
            'labels_dir': config.TRAIN_LABELS_DIR,
            'classes': config.CLASSES
        },
        dataloader_params=config.DATALOADER_PARAMS
    )

    val_data = coco_detection_yolo_format_val(
        dataset_params={
            'data_dir': config.DATA_DIR,
            'images_dir': config.VAL_IMAGES_DIR,
            'labels_dir': config.VAL_LABELS_DIR,
            'classes': config.CLASSES
        },
        dataloader_params=config.DATALOADER_PARAMS
    )

    test_data = coco_detection_yolo_format_val(
        dataset_params={
            'data_dir': config.DATA_DIR,
            'images_dir': config.TEST_IMAGES_DIR,
            'labels_dir': config.TEST_LABELS_DIR,
            'classes': config.CLASSES
        },
        dataloader_params=config.DATALOADER_PARAMS
    )


    model = models.get(config.MODEL_NAME, 
                    num_classes=config.NUM_CLASSES, 
                    pretrained_weights=config.PRETRAINED_WEIGHTS
                    )

    train_params = {
        "average_best_models":True,
        "warmup_mode": "linear_epoch_step",
        "warmup_initial_lr": 1e-6,
        "lr_warmup_epochs": 3,
        "initial_lr": 5e-4,
        "lr_mode": "cosine",
        "cosine_final_lr_ratio": 0.1,
        "optimizer": "Adam",
        "optimizer_params": {"weight_decay": 0.0001},
        "zero_weight_decay_on_bias_and_bn": True,
        "ema": True,
        "ema_params": {"decay": 0.9, "decay_type": "threshold"},
        "max_epochs": config.NUM_EPOCHS,
        "mixed_precision": True,
        "loss": PPYoloELoss(
            use_static_assigner=False,
            num_classes=config.NUM_CLASSES,
            reg_max=16
        ),
        "valid_metrics_list": [
            DetectionMetrics_050(
                score_thres=0.1,
                top_k_predictions=300,
                num_cls=config.NUM_CLASSES,
                normalize_targets=True,
                post_prediction_callback=PPYoloEPostPredictionCallback(
                    score_threshold=0.01,
                    nms_top_k=1000,
                    max_predictions=300,
                    nms_threshold=0.7
                )
            )
        ],
        "metric_to_watch": 'mAP@0.50'
    }

    trainer.train(model=model, 
                training_params=train_params, 
                train_loader=train_data, 
                valid_loader=val_data)

    best_model = models.get(config.MODEL_NAME,
                            num_classes=config.NUM_CLASSES,
                            checkpoint_path=os.path.join(config.CHECKPOINT_DIR, config.EXPERIMENT_NAME, 'average_model.pth'))

    trainer.test(model=best_model,
                test_loader=test_data,
                test_metrics_list=DetectionMetrics_050(score_thres=0.1, 
                                                    top_k_predictions=300, 
                                                    num_cls=config.NUM_CLASSES, 
                                                    normalize_targets=True, 
                                                    post_prediction_callback=PPYoloEPostPredictionCallback(score_threshold=0.01, 
                                                                                                            nms_top_k=1000, 
                                                                                                            max_predictions=300,                                                                              nms_threshold=0.7)
                                                    ))
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', '-d', type=str, default='dataset')
    parser.add_argument('--classes', '-c', type=str, default='classes.txt')
    parser.add_argument('--num_epochs', '-e', type=int, default=10)
    parser.add_argument('--batch_size', '-b', type=int, default=16)
    parser.add_argument('--num_workers', '-w',type=int, default=4)

    config = config(parser.parse_args())
    train(config)