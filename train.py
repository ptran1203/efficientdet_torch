from models.model import get_model, make_predictions, run_training
from dataloader import DatasetRetriever, get_img_list_from_df
from albumentations.pytorch.transforms import ToTensorV2
from tqdm import tqdm
import albumentations as A
import pandas as pd
import argparse
import os
import yaml
from utils import ConfigFromDict
from utils.scheduler import WarmupCosineSchedule

def get_config(cfg_file, args):
    with open(cfg_file) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
        cfg['epochs'] = args.epochs
        cfg['fold'] = args.fold
        print(yaml.dump(cfg))
        return ConfigFromDict(cfg)

def get_train_transforms(config):
    augments = []

    if config.fliplr:
        augments.append(A.HorizontalFlip(p=config.fliplr))

    if config.flipud:
        augments.append(A.VerticalFlip(p=config.flipud))

    if config.shift_scale_rot:
        augments.append(A.ShiftScaleRotate(
            p=config.shift_scale_rot,
            rotate_limit=config.rot_limit,
            shift_limit=config.shift_limit,
            scale_limit=config.scale_limit,
        ))

    if config.gaussian_blur:
        augments.append(A.GaussianBlur(p=config.gaussian_blur))

    if config.hue_sat:
        augments.append(A.HueSaturationValue(
            hue_shift_limit=config.hue_limit,
            sat_shift_limit=config.sat_limit, 
            val_shift_limit=config.val_huesat_limit,
            p=config.hue_sat
        ))

    if config.brightness_contrast:
        augments.append(A.RandomBrightnessContrast(
            brightness_limit=config.brightness_limit, 
            contrast_limit=config.contrast_limit,
            p=config.brightness_contrast
        ))
    
    augments.append(A.Resize(height=config.image_size, width=config.image_size, p=1.0))
    augments.append(ToTensorV2(p=1.0))

    return A.Compose(
        augments, 
        p=1.0,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0, 
            min_visibility=0,
            label_fields=['labels']
        )
    )

def get_valid_transforms(config):
    return A.Compose(
        [
            A.Resize(height=config.image_size, width=config.image_size, p=1.0),
            ToTensorV2(p=1.0),
        ], 
        p=1.0,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0, 
            min_visibility=0,
            label_fields=['labels']
        )
    )

class TrainGlobalConfig:
    num_workers = 2
    verbose = True
    verbose_step = 1
    step_scheduler = True
    validation_scheduler = False
    SchedulerClass = WarmupCosineSchedule

    def __init__(self, config):
        self.model_name = f'efficientdet_d{config.phi}_fold{config.fold}'
        self.warmup_epochs = config.warmup_epochs
        self.lr = config.lr
        self.n_epochs = config.epochs
        self.scheduler_params = dict(
            warmup_steps=self.warmup_epochs,
            t_total=self.n_epochs
        )
        self.folder = config.weight_dir
        self.batch_size = config.batch_size


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--cfg', type=str, default='config.yaml')
    parser.add_argument('--epochs', type=int, default=30)

    args = parser.parse_args()
    checkpoint = args.checkpoint
    fold = args.fold

    config = get_config(args.cfg, args)

    fold_csv = pd.read_csv(config.fold_csv)
    dataframe = pd.read_csv(config.data_csv)
    dataframe = dataframe[dataframe['class_id'] != 14].reset_index(drop= True)

    valid_imgs = get_img_list_from_df(fold_csv, [fold])
    # 5 folds
    train_imgs = get_img_list_from_df(fold_csv, [i for i in range(5) if i != fold])

    val_dataset = DatasetRetriever(
        image_ids=valid_imgs,
        marking=dataframe,
        transforms=get_valid_transforms(config),
        test=True,
        image_size=config.image_size,
        image_dir=config.image_dir,
    )

    train_dataset = DatasetRetriever(
        image_ids=train_imgs,
        marking=dataframe,
        transforms=get_train_transforms(config),
        test=False,
        image_size=config.image_size,
        image_dir=config.image_dir,
        mosaic=config.mosaic,
        random_intensity=config.random_intensity,
    )

    model = get_model(phi=config.phi,
                    num_classes=config.num_classes,
                    image_size=config.image_size,
                    checkpoint_path=checkpoint,
                    is_inference=False)

    train_config = TrainGlobalConfig(config)
    run_training(model, train_config, train_dataset, val_dataset, env='script')
