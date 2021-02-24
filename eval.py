from models.model import get_model, make_predictions
from dataloader import DatasetRetriever, get_img_list_from_df
from albumentations.pytorch.transforms import ToTensorV2
from tqdm import tqdm
import albumentations as A
import pandas as pd
import argparse
import os

def get_valid_transforms(gimage_size):
    return A.Compose(
        [
            A.Resize(height=gimage_size, width=gimage_size, p=1.0),
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--valid-csv', type=str) # validation csv file
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--phi', type=int, default=0)
    parser.add_argument('--data-csv', type=str, default='train_df.csv')
    parser.add_argument('--score-threshold', type=float, default=0.01)
    parser.add_argument('--iou-threshold', type=float, default=0.5)
    parser.add_argument('--image-size', type=int, default=640)
    parser.add_argument('--image-dir', type=str, default='/content')

    args = parser.parse_args()
    checkpoint = args.checkpoint
    valid_csv = args.valid_csv
    score_thr = args.score_threshold
    iou_thr = args.iou_threshold
    gimage_size = args.image_size
    image_dir = args.image_dir
    data_csv = args.data_csv
    fold = args.fold
    phi = args.phi

    if not os.path.exists(checkpoint):
        raise ValueError(f'{checkpoint} does not exist')

    if not os.path.exists(valid_csv):
        raise ValueError(f'{valid_csv} does not exist')

    val_df = pd.read_csv(valid_csv)
    dataframe = pd.read_csv(data_csv)
    dataframe = dataframe[dataframe['class_id'] != 14].reset_index(drop= True)

    valid_imgs = get_img_list_from_df(val_df, [fold])

    val_dataset = DatasetRetriever(
        image_ids=valid_imgs,
        marking=dataframe,
        transforms=get_valid_transforms(gimage_size),
        test=True,
        image_size=gimage_size,
        image_dir=image_dir,
    )

    model = get_model(phi=phi, num_classes=14,
                  image_size=gimage_size,
                  checkpoint_path=checkpoint,
                  is_inference=True)

    for image, target, _ in tqdm(val_dataset):
        boxes, scores, labels = make_predictions(
            model, image, score_thr=score_thr,
            iou_thr=iou_thr,
        )

        gt_boxes = target['boxes'].numpy()
        gt_boxes = gt_boxes[:, [1, 0, 3, 2]]
        gt_labels = target['labels'].numpy()

        with open(
            f"./evaluation/input/detection-results/val_{image_id}.txt", "w"
        ) as f:
            for box, cls, score in zip(boxes, classes, scores):
                x1, y1, x2, y2 = [int(v) for v in box]
                pred_text = f"{int(cls)} {round(score, 2)} {round(x1)} {round(y1)} {round(x2)} {round(y2)}\n"
                f.write(pred_text)

        with open(
            f"./evaluation/input/ground-truth/val_{image_id}.txt", "w"
        ) as f:
            for box, cls in zip(gt_boxes, gt_classes):
                x1, y1, x2, y2 = box
                pred_text = f"{int(cls)} {round(x1)} {round(y1)} {round(x2)} {round(y2)}\n"
                f.write(pred_text)
