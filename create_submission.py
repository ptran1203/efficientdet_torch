from models.model import get_model, make_predictions
from tqdm import tqdm
import torch
import pandas as pd
import argparse
import os
import cv2
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--phi', type=int, default=0)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--data-csv', type=str, default='train_df.csv')
    parser.add_argument('--score-threshold', type=float, default=0.01)
    parser.add_argument('--iou-threshold', type=float, default=0.5)
    parser.add_argument('--image-size', type=int, default=640)
    parser.add_argument('--image-dir', type=str, default='/content')
    parser.add_argument('--box-scale', type=int, default=2)
    parser.add_argument('--output-dir', type=str, default='/content')

    args = parser.parse_args()
    checkpoint = args.checkpoint
    score_thr = args.score_threshold
    iou_thr = args.iou_threshold
    gimage_size = args.image_size
    image_dir = args.image_dir
    data_csv = args.data_csv
    phi = args.phi
    box_scale = args.box_scale
    output_dir = args.output_dir

    if not os.path.exists(checkpoint):
        raise ValueError(f'{checkpoint} does not exist')


    model = get_model(phi=phi, num_classes=14,
                  image_size=gimage_size,
                  checkpoint_path=checkpoint,
                  is_inference=True)

    submission = {
        "image_id": [],
        "PredictionString": [],
    }

    for fname in tqdm(os.listdir(image_dir)):
        path = os.path.join(image_dir, fname)
        image = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32)
        image = cv2.resize(image, (gimage_size, gimage_size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        image = torch.from_numpy(image.transpose(2, 0, 1))
        image_id = fname.split(".")[0]

        boxes, scores, labels = make_predictions(
            model, image, score_thr=score_thr,
            iou_thr=iou_thr,
        )

        submission["image_id"].append(image_id)

        if len(boxes):
            pred = []
            for box, cls, score in zip(boxes, labels, scores):
                x1, y1, x2, y2 = [int(v) for v in box * box_scale]
                prediction_text = f"{int(cls) - 1} {score} {round(x1)} {round(y1)} {round(x2)} {round(y2)}"
                pred.append(prediction_text)
            submission["PredictionString"].append(" ".join(pred))
        else:
            submission["PredictionString"].append('14 1.0 0 0 1 1')

    filename = f'submission_effdetD{phi}_fold{args.fold}.csv'
    submission = pd.DataFrame(submission)
    submission.to_csv(os.path.join(output_dir, filename), index=False)
