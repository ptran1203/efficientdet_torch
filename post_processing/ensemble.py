import os
import pandas as pd
import json
import argparse
import numpy as np
from tqdm import tqdm
from ensemble_boxes import weighted_boxes_fusion, nms

submission_file = "submission_ensemble.csv"

working_dir = os.getcwd()
OUTPUT_DIR = os.path.join(working_dir, "output")

not os.path.exists(OUTPUT_DIR) and os.mkdir(OUTPUT_DIR)

if "post_processing" not in working_dir:
    working_dir = os.path.join(working_dir, "post_processing")

def read_csv(fname):
    return pd.read_csv(
        os.path.join(working_dir, "input", fname)
    )

def write_csv(df, fname):
    return df.to_csv(
        os.path.join(working_dir, "output", fname),
        index=False
    )

def get_weight(weights, idx, cls):
    return weights[idx][cls] if isinstance(weights[idx], dict) else weights[idx]

def parse_pred(pred, mode='xyxy'):
    '''
    mode: center|xyxy|xywh
    '''
    span = 6
    words = pred.split(" ")
    preds = [(words[i: i + span]) for i in range(0, len(words), span)]
    res = []
    for pred in preds:
        cls, conf, x1, y1, x2, y2 = pred
        cls = int(cls)
        conf = float(conf)
        x1 = float(x1)
        y1 = float(y1)
        x2 = float(x2)
        y2 = float(y2)
        if mode == 'center':
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            res.append([center_x, center_y, (x2 - x1), (y2 - y1), cls, conf])
        elif mode == 'xyxy':
            res.append([x1, y1, x2, y2, cls, conf])
        elif mode =='xywh':
            res.append([x1, y1, x2 - x1, y2 - y1, cls, conf])
        else:
            raise ValueError('mode')

    return res

def convert(pred, mode):
    boxes, labels, scores = [], [], []
    for p in pred:
        # center_x, center_y, (x2 - x1) / 2, (y2 - y1) / 2, cls, conf
        x1, y1, x2, y2, cls, score = p
        boxes.append([x1, y1, x2, y2])
        labels.append(cls)
        scores.append(score)

    return boxes, labels, scores


def get_wh(df, image_id):
    im_w, im_h = df.loc[df.image_id == image_id, ['width', 'height']].values[0]
    return im_w, im_h

def normbox(boxes, w, h):
    '''
    boxes: (x1, y1, x2, y2)
    '''
    if isinstance(boxes, list):
        boxes = np.array(boxes)
    
    boxes = np.stack([
        boxes[:, 0] / w,
        boxes[:, 1] / h,
        boxes[:, 2] / w,
        boxes[:, 3] / h,
    ], axis=1)
    return boxes.tolist()

def re_format_input(filename, mode, test_df):
    print(f"Prepare {filename}")
    df = read_csv(filename)
    boxes = []
    labels = []
    scores = []
    img_ids = []
    for i, row in tqdm(df.iterrows()):
        w, h = get_wh(test_df, row["image_id"])
        pred = row["PredictionString"]
        p = parse_pred(pred, mode)
        b, l, s = convert(p, mode)
        b = normbox(b, w, h)
        img_ids.append(row["image_id"])
        boxes.append(b)
        scores.append(s)
        labels.append(l)

    return boxes, labels, scores, img_ids

def re_format_ensemble(pred, w, h):
    res = []
    boxes, scores, labels = pred
    boxes = normbox(boxes, 1 / w, 1 / h)
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        x1 = int(round(x1))
        x2 = int(round(x2))
        y1 = int(round(y1))
        y2 = int(round(y2))
        res.append(f"{label} {score} {x1} {y1} {x2} {y2}")

    return " ".join(res)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--iou', type=int, default=0.5)
    parser.add_argument('--method', type=str, default='nms')
    args = parser.parse_args()
    merge_func = nms if args.method == 'nms' else weighted_boxes_fusion

    submission_files = os.listdir(os.path.join(working_dir, "input/submission"))
    submission_files = [x for x in submission_files if x.endswith(".csv")]
    all_preds = []
    img_ids = None
    mode = 'xyxy'
    test_df = pd.read_csv('../test.csv')

    submission_info = []
    weights = [1] * len(submission_files)

    img_ids_list = []
    boxes,labels,scores = [],[],[]
    for sub_file, weight in zip(submission_files, weights):
        print(sub_file)
        splited = sub_file.split("_")
        fold, model_name = splited[1:3]
        model_name = model_name.split(".")[0]
        if isinstance(weight, dict):
            weight = round(sum(weight.values()) / 14, 2)
        submission_info.append(f"{model_name}_{fold}_w{weight}")

        print(f"{model_name} {fold}: {weight}")
        box, label, score, img_ids = re_format_input("submission/" + sub_file, mode=mode, test_df=test_df)
        boxes.append(box)
        labels.append(label)
        scores.append(score)

    ens_pred = []
    iou_thresh = args.iou

    boxes = np.array(boxes)
    labels = np.array(labels)
    scores = np.array(scores)

    print(boxes.shape)

    for i in tqdm(range(len(img_ids))):
        image_id = img_ids[i]
        w, h = get_wh(test_df, image_id)
        ens = merge_func(boxes[:, i].tolist(), scores[:, i].tolist(), labels[:, i].tolist(), iou_thr=iou_thresh)
        ens_pred.append(re_format_ensemble(ens, w, h))

    ens_df = pd.DataFrame({
        "image_id": img_ids,
        "PredictionString": ens_pred,
    })

    out_file = f"ensemble_iou{iou_thresh}_{args.method}.csv"
    write_csv(ens_df, out_file)

    submission_script = "kaggle competitions submit -c vinbigdata-chest-xray-abnormalities-detection -f"
    msg = f"ensemble iou {iou_thresh} {' '.join(submission_info)}"
    print(f"{submission_script} {os.path.join(OUTPUT_DIR, out_file)} -m '{msg}'")
