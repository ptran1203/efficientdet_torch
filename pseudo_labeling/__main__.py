import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import json
import platform
from PIL import Image
from tqdm import tqdm
from helper import (
    visualize_detections,
    parse_boxes,
    to_array,
    computeIOU,
    label_map,
    parse_prediction_string,
)

# MACbook
if platform.system() == "Darwin":
    # 4x downscaled
    IMAGE_DIR = "/Users/macbook/Downloads/train_vinbigdata/train/"
    aratio = 2 / 4
else:
    IMAGE_DIR = "/home/ubuntu/Documents/archive/vinbigdata-coco-dataset-with-wbf-3x-downscaled/train_images"
    aratio = 1 / 3

def filter_prediction(
    pred_boxes,
    pred_scores,
    pred_labels,
    truth_boxes,
    truth_labels,
    iou_thr=0.5,
    score_thr=0.5,
    confuse_thr=0.7,
):
    correct_labels = []
    per_label_correct = {
        k: {
            "correct": 0,
            "total": 0
        }
        for k in range(14)
    }

    confuse_predictions = []

    for pbox, pscore, plabel in zip(pred_boxes, pred_scores, pred_labels):
        # Change data type
        for tbox, tlabel in zip(truth_boxes, truth_labels):
            int_label = int(tlabel)
            iou = computeIOU(pbox, tbox)

            if plabel == tlabel and pscore >= score_thr:
                if iou >= iou_thr:
                    per_label_correct[int_label]["correct"] += 1
                    # correct_labels.append(plabel)
                elif iou > 0:
                    correct_labels.append(plabel)

            if plabel != tlabel and iou >= iou_thr and pscore >= confuse_thr:
                confuse_predictions.append({
                    "truth_label": int(tlabel),
                    "predict": int(plabel),
                    "confidence": float(pscore),
                    "iou": float(iou),
                    "pred_boxes": pbox.tolist(),
                    "truth_boxes": tbox.tolist(),
                })

            elif plabel == tlabel and plabel == 2 and 0.5 > iou >= 0.1 and pscore >= 0.3:
                confuse_predictions.append({
                    "truth_label": int(tlabel),
                    "predict": int(plabel),
                    "confidence": float(pscore),
                    "iou": float(iou),
                    "pred_boxes": pbox.tolist(),
                    "truth_boxes": tbox.tolist(),
                })

    for tbox, tlabel in zip(truth_boxes, truth_labels):
        per_label_correct[int(tlabel)]["total"] += 1

    return correct_labels, per_label_correct, confuse_predictions

def scale_boxes(boxes, w, h):
    is_list = False
    if isinstance(boxes, list):
        boxes = np.array(boxes)
        is_list = True

    sw = w / 640
    sh = h / 640

    _boxes = np.stack([
        boxes[:, 0] * sw,
        boxes[:, 1] * sh,
        boxes[:, 2] * sw,
        boxes[:, 3] * sh,
    ], axis=1)

    return _boxes

def run_single_file(filename, confuse_thr, train_df):
    def update_count(g_per_class_correct, per_class_correct):
        # 14 classes
        for k, v in per_class_correct.items():
            g_per_label_correct[k]["total"] += per_class_correct[k]["total"]
            g_per_label_correct[k]["correct"] += per_class_correct[k]["correct"]

        return g_per_label_correct

    df = pd.read_csv(f"pseudo_labeling/input/{filename}")
    config = filename.replace("train_prediction_", "").split(".")[0]

    correct_preds = []
    g_per_label_correct = {
        k: {
            "correct": 0,
            "total": 0
        }
        for k in range(14)
    }

    confuse_predictions = []

    for i, row in tqdm(df.iterrows()):
        image_id = row['image_id']
        pred_boxes, pred_scores, pred_labels = parse_prediction_string(row['PredictionString'])

        gt_boxes = train_df.loc[train_df.image_id==image_id, ['x_min', 'y_min', 'x_max', 'y_max']].values
        gt_labels = train_df.loc[train_df.image_id==image_id, ['class_id']].values[0]
        im_w, im_h = train_df.loc[train_df.image_id == image_id, ['width', 'height']].values[0]

        # Scale prediction boxes
        pred_boxes = scale_boxes(pred_boxes, w=im_w, h=im_h)

        # produced by 640
        # gt_boxes[:, [0, 2]] *= im_w / 640
        # gt_boxes[:, [1, 3]] *= im_h / 640

        correct_pred, per_label_correct, confuse_prediction = filter_prediction(
            pred_boxes,
            pred_scores,
            pred_labels,
            gt_boxes,
            gt_labels,
            confuse_thr=confuse_thr,
        )

        for p in correct_pred:
            correct_preds.append(p)

        for p in confuse_prediction:
            confuse_predictions.append({
                "image_id": row["image_id"],
                **p
            })

        update_count(g_per_label_correct, per_label_correct)

    # sns.countplot(correct_preds)
    # plt.show()

    # print information
    ratios = []
    for k, v in g_per_label_correct.items():
        ratio = v['correct'] / v['total']
        ratios.append(ratio)
        # print(f"{k}: correct {v['correct']} total {v['total']}, ratio {ratio:2f}")

    # print(f"Mean {sum(ratios) / 14}") 
    with open(f"pseudo_labeling/output/confuse_prediction_{config}.json", "w") as f:
        json.dump(confuse_predictions, f, indent=4)

def samebox(b1, b2):
    return set(b1) == set(b2)

def merge_pred(preds):
    pred_labels = [pred["predict"] for pred in preds]
    if len(set(pred_labels)) != 1:
        print(f"Confuse predict {pred_labels}")
        return False

    pred_labels = [pred["truth_label"] for pred in preds]
    if len(set(pred_labels)) != 1:
        print(f"Confuse ground truth {pred_labels}")
        return False

    confs = []
    boxes = []
    image_id = ""
    truth_label = -1
    predict = -1
    pred_label = -1
    truth_boxes = []

    for pred in preds:
        confs.append(pred["confidence"])
        boxes.append(pred["pred_boxes"])
        image_id = pred["image_id"]
        truth_label = pred["truth_label"]
        pred_label = pred["predict"]
        truth_boxes = pred["truth_boxes"]

    merge_box = np.mean(np.array(boxes), axis=0).tolist()

    # scale
    merge_box = [x * aratio for x in merge_box]
    truth_boxes = [x * aratio for x in truth_boxes]

    return {
        "image_id": image_id,
        "truth_label": truth_label,
        "predict": pred_label,
        "truth_boxes": truth_boxes,
        "pred_boxes": merge_box,
        "iou": computeIOU(truth_boxes, merge_box),
        "confidence": sum(confs) / len(preds),
        "consensus": len(preds),
    }

def remove_files_in_dir(dir_):
    if not os.path.exists(dir_):
        os.mkdir(dir_)
    for f in os.listdir(dir_):
        path = os.path.join(dir_, f)
        if os.path.isfile(path):
            os.unlink(path)

if __name__ == "__main__":
    confuse_thr = 0.5
    input_dir = "pseudo_labeling/input"
    output_dir = "pseudo_labeling/output"
    not os.path.exists(output_dir) and os.mkdir(output_dir)
    filenames = os.listdir(input_dir)
    num_ensemble = len(filenames)
    train_df = pd.read_csv("train_df.csv")

    for fname in filenames:
        run_single_file(fname, confuse_thr, train_df)

    # Merge prediction

    preds = {}
    output_files = os.listdir(output_dir)
    for ofile in output_files:
        pred = json.load(open(os.path.join(output_dir, ofile), "r"))
        for p in pred:
            img_id = p["image_id"]
            box = p["truth_boxes"]
            if img_id in preds:
                # Get head
                _pred = preds[img_id][0]
                if samebox(_pred["truth_boxes"], box):
                    preds[img_id].append(p)
            else:
                preds[img_id] = [p]

    preds = list(preds.values())
    for i in range(len(preds)):
        # combine
        merged = merge_pred(preds[i])
        if merged:
            preds[i] = merged
        else:
            preds[i] = False

    # filter
    # preds = [p for p in preds if p and p["consensus"] / num_ensemble >= 0.5]

    save_image = True
    if save_image:
        # Remove old images
        remove_files_in_dir("pseudo_labeling/images")
        for conf in preds:
            image_path = os.path.join(IMAGE_DIR, conf['image_id'] + ".jpg")
            if not os.path.exists(image_path):
                print(f"File not exist, skip {conf['image_id']}")
                continue

            visualize_detections(
                Image.open(image_path),
                boxes=[conf["pred_boxes"]],
                classes=[conf["predict"]],
                scores=[conf["confidence"]],
                box_true=[conf["truth_boxes"]],
                label_true=[conf["truth_label"]],
                save_path=f"pseudo_labeling/images/confuse_{conf['image_id']}.jpg",
                title=f"Pre: {label_map[conf['predict']]} True: {label_map[conf['truth_label']]} con: {conf['consensus']}"
            )

    with open("pseudo_labeling/confuse_prediction_ensemble.json", "w") as f:
        json.dump(preds, f, indent=4)
