import os
import pandas as pd
import numpy as np
from ensemble_boxes import weighted_boxes_fusion

def parse_pred(pred):
    span = 6
    words = pred.split(" ")
    preds = [(words[i: i + span]) for i in range(0, len(words), span)]
    boxes = []
    classes = []
    scores = []

    for pred in preds:
        cls, conf, x1, y1, x2, y2 = pred
        cls = int(cls)
        conf = float(conf)
        x1 = float(x1)
        y1 = float(y1)
        x2 = float(x2)
        y2 = float(y2)
        boxes.append([x1, y1, x2, y2])
        classes.append(cls)
        scores.append(conf)

    # normalize
    boxes = np.array(boxes)
    boxes = boxes / np.max(boxes, axis=1)[:, np.newaxis]
    return boxes.tolist(), scores, classes

def re_format_input(filename):
    df = pd.read_csv(
        os.path.join(working_dir, "input", filename)
    )

    img_ids = []
    box_list = []
    class_list = []
    score_list = []
    for i, row in df.iterrows():
        pred = row["PredictionString"]
        boxes, scores, classes = parse_pred(pred)
        box_list += boxes
        class_list += classes
        score_list += scores
        img_ids.append(row["image_id"])

    return box_list, score_list, class_list, img_ids

def re_format_ensemble(wboxes, wscores, wclasses):
    res = []
    for box, score, cls in zip(wboxes, wscores, wclasses):
        x1, y1, x2, y2 = box
        if cls == 14:
            conf = 1.0

        res.append(f"{cls} {score} {x1} {y1} {x2} {y2}")

    return " ".join(res)


if __name__ == "__main__":
    working_dir = os.getcwd()

    submission_files = os.listdir(os.path.join(working_dir, "input/submission"))
    all_preds = []
    img_ids = None

    submission_info = []
    weights = [1, 2, 3]

    assert weights is None or len(weights) == len(submission_files), \
        f"Weights shoule equal to number of submission files, but got {len(weights)} and {len(submission_files)}"

    box_list = []
    class_list = []
    score_list = []

    for sub_file, weight in zip(submission_files, weights):
        fold, model_name, val_mAP = sub_file.split("_")[1:4]
        model_name = model_name.split(".")[0]
        submission_info.append(f"{model_name}_{fold}_{val_mAP}_w{weight}")
        print(f"{model_name} {fold}: {weight}")
        boxes, scores, classes, img_ids = re_format_input("submission/" + sub_file)
        box_list.append(boxes)
        class_list.append(classes)
        score_list.append(scores)

    # Group by image id

    ens_pred = []
    for i in range(len(img_ids)):
        b, s, c = [], [], []
        for j in range(len(box_list)):
            b.append(box_list[j])
            s.append(score_list[j])
            c.append(class_list[j])

        wboxes, wscores, wclasses = weighted_boxes_fusion(
            b,
            s,
            c,
            iou_thr=0.5,
            weights=weights)

        print(np.array(wboxes).shape)

        if i % 250 == 0:
            print(f"{i}/{len(img_ids)}")

    ens_pred.append(re_format_ensemble(wboxes, wscores, wclasses))

    ens_df = pd.DataFrame({
        "image_id": img_ids,
        "PredictionString": ens_pred,
    })

    out_file = f"submission_ensemblewbf__{'_'.join(submission_info)}.csv"
    outfile = os.path.join(working_dir, 'output', out_file)
    ens_df.to_csv(out_file, index=False)

    submission_script = "kaggle competitions submit -c vinbigdata-chest-xray-abnormalities-detection -f"
    msg = f"ensemble WBF {' '.join(submission_info)}"
    print(f"{submission_script} {outfile} -m '{msg}'")
