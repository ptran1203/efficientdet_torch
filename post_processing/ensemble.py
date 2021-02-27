import os
import pandas as pd
import json
from tqdm import tqdm

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

def GeneralEnsemble(dets, iou_thresh=0.4, weights=None):
    """
    General Ensemble - find overlapping boxes of the same class and average their positions
    while adding their confidences. Can weigh different detectors with different weights.
    No real learning here, although the weights and iou_thresh can be optimized.

    Input:
    - dets : List of detections. Each detection is all the output from one detector, and
            should be a list of boxes, where each box should be on the format
            [box_x, box_y, box_w, box_h, class, confidence] where box_x and box_y
            are the center coordinates, box_w and box_h are width and height resp.
            The values should be floats, except the class which should be an integer.

    - iou_thresh: Threshold in terms of IOU where two boxes are considered the same,
                if they also belong to the same class.

    - weights: A list of weights, describing how much more some detectors should
                be trusted compared to others. The list should be as long as the
                number of detections. If this is set to None, then all detectors
                will be considered equally reliable. The sum of weights does not
                necessarily have to be 1.

    Output:
        A list of boxes, on the same format as the input. Confidences are in range 0-1.
    """
    assert type(iou_thresh) == float

    ndets = len(dets)

    if weights is None:
        w = 1 / float(ndets)
        weights = [w] * ndets
    elif not isinstance(weights[0], dict):
        assert len(weights) == ndets

        s = sum(weights)
        for i in range(0, len(weights)):
            weights[i] /= s
    else:
        # Class weight
        assert len(weights) == ndets

    out = list()
    used = list()

    for idet in range(0, ndets):
        det = dets[idet]
        for box in det:
            if box in used:
                continue

            used.append(box)
            # Search the other detectors for overlapping box of same class
            found = []
            cls = box[4]
            for iodet in range(0, ndets):
                odet = dets[iodet]

                if odet == det:
                    continue

                bestbox = None
                bestiou = iou_thresh
                for obox in odet:
                    if obox not in used:
                        # Not already used
                        if box[4] == obox[4]:
                            # Same class
                            iou = computeIOU(box, obox)
                            if iou > bestiou:
                                bestiou = iou
                                bestbox = obox

                if bestbox is not None:
                    w = get_weight(weights, iodet, cls)
                    found.append((bestbox, w))
                    used.append(bestbox)

            # Now we've gone through all other detectors
            if len(found) == 0:
                new_box = list(box)
                new_box[5] /= ndets
                out.append(new_box)
            else:
                allboxes = [(box, get_weight(weights, idet, cls))]
                allboxes.extend(found)

                xc = 0.0
                yc = 0.0
                bw = 0.0
                bh = 0.0
                conf = 0.0

                wsum = 0.0
                for bb in allboxes:
                    w = bb[1]
                    wsum += w

                    b = bb[0]
                    xc += w * b[0]
                    yc += w * b[1]
                    bw += w * b[2]
                    bh += w * b[3]
                    conf += w * b[5]

                xc /= wsum
                yc /= wsum
                bw /= wsum
                bh /= wsum

                new_box = [xc, yc, bw, bh, box[4], conf]
                out.append(new_box)
    return out


def getCoords(box):
    x1 = float(box[0]) - float(box[2]) / 2
    x2 = float(box[0]) + float(box[2]) / 2
    y1 = float(box[1]) - float(box[3]) / 2
    y2 = float(box[1]) + float(box[3]) / 2
    return x1, x2, y1, y2


def computeIOU(box1, box2):
    x11, x12, y11, y12 = getCoords(box1)
    x21, x22, y21, y22 = getCoords(box2)

    x_left = max(x11, x21)
    y_top = max(y11, y21)
    x_right = min(x12, x22)
    y_bottom = min(y12, y22)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersect_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (x12 - x11) * (y12 - y11)
    box2_area = (x22 - x21) * (y22 - y21)

    divied = box1_area + box2_area - intersect_area

    if divied == 0:
        return 0.0

    iou = intersect_area / divied
    return iou

def parse_pred(pred):
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
        res.append([x1, y1, x2 - x1, y2 - y1, cls, conf])

    return res

def re_format_input(filename):
    df = read_csv(filename)
    preds = []
    img_ids = []
    for i, row in df.iterrows():
        pred = row["PredictionString"]
        preds.append(parse_pred(pred))
        img_ids.append(row["image_id"])

    return preds, img_ids

def re_format_ensemble(pred):
    res = []
    for p in pred:
        x1, y1, w, h, cls, conf = p
        if cls == 14:
            conf = 1.0
        res.append(f"{cls} {conf} {x1} {y1} {x1 + w} {y1 + h}")

    return " ".join(res)

def normalize_weight(weights):
    s = {k: 0 for k in weights[0].keys()}

    # To prevent div zero
    sigma = 0.0001
    for weight in weights:
        for cls, w in weight.items():
            w += sigma
            s[cls] += w

    for weight in weights:
        for cls, w in weight.items():
            weight[cls] = (weight[cls] + sigma) / s[cls]

def get_weight_from_json(submission_files):
    weights = []
    for f in submission_files:
        f = f.split(".")[0]
        splited = f.split("_")
        if len(splited) == 4:
            _, fold, mname, pyramid_levels = splited
        else:
            fold, mname, pyramid_levels = splited
        f = "_".join([mname, fold, pyramid_levels])
        json_f = f + ".json"
        json_f = os.path.join(working_dir, "input/weights", json_f)

        if not os.path.isfile(json_f):
            return False

        weight = json.load(open(json_f))
        weight = {int(k): v for k, v in weight.items()}
        weights.append(weight)

    return weights


if __name__ == "__main__":
    submission_files = os.listdir(os.path.join(working_dir, "input/submission"))
    submission_files = [x for x in submission_files if x.endswith(".csv")]
    all_preds = []
    img_ids = None

    submission_info = []
    weights = get_weight_from_json(submission_files)
    weights = None
    if weights:
        normalize_weight(weights)
        print(submission_files)
        print(json.dumps(weights, indent=4))
    else:
        weights = [1] * len(submission_files)

    weights = [3, 7]
    assert weights is None or len(weights) == len(submission_files), \
        f"Weights shoule equal to number of submission files, but got {len(weights)} and {len(submission_files)}"

    for sub_file, weight in zip(submission_files, weights):
        print(sub_file)
        splited = sub_file.split("_")
        fold, model_name = splited[1:3]
        model_name = model_name.split(".")[0]
        if isinstance(weight, dict):
            weight = round(sum(weight.values()) / 14, 2)
        submission_info.append(f"{model_name}_{fold}_w{weight}")

        print(f"{model_name} {fold}: {weight}")
        pred, img_ids = re_format_input("submission/" + sub_file)
        all_preds.append(pred)

    ens_pred = []
    iou_thresh = 0.5
    for i in tqdm(range(len(img_ids))):
        ens = GeneralEnsemble([preds[i] for preds in all_preds], iou_thresh=iou_thresh, weights=weights)
        ens_pred.append(re_format_ensemble(ens))

    ens_df = pd.DataFrame({
        "image_id": img_ids,
        "PredictionString": ens_pred,
    })

    out_file = f"ensemble_iou{iou_thresh}.csv"
    write_csv(ens_df, out_file)

    submission_script = "kaggle competitions submit -c vinbigdata-chest-xray-abnormalities-detection -f"
    msg = f"ensemble iou {iou_thresh} {' '.join(submission_info)}"
    print(f"{submission_script} {os.path.join(OUTPUT_DIR, out_file)} -m '{msg}'")
