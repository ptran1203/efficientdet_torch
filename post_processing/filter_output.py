import pandas as pd
import os
import numpy as np

working_dir = os.getcwd()

# Obtain file from ensemble output
submission_files = [f for f in os.listdir(os.path.join(working_dir, "output")) if "filtered" not in f]
if not submission_files:
    raise Exception("No submission files found")

submission_file = "output/" + submission_files[0]
submission_file_filtered = submission_file.replace(".csv", "").split("/")[1] + "_filtered.csv"
print(f"Run filter for file {submission_file}")

nofindings_str = "14 1.0 0 0 1 1"

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

filtered = 0

def filter_by_thres(pred, threshold=0.5):
    global filtered

    all_preds = []
    span = 6
    words = pred.split(" ")
    preds = [" ".join(words[i: i + span]) for i in range(0, len(words), span)]
    class_0 = []
    class_3 = []

    for p in preds:
        cls, confi = p.split(" ")[:2]
        confi = float(confi)
        if confi >= threshold:
            if int(cls) == 3 and confi >= 0.3:
                class_3.append((p, confi))
                all_preds.append(p)
                continue
            elif int(cls) == 0 and confi >= 0.3:
                class_0.append((p, confi))
                all_preds.append(p)
                continue
            else:
                all_preds.append(p)
        else:
            filtered += 1

    return " ".join(all_preds)



# filter by classification model

def get_cls_pred(fname):
    return read_csv(f"class_filter/{fname}.csv")

import sys
fname = sys.argv[1] if len(sys.argv) > 1 else "effb5"

print(fname)

cls_pred = get_cls_pred(fname)
submission = pd.read_csv(os.path.join(working_dir, submission_file))
merged = submission.merge(cls_pred, on="image_id", how="left")

low_confi = 0
high_confi = 0.95

c0, c1, c2 = 0, 0, 0
for i in range(len(merged)):
    p = merged.loc[i, "no_findings"]

    if fname == 'effb5':
        p = 1 - p

    if p < low_confi:
        c0 += 1
    elif low_confi <= p < high_confi:
        c1 += 1
        merged.loc[i, "PredictionString"] += f" 14 {p} 0 0 1 1"
    else:
        c2 += 1
        merged.loc[i, "PredictionString"] = "14 1.0 0 0 1 1"

merged = merged[["image_id", "PredictionString"]]

print(f"Keep {c0}, Add {c1}, replace {c2}")

merged.to_csv(os.path.join(working_dir, "output", submission_file_filtered), index=False)

print(f"Filter prediction less than 0.1 {filtered}")
submission_script = "kaggle competitions submit -c vinbigdata-chest-xray-abnormalities-detection -f"
msg = "2 class filtering"
print(f"{submission_script} {os.path.join(working_dir, 'output', submission_file_filtered)} -m '{msg}'")