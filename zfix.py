import pandas as pd
from tqdm import tqdm


def parse_pred(pred, w, h):
    span = 6
    words = pred.split(" ")
    preds = [(words[i: i + span]) for i in range(0, len(words), span)]
    res = []
    for pred in preds:
        cls, conf, x1, y1, x2, y2 = pred
        cls = int(cls)
        conf = float(conf)
        x1 = int(round(float(x1) / 1792 * w))
        y1 = int(round(float(y1) / 1792 * h))
        x2 = int(round(float(x2) / 1792 * w))
        y2 = int(round(float(y2) / 1792 * h))
        res.append([x1, y1, x2, y2, cls - 1, conf])

    return res

test_df = pd.read_csv('test.csv')

def re_format_ensemble(pred):
    res = []
    for p in pred:
        x1, y1, x2, y2, cls, conf = p
        res.append(f"{cls} {conf} {x1} {y1} {x2} {y2}")

    return " ".join(res)

submission = pd.read_csv('submission_effdetD4_fold0.csv')

for i, row in tqdm(submission.iterrows()):
    image_id = row['image_id']
    w, h = test_df.loc[test_df.image_id == image_id, ['width', 'height']].values[0]

    row["PredictionString"] = re_format_ensemble(parse_pred(row["PredictionString"], w, h))

submission.to_csv('submission_effdetD4_fold0_shifted.csv', index=False)
