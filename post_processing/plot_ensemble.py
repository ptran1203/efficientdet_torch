import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
label_map = {
    0: "Aortic enlargement",
    1: "Atelectasis",
    2: "Calcification",
    3: "Cardiomegaly",
    4: "Consolidation",
    5: "ILD",
    6: "Infiltration",
    7: "Lung Opacity",
    8: "Nodule/Mass",
    9: "Other lesion",
    10: "Pleural effusion",
    11: "Pleural thickening",
    12: "Pneumothorax",
    13: "Pulmonary fibrosis",
    14: "No finding",
}
import seaborn as sns
def visualize_detections(
    image, boxes, classes, scores, figsize=(6, 6), linewidth=2, class_map=label_map,
    box_true=None, label_true=None, save_path="", title=""
):
    """Visualize Detections"""
    if scores is None:
        scores = classes

    colors = sns.color_palette(None, len(class_map))
    image = np.array(image, dtype=np.uint8)
    plt.figure(figsize=figsize)
    plt.axis("off")
    plt.imshow(image, 'gray')
    ax = plt.gca()
    for i in range(len(boxes)):
        box, _cls, score = boxes[i], classes[i], scores[i]
        color = colors[int(_cls)]

        text = "{}: {:.2f}".format(class_map.get(_cls, "Unknown"), score)
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        patch = plt.Rectangle(
            [x1, y1], w, h, fill=False, edgecolor=color, linewidth=linewidth
        )
        ax.add_patch(patch)
        ax.text(x1, y1, text,
            bbox={"facecolor": color, "alpha": 0.5},
            clip_box=ax.clipbox,
            clip_on=True,
        )

    if save_path:
        plt.savefig(save_path)
        plt.clf()
    else:
        plt.show()

    return ax

def convert(pred, mode):
    boxes, labels, scores = [], [], []
    for p in pred:
        # center_x, center_y, (x2 - x1) / 2, (y2 - y1) / 2, cls, conf
        cenx, ceny, w, h, cls, score = p
        x1, x2, y1, y2 = getCoords([cenx, ceny, w, h], mode=mode)
        x1 /= 3
        x2 /= 3
        y1 /= 3
        y2 /= 3

        boxes.append([x1, y1, x2, y2])
        labels.append(cls)
        scores.append(score)

    return boxes, labels, scores


from ensemble import GeneralEnsemble, parse_pred, computeIOU, getCoords, get_wh, normbox
from ensemble_boxes import weighted_boxes_fusion, nms

import os
from PIL import Image
IMAGE_DIR = "/home/ubuntu/Downloads/test"

if __name__ == '__main__':
    img_dirs = "images"
    test_df = pd.read_csv('../test.csv')
    not os.path.exists(img_dirs) and os.mkdir(img_dirs)
    submissions = [
        pd.read_csv(f'input/submission/{f}') for f in os.listdir('input/submission') if f.endswith('.csv')
    ]

    mode = 'xyxy'

    image_id = '74b23792db329cff5843e36efb8aa65a'
    img = Image.open(os.path.join(IMAGE_DIR, f'{image_id}.jpg'))

    w, h = get_wh(test_df, image_id)

    submissions_pred = []

    for sub in submissions:
        submissions_pred.append(
            sub.loc[sub['image_id'] == image_id, ['PredictionString']].values[0]
        )

    # print(submissions_pred[0])
    boxes_list, labels_list, score_list = [], [], []
    for i, p in enumerate(submissions_pred):
        parsed = parse_pred(p[0], mode=mode)
        boxes, labels, scores = convert(parsed, mode=mode)
        print(len(parsed))

        visualize_detections(
            img,
            boxes,
            labels,
            scores,
            save_path=os.path.join(img_dirs, f'model_{i}.png')
        )
        
        boxes_list.append(_normbox(boxes, w, h))
        labels_list.append(labels)
        score_list.append(scores)

    # boxes, labels, scores = nms(boxes_list, labels_list, score_list, 0.5)
    boxes, scores, labels = nms(boxes_list, score_list, labels_list, iou_thr=0.5)

    # boxes, labels, scores = convert(ens, mode=mode)
    boxes = _normbox(boxes, 1/w, 1/h)
    print(len(boxes))
    visualize_detections(
        img,
        boxes,
        labels,
        scores,
        save_path=os.path.join(img_dirs, 'ens.png')
    )
