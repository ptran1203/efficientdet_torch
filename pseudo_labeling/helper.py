import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

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

def visualize_detections(
    image, boxes, classes, scores, figsize=(15, 15), linewidth=1, class_map={},
    box_true=None, label_true=None, save_path="", title=""
):
    """Visualize Detections"""
    image = np.array(image, dtype=np.uint8)
    plt.figure()
    plt.axis("off")
    plt.imshow(image, "gray")
    plt.title(title)
    ax = plt.gca()
    for i in range(len(boxes)):
        box, _cls, score = boxes[i], classes[i], scores[i]
        color = [1, 0, 0]
        text = "{}: {:.2f}".format(str(_cls), score)
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        patch = plt.Rectangle(
            [x1, y1], w, h, fill=False, edgecolor=color, linewidth=linewidth
        )
        ax.add_patch(patch)
        ax.text(
            x1,
            y1,
            text,
            bbox={"facecolor": color, "alpha": 0.5},
            clip_box=ax.clipbox,
            clip_on=True,
        )

    if box_true is not None and label_true is not None:
        for i in range(len(box_true)):
            color = [1, 1, 1]
            box_t, cls_t = box_true[i], label_true[i]
            text = "{}".format(cls_t)
            x1, y1, x2, y2 = box_t
            w, h = x2 - x1, y2 - y1
            patch = plt.Rectangle(
                [x1, y1], w, h, fill=False,
                edgecolor=color, linewidth=1
            )
            ax.add_patch(patch)
            ax.text(
                x1,
                y1,
                text,
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

def _clean_string(bbox_string):
    return bbox_string.replace("[", "").replace("]", "").strip()

def to_array(string):
    return np.fromstring(_clean_string(string), sep=' ')

def parse_boxes(values):
    boxes = values[1:][:-1].split("\n")

    return np.array([to_array(v) for v in boxes])

def computeIOU(box1, box2):
    x11, y11, x12, y12 = box1
    x21, y21, x22, y22 = box2

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

def parse_prediction_string(pred):
    span = 6
    words = pred.split(" ")
    preds = [(words[i: i + span]) for i in range(0, len(words), span)]
    boxes, scores, labels = [], [], []
    for pred in preds:
        cls, conf, x1, y1, x2, y2 = pred
        cls = int(cls)
        conf = float(conf)
        x1 = float(x1)
        y1 = float(y1)
        x2 = float(x2)
        y2 = float(y2)

        boxes.append([x1, y1, x2, y2])
        scores.append(conf)
        labels.append(cls)

    return boxes, scores, labels
