import seaborn as sns
import matplotlib.pyplot as plt
from const import LABEL_MAP

def visualize_detections(
    image, boxes, classes, scores=None, figsize=(6, 6), linewidth=1, class_map=LABEL_MAP,
):
    """Visualize Detections"""
    if scores is None:
        scores = classes

    colors = sns.color_palette(None, len(class_map))
    image = np.array(image, dtype=np.uint8)
    plt.figure(figsize=figsize)
    plt.axis("off")
    plt.imshow(image)
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

    plt.show()
