import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def text_to_dict(text):
    splited = text.split('\n')
    total = splited[0]
    per_class = splited[1:]

    total = float(total.split(':')[-1].strip())

    per_class_map = {}

    for p in per_class:
        k, v = p.split(':')
        k = k.strip()
        v = float(v.strip())
        per_class_map[k] = v

    per_class_map['total'] = total
    return per_class_map


def draw_mAP(fname):
    with open(fname, 'r') as f:
        raw_text = f.read()

    data = text_to_dict(raw_text)
    x = pd.Series(data).nlargest(15).plot(kind='barh')
    plt.show()


draw_mAP('val_fold4')