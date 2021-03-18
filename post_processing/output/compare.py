import pandas as pd
import numpy as np

def tostr(pred):
    pred = pred[: 100]
    return ' '.join([' '.join([str(i) for i in p]) for p in pred])

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
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        if mode == 'center':
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            res.append([center_x, center_y, (x2 - x1), (y2 - y1), cls, conf])
        elif mode == 'xyxy':
            res.append([cls, conf, x1, y1, x2, y2])
        elif mode =='xywh':
            res.append([x1, y1, x2 - x1, y2 - y1, cls, conf])
        else:
            raise ValueError('mode')

    res = np.array(res)
    res = res[res[:, 1].argsort()][::-1]
    return tostr(res.tolist())

def get_pred(df, image_id):
    pred = df.loc[df.image_id == image_id, ['PredictionString']].values[0][0]
    return parse_pred(pred)

f1 = pd.read_csv('./].csv')
f2 = pd.read_csv('ensemble_iou0.5_nms.csv')

img_id = '83caa8a85e03606cf57e49147d7ac569'

pred1 = get_pred(f1, img_id)
pred2 = get_pred(f2, img_id)

print(pred1)
print('---------------')
print(pred2)