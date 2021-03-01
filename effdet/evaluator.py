import torch
import json
import logging
import time
import numpy as np
from pycocotools.cocoeval import COCOeval

class Evaluator:

    def __init__(self, pred_yxyx=False):
        self.pred_yxyx = pred_yxyx
        self.img_indices = []
        self.predictions = []

    def add_predictions(self, detections, target):
        img_indices = target['img_idx']

        detections = detections.cpu().numpy()
        img_indices = img_indices.cpu().numpy()
        for img_idx, img_dets in zip(img_indices, detections):
            self.img_indices.append(img_idx)
            self.predictions.append(img_dets)

    def _coco_predictions(self):
        # generate coco-style predictions
        coco_predictions = []
        coco_ids = []
        for img_idx, img_dets in zip(self.img_indices, self.predictions):
            img_id = self._dataset.img_ids[img_idx]
            coco_ids.append(img_id)
            if self.pred_yxyx:
                # to xyxy
                img_dets[:, 0:4] = img_dets[:, [1, 0, 3, 2]]
            # to xywh
            img_dets[:, 2] -= img_dets[:, 0]
            img_dets[:, 3] -= img_dets[:, 1]
            for det in img_dets:
                score = float(det[4])
                if score < .001:  # stop when below this threshold, scores in descending order
                    break
                coco_det = dict(
                    image_id=int(img_id),
                    bbox=det[0:4].tolist(),
                    score=score,
                    category_id=int(det[5]))
                coco_predictions.append(coco_det)
        return coco_predictions, coco_ids

    def evaluate(self):
        pass

    def save(self, result_file):
        # save results in coco style, override to save in a alternate form
        if not self.distributed or dist.get_rank() == 0:
            assert len(self.predictions)
            coco_predictions, coco_ids = self._coco_predictions()
            json.dump(coco_predictions, open(result_file, 'w'), indent=4)


class CocoEvaluator(Evaluator):

    def __init__(self, dataset, pred_yxyx=False):
        super().__init__(pred_yxyx=pred_yxyx)
        self._dataset = dataset.parser
        self.coco_api = dataset.parser.coco

    def reset(self):
        self.img_indices = []
        self.predictions = []

    def evaluate(self):
        assert len(self.predictions)
        coco_predictions, coco_ids = self._coco_predictions()
        json.dump(coco_predictions, open('./temp.json', 'w'), indent=4)
        results = self.coco_api.loadRes('./temp.json')
        coco_eval = COCOeval(self.coco_api, results, 'bbox')
        coco_eval.params.imgIds = coco_ids  # score only ids we've used
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        metric = coco_eval.stats[0]  # mAP 0.5-0.95
        self.reset()
        return metric
