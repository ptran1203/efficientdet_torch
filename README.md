# Data preparation
- Boxes are mergeed using WBF with iou = 0.5

# Experience log

| Model | Config | Fold | Val mAP | LB - Before apply 2 class |
|--|--|--|--|--|
|YOLOV5x | Default | 4 | 0.418 | 0.133 |
|YOLOV5x | Mosaic: 0.5, Shear: 5.0, warmup_epochs: 10 | 4 | 0.412 | 0.116 |
|YOLOV5x | translate: 0.2, scale: 0.6  | 4 | 0.406 | 0.157 |
|YOLOV5x | translate: 0.2, scale: 0.6, label smoothing: 0.01  | 4 | 0.416 | 0.097 |
|YOLOV5x | translate: 0.2, scale: 0.6  | 0 | 0.405 | 0.138 |
|YOLOV5x | translate: 0.2, scale: 0.6, mosaic: 0.7  | 0 | 0.399 | **0.170** |