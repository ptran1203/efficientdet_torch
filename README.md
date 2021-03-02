# Data preparation
- Boxes are mergeed using WBF with iou = 0.5

# Experience log

| Model | Config | Fold | Val mAP | LB - Before apply 2 class |
|--|--|--|--|--|
|YOLOV5x | Default | 4 | 0.418 | 0.133 |
|YOLOV5x | Mosaic: 0.5, Shear: 5.0, warmup_epochs: 10 | 4 | - | - |
|YOLOV5x | Default | 1 | - | - |
