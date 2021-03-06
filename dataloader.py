
from torch.utils.data import Dataset
from utils.visualization import visualize_detections
import torch
import numpy as np
import cv2
import random
import os
class DatasetRetriever(Dataset):

    def __init__(
        self,
        marking,
        image_ids,
        transforms=None,
        test=False,
        image_size=640,
        mosaic=True,
        mixup=False,
        random_intensity=0,
        save_sample=0,
        image_dir='/content',
    ):
        super().__init__()

        self.image_ids = image_ids
        self.marking = marking
        self.transforms = transforms
        self.test = test
        self.image_size = image_size
        self.image_dir = image_dir
        self.mosaic_border = [-image_size // 2, -image_size // 2]
        self.mosaic = mosaic
        self.mixup = mixup
        self.save_sample = save_sample

        self.p_normal = 0.5
        self.p_mixup = 0.0
        self.p_mosaic = 0.0
        if mosaic and not mixup:
            self.p_mosaic = 0.5
        elif mixup and not mosaic:
            self.p_mixup = 0.5
        elif mixup and mosaic:
            self.p_mosaic = self.p_mixup = self.p_normal = 0.33
        else:
            self.p_mosaic = self.p_mixup = 0.0
            self.p_normal = 1.0

        self.random_intensity = random_intensity
        # HARD CODE
        self.num_classes = 14
        self.classes = list(range(self.num_classes))
        self.class_index_array = self.create_class_index()

        # Save some example for visulizations
        if self.save_sample > 0:
            save_dir = '/content/train_images'
            not os.path.exists(save_dir) and os.mkdir(save_dir)
            for i in range(self.save_sample):
                image, target, _ = self.__getitem__(0)
                image = image.numpy()
                boxes = target['boxes'].numpy()
                boxes = boxes[:, [1, 0, 3, 2]]
                labels = target['labels'].numpy()
                image = image.transpose((1, 2, 0))
                visualize_detections(image * 255.0, boxes, labels, save_path=os.path.join(save_dir, f'train_example_{i}.png'))

    def create_class_index(self):
        if self.test:
            return

        self.image_indices = range(len(self.image_ids))
        class_map = {}
        for c in self.classes:
            records = self.marking[self.marking.class_id == c]
            class_map[c] = []
            for _, row in records.iterrows():
                idx = np.where(self.image_ids == row.image_id)
                if len(idx[0]):
                    class_map[c].append(idx[0][0])

        return class_map

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        boxes = []
        if not self.test and random.random() < self.p_mosaic:
            image, boxes, labels = self.load_mosaic(index)
        if not self.test and random.random() < self.p_mixup:
            image, boxes, labels = self.load_mixup(index)
        if self.test or not len(boxes):
            image, boxes, labels = self.load_image_and_boxes(index)

        target = {}
        target['boxes'] = boxes
        target['labels'] = torch.tensor(labels)
        target['image_id'] = torch.tensor([index])

        if self.transforms:
            for i in range(10):
                sample = self.transforms(**{
                    'image': image,
                    'bboxes': target['boxes'],
                    'labels': labels
                })
                if len(sample['bboxes']) > 0:
                    image = sample['image']
                    target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)
                    target['boxes'][:,[0,1,2,3]] = target['boxes'][:,[1,0,3,2]]  #yxyx: be warning
                    break

        return image, target, image_id

    def __len__(self) -> int:
        return len(self.image_ids)

    def load_image_and_boxes(self, index):
        # Random select class first then select image for that class
        if not self.test:
            rand_class = random.choice(self.classes)
            index = random.choice(self.class_index_array[rand_class])

        image_id = self.image_ids[index]
        image = cv2.imread(f'{self.image_dir}/{image_id}.jpg', cv2.IMREAD_COLOR).copy().astype(np.float32)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

        if self.random_intensity:
            delta = np.random.randint(-self.random_intensity, self.random_intensity)
            image += delta
            np.clip(image, 0, 255, out=image)

        image /= 255.0

        records = self.marking[self.marking['image_id'] == image_id]
        boxes = records[['x_min', 'y_min', 'x_max', 'y_max']].values
        # 0 is background
        labels = records['class_id'].values + 1

        return image, boxes, labels

    def load_mosaic(self, index):
        """ 
        load_mosaic from https://github.com/ultralytics/yolov5/blob/master/utils/datasets.py
        This implementation of cutmix author:  https://www.kaggle.com/nvnnghia 
        Refactoring and adaptation: https://www.kaggle.com/shonenkov
        """
        imsize = self.image_size
        s = imsize
    
        yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border]
        indexes = [index] + [random.randint(0, self.image_ids.shape[0] - 1) for _ in range(3)]

        result_boxes = []
        result_labels = []

        for i, index in enumerate(indexes):
            image, boxes, labels = self.load_image_and_boxes(index)
            h, w, _ = image.shape
            if i == 0:
                result_image = np.full((s * 2, s * 2, 3), 1, dtype=np.float32)
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            result_image[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b

            boxes[:, 0] += padw
            boxes[:, 1] += padh
            boxes[:, 2] += padw
            boxes[:, 3] += padh

            result_boxes.append(boxes)
            result_labels.append(labels)

        result_boxes = np.concatenate(result_boxes, 0)
        result_labels = np.concatenate(result_labels, 0)
        np.clip(result_boxes[:, 0:], 0, 2 * s, out=result_boxes[:, 0:])
        result_boxes = result_boxes.astype(np.int32)
        valid_indices = np.where((result_boxes[:, 2] - result_boxes[:, 0]) * (result_boxes[:, 3]-result_boxes[:, 1]) > 0)
        result_boxes = result_boxes[valid_indices]
        result_labels = result_labels[valid_indices]

        return result_image, result_boxes, result_labels

    def load_mixup(self, index):
        img, boxes, labels = self.load_image_and_boxes(index)
        img2, boxes2,labels2 = self.load_image_and_boxes(index)

        r = np.random.beta(8.0, 8.0)  # mixup ratio, alpha=beta=8.0
        img = (img * r + img2 * (1 - r))
        labels = np.concatenate((labels, labels2), 0)
        boxes = np.concatenate((boxes, boxes2), 0)

        return img, boxes, labels

def get_img_list_from_df(fold_df, folds):
    return fold_df[fold_df['fold'].isin(folds)]['image_id'].values
