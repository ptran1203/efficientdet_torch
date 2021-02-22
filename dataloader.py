
from torch.utils.data import Dataset
import torch
import numpy as np
import cv2
class DatasetRetriever(Dataset):

    def __init__(
        self,
        marking,
        image_ids,
        transforms=None,
        test=False,
        root_path='/content',
        fold_dict={}
    ):
        super().__init__()

        self.image_ids = image_ids
        self.marking = marking
        self.transforms = transforms
        self.test = test
        self.root_path = root_path
        self.fold_dict = fold_dict

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        
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
        image_id = self.image_ids[index]        
        fold_id = self.fold_dict[image_id]
        image = cv2.imread(f'{self.root_path}/fold{fold_id}/{image_id}.jpg', cv2.IMREAD_COLOR).copy().astype(np.float32)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        records = self.marking[self.marking['image_id'] == image_id]
        boxes = records[['x_min', 'y_min', 'x_max', 'y_max']].values
        labels = records['class_id'].values
        return image, boxes, labels
