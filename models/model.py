
import torch
import time
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain, DetBenchEval
from effdet.efficientdet import HeadNet
from tqdm import tqdm, tqdm_notebook
from utils import AverageMeter
from ensemble_boxes import weighted_boxes_fusion, nms



class Fitter:

    def __init__(self, model, device, config, base_dir='/content'):
        self.config = config
        self.epoch = 0

        self.base_dir = config.folder
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
        
        self.log_path = f'{self.base_dir}/log.txt'
        self.best_summary_loss = 10**5

        self.model = model
        self.device = device

        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ] 

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=config.lr, momentum=0.9)
        self.scheduler = config.SchedulerClass(self.optimizer, **config.scheduler_params)
        self.lr_list = []
        self.log(f'Fitter prepared. Device is {self.device}')
        

    def fit(self, train_loader, validation_loader):
        for e in range(self.config.n_epochs):
            if self.config.verbose:
                lr = self.optimizer.param_groups[0]['lr']
                timestamp = datetime.utcnow().isoformat()
                self.log(f'\n{timestamp}\nLR: {lr}')
                self.lr_list.append(lr)
            
            t = time.time()
            summary_loss = self.train_one_epoch(train_loader)

            self.log_each_epoch(t, summary_loss)

            t = time.time()
            summary_loss = self.validation(validation_loader)

            self.log_each_epoch(t, summary_loss, is_training=False)

            if summary_loss.avg < self.best_summary_loss:
                self.model.eval()
                save_path = f'{self.base_dir}/{self.config.model_name}.bin'
                self.log(f'Val loss improved from {self.best_summary_loss} to {summary_loss.avg}, save checkpoint to {save_path}')
                self.best_summary_loss = summary_loss.avg
                self.save(save_path)

            # if self.config.validation_scheduler:
            #     self.scheduler.step(metrics=1)

            self.epoch += 1

        self._plot_lr()

    def validation(self, val_loader):
        self.model.eval()
        summary_loss = AverageMeter()
        for step, (images, targets, image_ids) in enumerate(val_loader):
            with torch.no_grad():
                images = torch.stack(images)
                batch_size = images.shape[0]
                images = images.to(self.device).float()
                boxes = [target['boxes'].to(self.device).float() for target in targets]
                labels = [target['labels'].to(self.device).float() for target in targets]

                loss, _, _ = self.model(images, boxes, labels)
                summary_loss.update(loss.detach().item(), batch_size)

        return summary_loss

    def train_one_epoch(self, train_loader):
        self.model.train()
        summary_loss = AverageMeter()
        t = time.time()
        # train on colab use tqdm_notebook
        for step, (images, targets, image_ids) in  enumerate(tqdm_notebook(train_loader)):

            images = torch.stack(images)
            images = images.to(self.device).float()
            batch_size = images.shape[0]
            boxes = [target['boxes'].to(self.device).float() for target in targets]
            labels = [target['labels'].to(self.device).float() for target in targets]

            self.optimizer.zero_grad()
            
            loss, _, _ = self.model(images, boxes, labels)
            
            loss.backward()

            summary_loss.update(loss.detach().item(), batch_size)

            self.optimizer.step()

            if self.config.step_scheduler:
                self.scheduler.step()

        return summary_loss
    
    def save(self, path):
        self.model.eval()
        torch.save({
            'model_state_dict': self.model.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_summary_loss': self.best_summary_loss,
            'epoch': self.epoch,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_summary_loss = checkpoint['best_summary_loss']
        self.epoch = checkpoint['epoch'] + 1
        

    def log_each_epoch(self, t, loss, is_training=True):
        stage = 'train' if is_training else 'val'

        if is_training:
            self.log(f"\nEpoch - [{self.epoch}/{self.config.n_epochs}] - {(time.time() - t):.5f} secs")
            self.log(f"Learning rate : {self.lr_list[-1]:.4e} ")

        self.log(f"+ {stage} loss - {loss.avg:.5f}")

    def log(self, message):
        if self.config.verbose:
            print(message)

    def _plot_lr(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.lr_list)
        plt.xlabel('epoch')
        plt.ylabel('Learning rate')
        plt.suptitle(self.scheduler.name)
        plt.show()


def get_model(phi, num_classes, image_size, checkpoint_path, is_inference=False):
    config = get_efficientdet_config(f'tf_efficientdet_d{phi}')
    net = EfficientDet(config, pretrained_backbone=True)
    config.num_classes = num_classes
    config.image_size = image_size
    net.class_net = HeadNet(config,
        num_outputs=config.num_classes,
        norm_kwargs=dict(eps=.001, momentum=.01))

    if checkpoint_path and os.path.isfile(checkpoint_path):
        try:
            import gc
            checkpoint = torch.load(checkpoint_path)
            net.load_state_dict(checkpoint['model_state_dict'])
            print(f'Weight loaded from {checkpoint_path}')
            del checkpoint
            gc.collect()
        except:
            print(f'Could not load weight from {checkpoint_path}')

    if is_inference:
        net = DetBenchEval(net, config)
        net.eval()
        return net.cuda()

    return DetBenchTrain(net, config)


def collate_fn(batch):
    return tuple(zip(*batch))

def run_training(model, TrainGlobalConfig, train_dataset, val_dataset):
    device = torch.device('cuda:0')
    model.to(device)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=TrainGlobalConfig.batch_size,
        sampler=RandomSampler(train_dataset),
        pin_memory=False,
        drop_last=True,
        num_workers=TrainGlobalConfig.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=TrainGlobalConfig.batch_size,
        num_workers=TrainGlobalConfig.num_workers,
        shuffle=False,
        sampler=SequentialSampler(val_dataset),
        pin_memory=False,
        collate_fn=collate_fn,
    )

    fitter = Fitter(model=model, device=device, config=TrainGlobalConfig)
    fitter.fit(train_loader, val_loader)

def merge_preds(predictions, image_size=512,
                iou_thr=0.5, weights=None, merge_box='nms'):

    boxes = [(prediction['boxes'] / image_size).tolist()  for prediction in predictions]
    scores = [prediction['scores'].tolist()  for prediction in predictions]
    labels = [prediction['labels'].tolist() for prediction in predictions]
    if merge_box == 'wbf':
        boxes, scores, labels = weighted_boxes_fusion(boxes, scores, labels, weights=None, iou_thr=iou_thr)

    else:
        boxes, scores, labels = nms(boxes, scores, labels, weights=None, iou_thr=iou_thr)

    boxes = np.array(boxes) * image_size
    boxes = boxes.astype(np.int32).clip(min=0, max=image_size - 1)
    return boxes, np.array(scores), np.array(labels).astype(np.int32)

def make_predictions(model, images, score_thr=0.2, iou_thr=0.5, merge_box='nms'):
    im_size = images.shape[1]
    images = torch.stack((images,)).cuda().float()
    predictions = []
    with torch.no_grad():
        det = model(images, torch.tensor([1]*images.shape[0]).float().cuda())
        for i in range(images.shape[0]):
            np_det = det[i].detach().cpu().numpy()
            boxes = np_det[:, :4]
            scores = np_det[:, 4]
            labels = np_det[:, -1]
            indexes = np.where(scores >= score_thr)[0]
            boxes = boxes[indexes]
            boxes[:, 2] = boxes[:, 2] + boxes[:, 0]
            boxes[:, 3] = boxes[:, 3] + boxes[:, 1]
            boxes = boxes.astype(np.int32).clip(min=0, max=im_size)
            predictions.append({
                'boxes': boxes,
                'scores': scores[indexes],
                'labels': labels[indexes]
            })

    return merge_preds(predictions, image_size=im_size)

