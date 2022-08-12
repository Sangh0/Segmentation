import numpy as np
import cv2
from PIL import Image
from glob import glob

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from ..util.transform import (
    Compose, RandomCrop, HorizontalFlip, RandomScale, ColorJitter,
)

"""
path : camvid/
├── train
│    ├─ train_images
│       ├─ image1.jpg
│       ├─ ...
│    ├─ train_labels
│       ├─ label1.jpg 
│       ├─ ...
├── valid
│    ├─ valid_images
│       ├─ image1.jpg
│       ├─ ...
│    ├─ valid_labels
│       ├─ label1.jpg
│       ├─ ... 
├── test
│    ├─ test_images
│       ├─ image1.jpg
│       ├─ ...
│    ├─ test_labels
│       ├─ label1.jpg
│       ├─ ... 
"""

"""
Format of label: 3 RGB channels with color segmentation map
If you have integer label with segmentation map, go to datasets/cityscapes.py
"""

class CamVidDataset(Dataset):
    def __init__(
        self,
        path,
        subset,
        cropsize=None,
        ignore_index=255,
    ):
        assert subset in ('train', 'valid', 'test')
        self.image_files = glob(path+'/'+subset+'/'+subset+'_images/*.png')
        self.color_files = glob(path+'/'+subset+'/'+subset+'_labels/*.png')
        self.subset = subset
        self.ignore_index = ignore_index
        
        self.totensor = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.augment = Compose([
            HorizontalFlip(),
            RandomScale((0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)),
            RandomCrop(cropsize)
        ])
        self.color_list = [[0,128,192], [128,0,0], [64,0,128], 
                           [192,192,128], [64,64,128], [64,64,0],
                           [128,64,128], [0,0,192], [192,128,128],
                           [128,128,128], [128,128,0]]
        self.classes = 11
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        images = Image.open(self.image_files[idx]).convert('RGB')
        colors = Image.open(self.color_files[idx]).convert('RGB')
        labels = self.color2label(colors)
        if self.subset == 'train':
            im_lb = dict(im=images, lb=Image.fromarray(labels))
            im_lb = self.augment(im_lb)
            images, labels = im_lb['im'], im_lb['lb']
        labels = np.array(labels)[np.newaxis,:]
            
        return self.totensor(images), torch.LongTensor(labels), self.get_edge(labels)
            
    def color2label(self, color_map):
        color_map = np.array(color_map)
        label = np.ones(color_map.shape[:2]) * self.ignore_index
        for i, color in enumerate(self.color_list):
            label[(color_map==color).sum(2)==3] = i
        return label

    def get_edge(self, label, edge_size=4):
        edge = cv2.Canny(label.squeeze(), 0.1, 0.2)
        kernel = np.ones((edge_size, edge_size), np.uint8)
        edge = (cv2.dilate(edge, kernel, iterations=1) > 50)
        edge = np.expand_dims(edge, axis=0)
        return torch.LongTensor(edge)


def load_cityscapes_dataset(
    path: str, 
    height: int=720,
    width: int=960,
    get_val_set: bool=True, 
    batch_size: int=12,
):
    out = {
        'train': DataLoader(
            CamVidDataset(path=path, subset='train', cropsize=(width,height)),
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        ),

        'valid': DataLoader(
            CamVidDataset(path=path, subset='valid', cropsize=(width,height)),
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )
    }
    return out