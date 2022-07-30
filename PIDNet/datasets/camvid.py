import numpy as np
from PIL import Image
from glob import glob

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from ..util.transform import Compose, RandomCrop, HorizontalFlip, RandomScale, ColorJitter

"""
path : cityscapes/
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
            
        return self.totensor(images), torch.LongTensor(labels)
            
    def color2label(self, color_map):
        color_map = np.array(color_map)
        label = np.ones(color_map.shape[:2]) * self.ignore_index
        for i, color in enumerate(self.color_list):
            label[(color_map==color).sum(2)==3] = i
        return label