import numpy as np
from PIL import Image
from glob import glob

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from ..util.transform import Compose, RandomCrop, HorizontalFlip, RandomScale, ColorJitter

"""
path : cityscapes/
├── images
│    ├─ train
│       ├─ image1.jpg
│       ├─ ...
│    ├─ valid
│       ├─ image1.jpg 
│       ├─ ...
│    ├─ test
│       ├─ image1.jpg 
│       ├─ ...
├── labels
│    ├─ train
│       ├─ label1.jpg
│       ├─ ...
│    ├─ valid
│       ├─ label1.jpg
│       ├─ ... 
"""

"""
Format of label: single channel with integer labels
"""

class CityscapesDataset(Dataset):
    def __init__(
        self,
        path,
        subset,
        cropsize=None,
        ignore_index=255,
    ):
        assert subset in ('train', 'valid', 'test')
        self.image_files = glob(path+'/images/'+subset+'/**/*.png')
        if subset not in 'test':
            self.label_files = [
                file for file in glob(path+'/labels/'+subset+'/**/*.png') \
                if 'gtFine_labelIds' in file
            ]
        self.subset = subset
        self.cropsize = cropsize
        self.ignore_index = ignore_index
        self.totensor = transforms.Compose([
            transforms.ToTensor(),
#             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.augment = Compose([
            HorizontalFlip(),
            RandomScale((0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)),
            RandomCrop(cropsize)
        ])
        self.mapping_20 = {
            0: ignore_index, 1: ignore_index, 2: ignore_index, 3: ignore_index, 
            4: ignore_index, 5: ignore_index, 6: ignore_index, 7: 0, 8: 1, 
            9: ignore_index, 10: ignore_index, 11: 2, 12: 3, 13: 4, 
            14: ignore_index, 15: ignore_index, 16: ignore_index, 17: 5, 
            18: ignore_index, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 
            25: 12, 26: 13, 27: 14, 28: 15, 29: ignore_index, 30: ignore_index, 
            31: 16, 32: 17, 33: 18, -1: ignore_index
        }
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        images = Image.open(self.image_files[idx]).convert('RGB')
        if self.subset=='train':
            labels = Image.open(self.label_files[idx]).convert('L')
            im_lb = dict(im=images, lb=labels)
            im_lb = self.augment(im_lb)
            images, labels = im_lb['im'], im_lb['lb']
            labels = np.array(labels)[np.newaxis,:]
            return self.totensor(images), self.convert_label(labels)
        elif self.subset=='valid':
            labels = Image.open(self.label_files[idx]).convert('L')
            labels = np.array(labels)[np.newaxis,:]
            return self.totensor(images), self.convert_label(labels)
        else:
            return self.totensor(images)
    
    def convert_label(self, label):
        for k in self.mapping_20:
            label[label==k] = self.mapping_20[k]
        return torch.LongTensor(label)