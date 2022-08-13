import numpy as np
from glob import glob
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from util.torchtransform import (
    RandomCrop, HorizontalFlip, RandomScale, ColorJitter, Compose
)


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
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.train_augment = Compose([
            ColorJitter(
                brightness=0.5,
                contrast=0.5,
                saturation=0.5),
            HorizontalFlip(),
            RandomScale((0.75, 1.0, 1.25, 1.5, 1.75, 2.0)),
            RandomCrop(cropsize)
        ])
        self.valid_augment = Compose([
            RandomCrop
        ])
        self.mapping_20 = {
            0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255, 7: 0, 
            8: 1, 9: 255, 10: 255, 11: 2, 12: 3, 13: 4, 14: 255, 15: 255, 
            16: 255, 17: 5, 18: 255, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 
            24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 29: 255, 30: 255, 31: 16, 
            32: 17, 33: 18, -1: -1
        }
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        images = Image.open(self.image_files[idx]).convert('RGB')
        if self.subset=='train':
            labels = Image.open(self.label_files[idx]).convert('L')
            im_lb = dict(im=images, lb=labels)
            im_lb = self.train_augment(im_lb)
            images, labels = im_lb['im'], im_lb['lb']
            labels = np.array(labels).astype(np.int64)[np.newaxis,:]
            return self.totensor(images), self.convert_label(labels)
        elif self.subset=='valid':
            labels = Image.open(self.label_files[idx]).convert('L')
            labels = np.array(labels).astype(np.int64)[np.newaxis,:]
            return self.totensor(images), self.convert_label(labels)
        else:
            return self.totensor(images)
    
    def convert_label(self, label):
        for k in self.mapping_20:
            label[label==k] = self.mapping_20[k]
        return torch.LongTensor(label)
