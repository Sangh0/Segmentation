import cv2
import numpy as np
from glob import glob
import albumentations as A
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class CityscapesDataset(Dataset):
    
    def __init__(self,
                 path,
                 ignore_index=255,
                 subset='train',
                 transform_=None,
                 n_classes=19,
                 ):
        assert subset in ('train', 'valid', 'test')
        self.image_files = glob(path+'/images'+subset+'/**/*.png')
        if subset not in 'test':
            self.label_files = [
                file for file in glob(path+'/labels/'+subset+'/**/*.png')
                if 'gtFine_color' in file
            ]
        self.ignore_index = ignore_index
        self.subset = subset
        self.totensor = transforms.Compose([transforms.ToTensor()])
        self.n_classes = n_classes

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        if self.subset=='train' or self.subset=='valid':
            images = cv2.cvtColor(
                cv2.imread(self.image_files[idx], cv2.IMREAD_COLOR),
                cv2.COLOR_BGR2RGB
            )
            labels = cv2.cvtColor(
                cv2.imread(self.image_files[idx], cv2.IMREAD_COLOR),
                cv2.COLOR_BGR2RGB
            )
        if self.transforms_ is not None:
            augmented = self.transform_(image=images, mask=labels)
            images = augmented['image']
            labels = augmented['mask']
        return self.totensor(images), self.totensor(self.one_hot_encoding(labels, self.n_classes))

    def one_hot_encoding(self, labels, n_classes):
        one_hot_ = np.zeros(
            (labels.shape[0], labels.shape[1], labels.shape[2], n_classes), dtype=np.float32
        )
        for i in range(labels.shape[0]):
            for j in range(labels.shape[1]):
                for k in range(labels.shape[2]):
                    one_hot_[i,j,k, labels[i,j,k]] = 1
        return one_hot_
