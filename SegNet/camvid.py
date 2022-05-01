import cv2
import numpy as np
import pandas as pd
from glob import glob
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class CamVidDataset(Dataset):
    def __init__(self, 
                 path, 
                 class_path,
                 transforms_, 
                 subset='train'):

        self.image_files = glob(path+subset+'/'+subset+'_images/*.png')
        self.label_files = glob(path+subset+'/'+subset+'_labels/*.png')
        class_map = pd.read_csv(class_path, index_col=0)
        self.rgb_class = {name:list(class_map.loc[name, :][:3]) for name in class_map.index}
        self.transforms_ = transforms.Compose(transforms_)
        self.subset = subset
        
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        images = cv2.cvtColor(
            cv2.imread(self.image_files[idx], cv2.IMREAD_COLOR),
            cv2.COLOR_BGR2RGB
        )
        labels = cv2.cvtColor(
            cv2.imread(self.label_files[idx], cv2.IMREAD_COLOR),
            cv2.COLOR_BGR2RGB
        )
        return self.transforms_(images), self.transforms_(self.one_hot_encoding(labels))

    def one_hot_encoding(self, labels):
        semantic_map = []
        for color in list(self.rgb_data.values()):
            equality = np.equal(labels, color)
            class_map = np.all(equality, axis=-1)
            semantic_map.append(class_map)
        semantic_map = np.stack(semantic_map, axis=-1)
        return np.float32(semantic_map)