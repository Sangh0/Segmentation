import os
import cv2
import numpy as np

from glob import glob
from tqdm.notebook import tqdm

# Define a function for loading image with image resize
def load_image_with_resize(path, n_pixel_L, n_pixel_R, subset='train'):
    image_list, label_list = [], []
    image_files = glob(path+subset+'/'+subset+'_images/*.png')
    label_files = glob(path+subset+'/'+subset+'_labels/*.png')
    for file in tqdm(image_files):
        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (n_pixel_R, n_pixel_L), cv2.INTER_NEAREST)
        image_list.append(img)
    for file in tqdm(label_files):
        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (n_pixel_R, n_pixel_L), cv2.INTER_NEAREST)
        label_list.append(img)
    
    return np.array(image_list), np.array(label_list)