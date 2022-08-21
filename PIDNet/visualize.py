import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import torch

from util.transform import UnNormalize
from util.metrics import Metrics

class VisualizeOnNotebook(object):

    """
    Args:
        - images: 3 RGB channel images with [B, C, H, W]
        - outputs: number of class channels with [B, class number, H, W]
        - labels: 1 channel images with [B, 1, H, W]
    """

    def __init__(
        self,
        images,
        outputs,
        labels=None,
    ):
        self.labels_info = {
            0: [128,64,128],    # road
            1: [244, 35, 232],  # sidewalk
            2: [70, 70, 70],    # building
            3: [102, 102, 156], # wall
            4: [190, 153, 153], # fence
            5: [153, 153, 153], # pole
            6: [250, 170, 30],  # traffic light
            7: [220, 220, 0],   # traffic sign
            8: [107, 142, 35],  # vegetation
            9: [152, 251, 152], # terrain
            10: [70, 130, 180], # sky
            11: [220, 20, 60],  # person
            12: [255, 0, 0],    # rider
            13: [0, 0, 142],    # car
            14: [0, 0, 70],     # truck
            15: [0, 60, 100],   # bus
            16: [0, 80, 100],   # train
            17: [0, 0, 230],    # motorcycle
            18: [119, 11, 32]   # bicycle
        }

        self.images = images
        self.outputs = outputs
        self.labels = labels

        self.metric = Metrics(n_classes=len(self.labels_info.keys()), dim=0)
        self.un_normalize = UnNormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        convert_outputs = torch.argmax(outputs, dim=1)
        self.RGB_outputs = self.segmap2image(convert_outputs).astype(np.int32)
        
        if labels is not None:
            if len(labels.shape) == 4:
                convert_labels = torch.argmax(labels, dim=1)
            elif len(labels.shape) == 3:
                convert_labels = labels.squeeze(dim=1)

            self.RGB_labels = self.segmap2image(convert_labels).astype(np.int32)

    def segmap2image(self, segmaps):
        B, H, W = segmaps.size()
        map2image = np.zeros(shape=(B, H, W, 3), dtype=np.int32)
        
        for i in tqdm(self.labels_info.keys()):
            map2image[(segmaps.unsqueeze(dim=0)==i).all(axis=0)] = self.labels_info[i]
            
        del segmaps
        return map2image

    def visualize(self, num, dataset_name):

        assert dataset_name in ('train', 'valid', 'test'), \
            'choose between train, valid and test'

        if dataset_name == 'test':
            for i in range(num):
                plt.figure(figsize=(25, 8))
                plt.subplot(131)
                plt.imshow(self.un_normalize(self.images[i]).permute(1,2,0))
                plt.title('Original Image')
                plt.axis('off')
                plt.subplot(132)
                plt.imshow(self.RGB_outputs[i])
                plt.title('Predicted Image')
                plt.axis('off')
                plt.subplot(133)
                plt.imshow(self.un_normalize(self.images[i]).permute(1,2,0))
                plt.imshow(self.RGB_outputs[i], alpha=0.5)
                plt.title('Overlay Image')
                plt.axis('off')
                plt.show()

        else:
            for i in range(num):
                print(self.outputs[i].shape, self.labels[i].shape)
                miou = self.metric.mean_iou(self.outputs[i], self.labels[i])
                fig, ax = plt.subplots(2,2, figsize=(20,10))
                fig.suptitle(f'Mean IOU: {miou:.3f}', fontsize=25)
                ax[0,0].imshow(self.un_normalize(self.images[i]).permute(1,2,0))
                ax[0,0].set_title('Original Image')
                ax[0,0].axis('off')
                ax[0,1].imshow(self.RGB_outputs[i])
                ax[0,1].set_title('Predicted Image')
                ax[0,1].axis('off')
                ax[1,0].imshow(self.un_normalize(self.images[i]).permute(1,2,0))
                ax[1,0].imshow(self.RGB_outputs[i], alpha=0.5)
                ax[1,0].set_title('Overlay Image')
                ax[1,0].axis('off')
                ax[1,1].imshow(self.RGB_labels[i])
                ax[1,1].set_title('Label Image')
                ax[1,1].axis('off')
                plt.show()