import numpy as np
import torch
import torch.nn.functional as F

# We set mIoU score to check accuracy of model
# this function can use only a type of torch tensor
class Metrics:
    def __init__(self, n_classes=12, dim=1, smooth=1e-10):
        self.n_classes = n_classes
        self.dim = dim
        self.smooth = smooth

    def mean_iou(self, pred_mask, label_mask):
        # pred_mask = torch.argmax(pred_mask, dim=self.dim)
        pred_mask = pred_mask.view(-1)
        # label_mask = torch.argmax(label_mask, dim=self.dim)
        label_mask = label_mask.view(-1)

        iou_per_class = []

        for clas in range(self.n_classes):
            true_class = pred_mask == clas
            true_label = label_mask == clas

            if true_label.long().sum().item() == 0:
                iou_per_class.append(np.nan)
            else:
                intersect = torch.logical_and(true_class, true_label).sum().float().item()
                union = torch.logical_or(true_class, true_label).sum().float().item()

                iou = (intersect+self.smooth) / (union+self.smooth)
                iou_per_class.append(iou)

        return np.nanmean(iou_per_class)

    def f1_score(self, pred_mask, label_mask):
        pred_mask = F.sigmoid(pred_mask)

        pred_mask = pred_mask.view(-1)
        label_mask = label_mask.view(-1)

        intersection = (pred_mask*label_mask).sum()
        union = pred_mask.sum() + label_mask.sum()

        f1_score = 2.0 * (intersection+self.smooth) / (union+self.smooth)

        return f1_score