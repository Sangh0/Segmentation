import numpy as np
import torch
import torch.nn.functional as F

class Metrics(object):
    def __init__(self, n_classes=19, dim=1, smooth=1e-10):
        self.n_classes = n_classes
        self.dim = dim
        self.smooth = smooth

    def mean_iou(self, pred_mask, label_mask):
        pred_mask = torch.argmax(pred_mask, dim=self.dim)
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

    def pixel_acc(self, pred, label):
        _, preds = torch.max(pred, dim=self.dim)
        valid = (label >= 0).long()
        acc_sum = torch.sum(valid * (preds == label).long())
        pixel_sum = torch.sum(valid)
        acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
        return acc