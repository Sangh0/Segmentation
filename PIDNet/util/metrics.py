import numpy as np
import torch
import torch.nn.functional as F

class Metrics(object):
    def __init__(self, n_classes=19, dim=1, smooth=1e-10):
        self.n_classes = n_classes
        self.dim = dim
        self.smooth = smooth

    def mean_iou(self, pred, label):
        if len(pred.shape) == 4 and pred.size(1) != 1:
            pred = torch.argmax(pred, dim=self.dim)
        elif pred.size(1) == 1:
            pred = pred.squeeze(dim=self.dim)
        if len(label.shape) == 4 and label.size(1) != 1:
            label = torch.argmax(label, dim=self.dim)
        elif label.size(1) == 1:
            label = label.squeeze(dim=self.dim)

        pred = pred.contiguous().view(-1)
        label = label.contiguous().view(-1)

        iou_per_class = []

        for clas in range(self.n_classes):
            true_class = pred == clas
            true_label = label == clas

            if true_label.long().sum().item() == 0:
                iou_per_class.append(np.nan)
            else:
                intersect = torch.logical_and(true_class, true_label).sum().float().item()
                union = torch.logical_or(true_class, true_label).sum().float().item()

                iou = (intersect+self.smooth) / (union+self.smooth)
                iou_per_class.append(iou)

        return np.nanmean(iou_per_class)

    def pixel_acc(self, pred, label):
        if len(pred.shape) == 4 and pred.size(1) != 1:
            pred = torch.argmax(pred, dim=self.dim)
        elif pred.size(1) == 1:
            pred = pred.squeeze(dim=self.dim)
        if len(label.shape) == 4 and label.size(1) != 1:
            label = torch.argmax(label, dim=self.dim)
        elif label.size(1) == 1:
            label = label.squeeze(dim=self.dim)

        pred = pred.contiguous().view(-1)
        label = label.contiguous().view(-1)

        intersection = torch.sum(pred==label)
        difference = torch.sum(pred!=label)

        union = intersection + difference * 2 + self.smooth

        return intersection/union