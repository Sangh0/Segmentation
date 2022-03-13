import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# We set early stopping to check over fitting of model
# we use early stopping because of over-fitting
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        
    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.3f} --> {val_loss:.3f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


# Define focal loss function (this function was introduced in RetinaNet paper)
# The loss function of SegNet paper is cross entropy loss 
# but this function has a problem that called data imabalance
# so, in this paper, author use an median frequency balancing to solve its problem
# I think I can solve this problem using focal loss
# this function is used to weight small objects
# so I will use focal loss function to train SegNet 
"""
default values:
    alpha : 0.25 (by RetinaNet paper)
        If alpha is 0.25, the weight of foreground is 0.25 and the weight of background is 0.75
    gamma : 2 (by paper)
"""

class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, alpha=0.25, gamma=2, weight=None, reduction='mean'):
        super(FocalLoss, self).__init__(weight, reduction=reduction)
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight

    def forward(self, input_, target):
        ce_loss = F.cross_entropy(input_, target, reduction=self.reduction, weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = (self.alpha * (1-pt)**self.gamma * ce_loss).mean()

        return focal_loss


# We set mIoU score to check accuracy of model
# this function can use only a type of torch tensor
class Metrics:
    def __init__(self, n_classes=12, dim=1, smooth=1e-10):
        self.n_classes = n_classes
        self.dim = dim
        self.smooth = smooth

    def get_miou(self, pred_mask, label_mask):
        pred_mask = torch.argmax(pred_mask, dim=self.dim)
        pred_mask = pred_mask.contiguous().view(-1)
        label_mask = torch.argmax(label_mask, dim=self.dim)
        label_mask = label_mask.contiguous().view(-1)

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