import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, weight=None, reduction='mean'):
        super(FocalLoss, self).__init__(weight, reduction=reduction)
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction=self.reduction, weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = (self.alpha * (1-pt)**self.gamma * ce_loss).mean()

        return focal_loss
        

# Define Dice Loss function
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, outputs, targets, smooth=1):
        outputs = F.sigmoid(outputs)

        outputs = outputs.view(-1)
        targets = targets.view(-1)

        intersection = (outputs*targets).sum()
        union = outputs.sum()+targets.sum()
        
        loss = 1-(2*intersection+smooth)/(union+smooth)
        return loss


# Define OHEM Cross Entorpy Loss function
class OhemCELoss(nn.Module):

    def __init__(self, thresh, ignore_lb=255):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, requires_grad=False, dtype=torch.float)).cuda()
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

    def forward(self, logits, labels):
        n_min = labels[labels != self.ignore_lb].numel() // 16
        loss = self.criteria(logits, labels).view(-1)
        loss_hard = loss[loss > self.thresh]
        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)
        return torch.mean(loss_hard)