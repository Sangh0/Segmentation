import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class CELoss(nn.Module):
    
    def __init__(self, ignore_index=255, weight=None):
        super(CELoss, self).__init__()
        self.ignore_index = ignore_index
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_index,
        )

    def forward(self, pred, label):
        return self.criterion(pred, label)
        

class OhemCELoss(nn.Module):

    def __init__(self, thresh=0.7, ignore_index=255, weight=None):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, requires_grad=False, dtype=torch.float)).cuda()
        self.ignore_lb = ignore_index
        self.criteria = nn.CrossEntropyLoss(
            weight=weight, 
            ignore_index=ignore_index, 
            reduction='none',
        )

    def forward(self, logits, labels):
        n_min = labels[labels != self.ignore_lb].numel() // 16
        loss = self.criteria(logits, labels).view(-1)
        loss_hard = loss[loss > self.thresh]
        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)
        return torch.mean(loss_hard)


class BoundaryLoss(nn.Module):
    
    def __init__(self, coeff_bce=20.):
        super(BoundaryLoss, self).__init__()
        self.coeff_bce = coeff_bce

    def weighted_bce(self, pred, label):
        N, C, H, W = pred.size()
        pred = pred.permute(0,2,3,1).contiguous().view(1, -1)
        label = label.view(1, -1)

        pos_index = (label == 1)
        neg_index = (label == 0)

        weight = torch.zeros_like(pred)
        pos_num = pos_index.sum()
        neg_num = neg_index.sum()
        sum_num = pos_num + neg_num
        weight[pos_index] = neg_num * 1.0 / sum_num
        weight[neg_index] = pos_num * 1.0 / sum_num

        loss = F.binary_cross_entropy_with_logits(pred, label, weight, reduction='mean')

        return loss

    def forward(self, pred, label):
        return self.coeff_bce * self.weighted_bce(pred, label)