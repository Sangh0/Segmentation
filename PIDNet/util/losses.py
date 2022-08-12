import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    
    def __init__(
        self, 
        ignore_label=-1, 
        weight=None,
        balance_weights=[0.5, 0.5],
        sb_weights=0.5,
    ):
        super(CrossEntropyLoss, self).__init__()
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_label,
        )
        self.balance_weights = balance_weights
        self.sb_weights = sb_weights

    def _forward(self, score, target):
        loss = self.criterion(score, target)
        return loss

    def forward(self, score, target):

        if len(score) == 1:
            score = [score]
            
        if len(self.balance_weights) == len(score):
            return sum([w * self._forward(x, target) for (w, x) in zip(self.balance_weights, score)])
        elif len(score) == 1:
            return self.sb_weights * self._forward(score[0], target)
        
        else:
            raise ValueError("lengths of prediction and target are not identical!")
        

class OhemCELoss(nn.Module):
    
    def __init__(
        self, 
        ignore_label=255, 
        thres=0.7,
        min_kept=100000, 
        weight=None,
        balance_weights=[0.5, 0.5],
        sb_weights=0.5,
    ):
        super(OhemCELoss, self).__init__()
        self.thresh = thres
        self.min_kept = max(1, min_kept)
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_label,
            reduction='none',
        )
        self.balance_weights = balance_weights
        self.sb_weights = sb_weights

    def _ce_forward(self, score, target):
        loss = self.criterion(score, target)
        return loss

    def _ohem_forward(self, score, target, **kwargs):
        pred = F.softmax(score, dim=1)
        pixel_losses = self.criterion(score, target).contiguous().view(-1)
        mask = target.contiguous().view(-1) != self.ignore_label

        tmp_target = target.clone()
        tmp_target[tmp_target == self.ignore_label] = 0
        pred = pred.gather(1, tmp_target.unsqueeze(1))
        pred, ind = pred.contiguous().view(-1,)[mask].contiguous().sort()
        min_value = pred[min(self.min_kept, pred.numel() - 1)]
        threshold = max(min_value, self.thresh)

        pixel_losses = pixel_losses[mask][ind]
        pixel_losses = pixel_losses[pred < threshold]
        return pixel_losses.mean()

    def forward(self, score, target):
        if not (isinstance(score, list) or isinstance(score, tuple)):
            score = [score]

        if len(self.balance_weights) == len(score):
            functions = [self._ce_forward] * \
                (len(self.balance_weights) - 1) + [self._ohem_forward]
            return sum([
                w * func(x, target)
                for (w, x, func) in zip(self.balance_weights, score, functions)
            ])
        
        elif len(score) == 1:
            return self.sb_weights * self._ohem_forward(score[0], target)
        
        else:
            raise ValueError("lengths of prediction and target are not identical!")
        

class BoundaryLoss(nn.Module):
    
    def __init__(self, coeff_bce=20.):
        super(BoundaryLoss, self).__init__()
        self.coeff_bce = coeff_bce

    def weighted_bce(self, score, target):
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