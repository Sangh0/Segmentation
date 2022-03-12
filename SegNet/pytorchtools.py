import torch
import numpy as np

# set early stopping to check over fitting of model
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


# set mIoU score to check accuracy of model
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