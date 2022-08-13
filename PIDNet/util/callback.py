import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CheckPoint(object):
    def __init__(self, verbose=False, trace_func=print):
        self.verbose = verbose
        self.trace_func = trace_func
        self.val_loss_min = np.Inf
        self.best_score = None

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score > self.best_score:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            self.trace_func(f'\nValidation loss decreased ({self.val_loss_min:.3f} --> {val_loss:.3f}).  Saving model ...')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss


class EarlyStopping(object):
    def __init__(self, patience=7, verbose=False, delta=0, path='es_checkpoint.pt', trace_func=print):
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
            self.save_model(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'\nEarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_model(val_loss, model)
            self.counter = 0
            
    def save_model(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'\nValidation loss decreased ({self.val_loss_min:.3f} --> {val_loss:.3f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss