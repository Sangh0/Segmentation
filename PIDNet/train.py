import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .util.losses import OhemCELoss, BoundaryLoss
from .util.scheduler import PolynomialLRDecay
from .util.metrics import Metrics
from .util.callback import EarlyStopping, CheckPoint
from .combination import Combination


class TrainModel(object):

    def __init__(
        self,
        model,
        lr,
        epochs,
        weight_decay,
        num_classes,
        lr_scheduling=False,
        check_point=False,
        early_stop=False,
        ignore_index=255,
    ):
        assert (check_point==True and early_stop==False) or (check_point==False and early_stop==True), \
            'Choose between Early Stopping and Check Point'

        self.combination = Combination(
            model=model,
            sem_loss=OhemCELoss(),
            bd_loss=BoundaryLoss(),
            metrics=Metrics(n_classes=num_classes, dim=1),
            ignore_index=self.ignore_index,
        )

        self.epochs = epochs
        self.optimizer = optim.SGD(
            model.parameters(), 
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay,
        )
        
        self.lr_scheduling = lr_scheduling
        self.lr_scheduler = PolynomialLRDecay(self.optimizer, max_decay_steps=self.epochs)

        self.check_point = check_point
        self.cp = CheckPoint(verbose=True)

        self.early_stop = early_stop
        self.es = EarlyStopping(patience=50, verbose=True, path='./weights/early_stop.pt')
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ignore_index = ignore_index

    def fit(self, train_data, validation_data):
        total_loss_list, sem_loss_list, bd_loss_list = [], [], []
        pix_acc_list, miou_list = [], []

        val_total_loss_list, val_sem_loss_list, val_bd_loss_list = [], [], []
        val_pix_acc_list, val_miou_list = [], []
        
        print('Start Model Training...!')
        start_training = time.time()
        for epoch in range(self.epochs):
            init_time = time.time()

            total_loss, bd_loss, sem_loss, pix_acc, miou = self.train_on_batch(train_data)
            total_loss_list.append(total_loss)
            sem_loss_list.append(sem_loss)
            bd_loss_list.append(bd_loss)
            pix_acc_list.append(pix_acc)
            miou_list.append(miou)

            val_total_loss, val_bd_loss, val_sem_loss, val_pix_acc, val_miou = self.validate_on_batch(validation_data)
            val_total_loss_list.append(total_loss)
            val_sem_loss_list.append(sem_loss)
            val_bd_loss_list.append(bd_loss)
            val_pix_acc_list.append(pix_acc)
            val_miou_list.append(miou)

            end_time = time.time()

            print(f'\n[Epoch {epoch+1}/{self.epochs}]'
                  f'  [time: {end_time-init_time:.2f}s]'
                  f'  [lr = {self.optimizer.param_groups[0]["lr"]}]')
            print(f'\n[train total loss: {total_loss:.3f}]'
                  f'  [train semantic loss: {sem_loss:.3f}]'
                  f'  [train boundary loss: {bd_loss:.3f}]')
            print(f'\n[valid total loss: {val_total_loss:.3f}]'
                  f'  [valid semantic loss: {val_sem_loss:.3f}]'
                  f'  [valid boundary loss: {val_bd_loss:.3f}]')

            if self.lr_scheduling:
                self.lr_scheduler.step()

            if self.check_point:
                path = f'check_point_{epoch+1}.pt'
                self.cp(val_total_loss, self.combination.model, path)
            
            if self.early_stop:
                self.es(val_total_loss, self.combination.model)
                if self.es.early_stop:
                    print('\n##########################\n'
                          '##### Early Stopping #####\n'
                          '##########################')
                    break

        return {
            'model': self.combination.model,
            'training_log': {
                'loss': total_loss_list,
                'semantic_loss': sem_loss_list,
                'boundary_loss': bd_loss_list,
                'pixel_accuracy': pix_acc_list,
                'mean_iou': miou_list,
            },
            'validating_log': {
                'loss': val_total_loss_list,
                'semantic_loss': val_sem_loss_list,
                'boundary_loss': val_bd_loss_list,
                'pixel_accuracy': val_pix_acc_list,
                'mean_iou': val_miou_list,
            }
        }

    def validate_on_batch(self, validation_data):
        self.model.eval()
        with torch.no_grad():
            total_loss, bd_loss, sem_loss, pix_acc, miou = 0, 0, 0, 0, 0
            for batch, (images, labels, edges) in enumerate(validation_data):
                images = images.to(self.device)
                labels = labels.to(self.device)
                edges = edges.to(self.device)

                outputs = self.combination(images, labels, edges)

                total_loss += outputs['total_loss'].item()
                bd_loss += outputs['boundary_loss'].item()
                sem_loss += outputs['semantic_loss'].item()
                pix_acc += outputs['pixel_accuracy'].item()
                miou += outputs['mean_iou'].item()

                del images; del labels; del edges
                del outputs
                torch.cuda.empty_cache()

            return [
                total_loss/(batch+1),
                bd_loss/(batch+1),
                sem_loss/(batch+1),
                pix_acc/(batch+1),
                miou/(batch+1),
            ]

    
    def train_on_batch(self, train_data):
        total_loss, bd_loss, sem_loss, pix_acc, miou = 0, 0, 0, 0, 0
        for batch, (images, labels, edges) in enumerate(train_data):
            self.combination.train()
            
            images = images.to(self.device)
            labels = labels.to(self.device)
            edges = edges.to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.combination(images, labels, edges)

            loss = outputs['total_loss']

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            bd_loss += outputs['boundary_loss'].item()
            sem_loss += outputs['semantic_loss'].item()
            pix_acc += outputs['pixel_accuracy'].item()
            miou += outputs['mean_iou'].item()

            del images; del labels; del edges
            del outputs
            torch.cuda.empty_cache()

        return [
            total_loss/(batch+1),
            bd_loss/(batch+1),
            sem_loss/(batch+1),
            pix_acc/(batch+1),
            miou/(batch+1),
        ]