import time
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from util.losses import OhemCELoss, BoundaryLoss
from util.scheduler import PolynomialLRDecay
from util.metrics import Metrics
from util.callback import EarlyStopping, CheckPoint
from combination import Combination


class TrainModel(object):
    """
    Args:
        - model: Model for training
        - lr: learning rate
        - epochs: max epochs
        - weight_decay: l2 penalty
        - num_classes: total number of class in dataset
        - t_threshold: threshold value for l3 function
        - loss_weights: weight values for entire loss function
                        From the left of the list, its lambda0, labmda1, labmda2 and lambda3
        - lr_scheduling: apply learning rate scheduler
        - check_point: save the weight with best score during training
        - early_stop: apply early stopping to avoid over-fitting
        - ignore_index: ignore index of dataset
    """
    def __init__(
        self,
        model: nn.Module,
        lr: float,
        epochs: int,
        weight_decay: float,
        num_classes: int,
        t_threshold: float=0.8,
        loss_weights: list=[0.4, 20, 1, 1],
        lr_scheduling: bool=False,
        check_point: bool=False,
        early_stop: bool=False,
        ignore_index: int=255,
    ):
    
        assert (check_point==True and early_stop==False) or (check_point==False and early_stop==True), \
            'Choose between Early Stopping and Check Point'
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.combi = Combination(
            model=model.to(self.device),
            sem_loss=OhemCELoss(balance_weights=[loss_weights[0], loss_weights[2]]).to(self.device),
            bd_loss=BoundaryLoss(coeff_bce=loss_weights[1]).to(self.device),
            metrics=Metrics(n_classes=num_classes, dim=1),
            t_threshold=t_threshold,
            ignore_index=ignore_index,
        )

        self.epochs = epochs
        self.optimizer = optim.SGD(
            self.combi.model.parameters(), 
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
        
        self.ignore_index = ignore_index

        self.writer = SummaryWriter()

    def fit(self, train_data, validation_data):

        print('Start Model Training...!')
        start_training = time.time()
        pbar = tqdm(range(self.epochs), total=int(self.epochs))
        for epoch in pbar:
            init_time = time.time()

            total_loss, bd_loss, sem_loss, pix_acc, miou = \
                self.train_on_batch(train_data, epoch)

            val_total_loss, val_bd_loss, val_sem_loss, val_pix_acc, val_miou = \
                self.validate_on_batch(validation_data, epoch)

            end_time = time.time()

            self.writer.add_scalar('lr', self.optimizer.param_groups[0]["lr"], epoch)

            print(f'\n{"="*45} Epoch {epoch+1}/{self.epochs} {"="*45}\n'
                  f'time: {end_time-init_time:.2f}s'
                  f'  lr = {self.optimizer.param_groups[0]["lr"]}')
            print(f'\ntrain average total loss: {total_loss:.3f}'
                  f'   semantic loss: {sem_loss:.3f}'
                  f'   boundary loss: {bd_loss:.3f}'
                  f'\ntrain average pixel accuracy: {pix_acc:.3f}'
                  f'   mean IOU: {miou:.3f}')
            print(f'\nvalid average total loss: {val_total_loss:.3f}'
                  f'   semantic loss: {val_sem_loss:.3f}'
                  f'   boundary loss: {val_bd_loss:.3f}'
                  f'\nvalid average pixel accuracy: {val_pix_acc:.3f}'
                  f'   mean IOU: {val_miou:.3f}')
            print(f'\n{"="*103}')

            if self.lr_scheduling:
                self.lr_scheduler.step()

            if self.check_point:
                path = f'./weights/check_point_{epoch+1}.pt'
                self.cp(val_total_loss, self.combi.model, path)
            
            if self.early_stop:
                self.es(val_total_loss, self.combi.model)
                if self.es.early_stop:
                    print('\n##########################\n'
                          '##### Early Stopping #####\n'
                          '##########################')
                    self.writer.close()
                    break
        
        end_training = time.time()
        print(f'\nTotal time for training is {end_training-start_training:.2f}s')

        return {
            'model': self.combi.model,
        }
    
    @torch.no_grad()
    def validate_on_batch(self, validation_data, epoch):
        self.combi.model.eval()
        total_loss, bd_loss, sem_loss, pix_acc, miou = 0, 0, 0, 0, 0
        for batch, (images, labels, edges) in enumerate(validation_data):
            images = images.to(self.device)
            labels = labels.to(self.device)
            edges = edges.to(self.device)

            outputs = self.combi(images, labels, edges)

            loss = outputs['total_loss'].mean()
            boundary_loss = outputs['boundary_loss'].mean().item()
            semantic_loss = outputs['semantic_loss'].mean().item()
            pixel_accuracy = outputs['pixel_accuracy'].mean().item()
            mean_iou = outputs['mean_iou'].item()

            total_loss += loss.item()
            bd_loss += boundary_loss
            sem_loss += semantic_loss
            pix_acc += pixel_accuracy
            miou += mean_iou
            
            if batch == 0:
                print(f'\n{" "*20} Validate Step {" "*20}')
            
            if (batch+1) % 10 == 0:
                print(f'\n[Batch {batch+1}/{len(validation_data)}]'
                      f'\ntotal loss: {loss:.3f}'
                      f'  boundary loss: {boundary_loss:.3f}'
                      f'  semantic loss: {semantic_loss:.3f}'
                      f'\npixel accuracy: {pixel_accuracy:.3f}'
                      f'  mean IOU: {mean_iou:.3f}')

            steps = epoch * len(validation_data) + batch
            self.writer.add_scalar('Valid/Total Loss', loss, steps)
            self.writer.add_scalar('Valid/Semantic Loss', semantic_loss, steps)
            self.writer.add_scalar('Valid/Boundary Loss', boundary_loss, steps)
            self.writer.add_scalar('Valid/Pixel Accuracy', pixel_accuracy, steps)
            self.writer.add_scalar('Valid/Mean IOU', mean_iou, steps)

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
    
    def train_on_batch(self, train_data, epoch):
        total_loss, bd_loss, sem_loss, pix_acc, miou = 0, 0, 0, 0, 0
        for batch, (images, labels, edges) in enumerate(train_data):
            self.combi.model.train()
            
            images = images.to(self.device)
            labels = labels.to(self.device)
            edges = edges.to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.combi(images, labels, edges)

            loss = outputs['total_loss'].mean()
            boundary_loss = outputs['boundary_loss'].mean().item()
            semantic_loss = outputs['semantic_loss'].mean().item()
            pixel_accuracy = outputs['pixel_accuracy'].mean().item()
            mean_iou = outputs['mean_iou'].item()

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            bd_loss += boundary_loss
            sem_loss += semantic_loss
            pix_acc += pixel_accuracy
            miou += mean_iou

            if batch == 0:
                print(f'\n{" "*20} Training Step {" "*20}')

            if (batch+1) % 20 == 0:                   
                print(f'\n[Batch {batch+1}/{len(train_data)}]'
                      f'\ntotal loss: {loss:.3f}'
                      f'  boundary loss: {boundary_loss:.3f}'
                      f'  semantic loss: {semantic_loss:.3f}'
                      f'\npixel accuracy: {pixel_accuracy:.3f}'
                      f'  mean IOU: {mean_iou:.3f}')

            steps = epoch * len(train_data) + batch
            self.writer.add_scalar('Train/Total Loss', loss, steps)
            self.writer.add_scalar('Train/Semantic Loss', semantic_loss, steps)
            self.writer.add_scalar('Train/Boundary Loss', boundary_loss, steps)
            self.writer.add_scalar('Train/Pixel Accuracy', pixel_accuracy, steps)
            self.writer.add_scalar('Train/Mean IOU', mean_iou, steps)

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