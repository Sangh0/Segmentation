import argparse
from tabnanny import check
import time
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchsummary import summary

from datasets.cityscapes import CityscapesDataset
from util.torchcallback import EarlyStopping, CheckPoint
from til.torchmetrics import Metrics
from util.torchlosses import OhemCELoss
from util.torchscheduler import PolynomialLRDecay
from model import BiSeNetV2


def get_args_parser():
    parser = argparse.ArgumentParser(description='Training PIDNet', add_help=False)
    parser.add_argument('--data_dir', type=str, required=True,
                        help='directory where your dataset is located')
    parser.add_argument('--lr', default=5e-2, type=float,
                        help='learning rate')
    parser.add_argument('--epochs', default=1000, type=int,
                        help='Epochs for training model')
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch Size for training model')
    parser.add_argument('--weight_decay', default=5e-4, type=float,
                        help='weight decay of optimizer SGD')
    parser.add_argument('--num_classes', default=19, type=int, required=True,
                        help='class number of dataset')
    parser.add_argument('--lr_scheduling', default=True, type=bool,
                        help='apply learning rate scheduler')
    parser.add_argument('--check_point', default=True, type=bool,
                        help='apply check point for saving weights of model')
    parser.add_argument('--early_stop', default=False, type=bool,
                        help='apply early stopping')
    parser.add_argument('--height', default=512, type=int,
                        help='height size of image')
    parser.add_argument('--width', default=1024, type=int,
                        help='width size of image')
    return parser


def main(args):

    lr = args.lr
    num_classes = args.num_classes
    EPOCH = args.epochs
    device = torch.device('cuda')
    height = args.height
    width = args.width
    batch_size = args.batch_size

    model = BiSeNetV2(num_classes=num_classes).to(device)
    es_save_path = './weight/es_checkpoint.pt'
    cp_save_path = './weight/cp_checkpoint.pt'
    loss_func = OhemCELoss(thresh=0.7, ignore_lb=255).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=args.weight_decay)
    lr_scheduler = PolynomialLRDecay(optimizer, max_decay_steps=EPOCH)
    metric = Metrics(n_classes=num_classes, dim=1)
    checkpoint = CheckPoint(verbose=True, path=cp_save_path)
    early_stopping = EarlyStopping(patience=20, verbose=True, path=es_save_path)


    def valid_step(model, validation_data):
       model.eval()
       with torch.no_grad():
           vbatch_loss, vbatch_miou = 0, 0
           for vbatch, (val_images, val_labels) in enumerate(validation_data):
               val_images, val_labels = val_images.to(device), val_labels.to(device)

               val_outputs, v_s2, v_s3, v_s4, v_s5 = model(val_images)

               val_miou = metric.mean_iou(val_outputs, val_labels)
               vbatch_miou += val_miou.item()

               p_val_loss = loss_func(val_outputs, val_labels.squeeze())
               a_val_loss1 = loss_func(v_s2, val_labels.squeeze())
               a_val_loss2 = loss_func(v_s3, val_labels.squeeze())
               a_val_loss3 = loss_func(v_s4, val_labels.squeeze())
               a_val_loss4 = loss_func(v_s5, val_labels.squeeze())
               val_loss = p_val_loss + (a_val_loss1+a_val_loss2+a_val_loss3+a_val_loss4)
               vbatch_loss += val_loss.item()

               del val_images; del val_labels; del val_outputs
               del v_s2; del v_s3; del v_s4; del v_s5
               torch.cuda.empty_cache()

       return vbatch_loss/(vbatch+1), vbatch_miou/(vbatch+1)

    def train_on_batch(model, train_data):
        batch_loss, batch_miou = 0, 0
        for batch, (train_images, train_labels) in enumerate(train_data):
            model.train()

            train_images = train_images.to(device)
            train_labels = train_labels.to(device)

            optimizer.zero_grad()

            train_outputs, s2, s3, s4, s5 = model(train_images)

            miou = metric.mean_iou(train_outputs, train_labels)
            batch_miou += miou.item()

            p_loss = loss_func(train_outputs, train_labels.squeeze())
            a_loss1 = loss_func(s2, train_labels.squeeze())
            a_loss2 = loss_func(s3, train_labels.squeeze())
            a_loss3 = loss_func(s4, train_labels.squeeze())
            a_loss4 = loss_func(s5, train_labels.squeeze())
            loss = p_loss + (a_loss1+a_loss2+a_loss3+a_loss4)
            batch_loss += loss.item()

            loss.backward()
            optimizer.step()

            del train_images; del train_labels; del train_outputs
            del s2; del s3; del s4; del s5
            torch.cuda.empty_cache()

        return batch_loss/(batch+1), batch_miou/(batch+1)

    def train_step(model,
                   train_data,
                   validation_data,
                   epochs,
                   learning_rate_scheduler=False,
                   check_point=False,
                   early_stop=False,
                   last_epoch_save_path='./model/last_checkpoint.pt'):

        loss_list, miou_list = [], []
        val_loss_list, val_miou_list = [], []

        print('Start Model Training...!')
        start_training = time.time()
        for epoch in tqdm(range(epochs)):
            init_time = time.time()

            train_loss, train_miou = train_on_batch(model, train_data)
            loss_list.append(train_loss)
            miou_list.append(train_miou)

            val_loss, val_miou = valid_step(model, validation_data)
            val_loss_list.append(val_loss)
            val_miou_list.append(val_miou)

            end_time = time.time()

            print(f'\n[Epoch {epoch+1}/{epochs}]'
                  f'  [time: {end_time-init_time:.3f}s]'
                  f'  [lr = {optimizer.param_groups[0]["lr"]}]')
            print(f'[train loss: {train_loss:.3f}]'
                  f'  [train miou: {train_miou:.3f}]'
                  f'  [valid loss: {val_loss:.3f}]'
                  f'  [valid miou: {val_miou:.3f}]')

            if learning_rate_scheduler:
                lr_scheduler.step()

            if check_point:
                checkpoint(val_loss, model)

            if early_stop:
                assert check_point==False, 'Choose between Early Stopping and Check Point'
                early_stopping(val_loss, model)
                if early_stopping.early_stop:
                    print('\n##########################\n'
                          '##### Early Stopping #####\n'
                          '##########################')
                    break

        if early_stop==False and check_point==False:
            torch.save(model.state_dict(), last_epoch_save_path)
            print('Saving model of last epoch.')

        end_training = time.time()
        print(f'\nTotal time for training is {end_training-start_training:.3f}s')

        return {
            'model': model, 
            'loss': loss_list, 
            'miou': miou_list, 
            'val_loss': val_loss_list, 
            'val_miou': val_miou_list
            }

    train_loader = DataLoader(
        CityscapesDataset(path=args.data_dir, subset='train', cropsize=(width, height)),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    valid_loader = DataLoader(
        CityscapesDataset(path=args.data_dir, subset='valid', cropsize=(width, height)),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    summary(BiSeNetV2(num_classes=num_classes), (3,height,width), device='cpu')
    
    history = train_step(
        model,
        train_data=train_loader,
        validation_data=valid_loader,
        learning_rate_scheduler=args.lr_scheduling,
        check_point=args.check_point,
        early_stop=args.early_stop,
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser('BiSeNetV2 training', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)