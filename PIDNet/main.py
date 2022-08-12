import argparse
from ast import parse

import torch
from torch.utils.data import DataLoader
from torchsummary import summary

from .models.pidnet import get_model
from .train import TrainModel
from .datasets.cityscapes import CityscapesDataset

def get_args_parser():
    parser = argparse.ArgumentParser(description='Training PIDNet', add_help=False)
    parser.add_argument('--data_dir', type=str, required=True,
                        help='directory where your dataset is located')
    parser.add_argument('--model_name', default='pidnet_s', type=str, required=True,
                        help='Name of PIDNet model, select between pidnet_s, pidnet_m and pidnet_l')
    parser.add_argument('--lr', default=1e-2, type=float,
                        help='learning rate')
    parser.add_argument('--epochs', default=484, type=int,
                        help='Epochs for training model')
    parser.add_argument('--batch_size', default=12, type=int,
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
    parser.add_argument('--img_height', default=1024, type=int,
                        help='height size of image')
    parser.add_argument('--img_width', default=1024, type=int,
                        help='width size of image')
    return parser

def main(args):
    path = args.data_dir

    width = args.img_width
    height = args.img_height
    batch_size = args.batch_size

    train_loader = DataLoader(
        CityscapesDataset(path=path, subset='train', cropsize=(width,height)),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    valid_loader = DataLoader(
        CityscapesDataset(path=path, subset='valid', cropsize=(width,height)),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    model = get_model(
        model_name=args.model_name, 
        num_classes=args.num_classes,
        inference_phase=False,
    )

    summary(model)

    model = TrainModel(
        model=model,
        lr=args.lr,
        epochs=args.epochs,
        weight_decay=args.weight_decay,
        num_classes=args.num_classes,
        lr_scheduling=args.lr_scheduling,
        check_point=args.check_point,
        early_stop=args.early_stop,
    )

    history = model.fit(train_loader, valid_loader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('PIDNet training', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)