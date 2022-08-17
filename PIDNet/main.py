import os
import ast
import argparse

import torch
from torchsummary import summary

from models.pidnet import get_model
from train import TrainModel
from datasets.cityscapes import load_cityscapes_dataset

def arg_as_list(param):
    arg = ast.literal_eval(param)
    if type(arg) is not list:
        raise argparse.ArgumentTypeError('Argument \ "%s\" is not a list'%(arg))
    return arg

def get_args_parser():
    parser = argparse.ArgumentParser(description='Training PIDNet', add_help=False)
    parser.add_argument('--data_dir', type=str, required=True,
                        help='directory where your dataset is located')
    parser.add_argument('--model_name', type=str, required=True,
                        help='Name of PIDNet model, select between pidnet_s, pidnet_m and pidnet_l')
    parser.add_argument('--lr', default=1e-2, type=float,
                        help='learning rate')
    parser.add_argument('--epochs', default=484, type=int,
                        help='Epochs for training model')
    parser.add_argument('--batch_size', default=12, type=int,
                        help='Batch Size for training model')
    parser.add_argument('--weight_decay', default=5e-4, type=float,
                        help='weight decay of optimizer SGD')
    parser.add_argument('--num_classes', type=int, required=True,
                        help='class number of dataset')
    parser.add_argument('--loss_weights', default=[0.4, 20, 1, 1], type=arg_as_list,
                        help='weight of each loss functions, length: 4')
    parser.add_argument('--t_threshold', default=0.8, type=float,
                        help='threshold value for l3 function')
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
    parser.add_argument('--device', type=str, 
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='set device for faster model training')
    return parser

def main(args):

    dataset = load_cityscapes_dataset(
        path=args.data_dir,
        height=args.img_height,
        width=args.img_width,
        get_val_set=True,
        batch_size=args.batch_size,
    )

    train_loader, valid_loader = dataset['train_set'], dataset['valid_set']

    pidnet = get_model(
        model_name=args.model_name, 
        num_classes=args.num_classes,
        inference_phase=False,
    )

    summary(pidnet, (3, args.img_height, args.img_width), device='cpu')

    model = TrainModel(
        model=pidnet,
        lr=args.lr,
        epochs=args.epochs,
        weight_decay=args.weight_decay,
        num_classes=args.num_classes,
        t_threshold=args.t_threshold,
        loss_weights=args.loss_weights,
        lr_scheduling=args.lr_scheduling,
        check_point=args.check_point,
        early_stop=args.early_stop,
    )

    history = model.fit(train_loader, valid_loader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('PIDNet training', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)