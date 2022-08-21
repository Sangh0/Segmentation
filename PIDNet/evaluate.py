import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F

from util.metrics import Metrics
from models.pidnet import get_model
from datasets.cityscapes import load_cityscapes_dataset
from util.transform import UnNormalize


@torch.no_grad
def evaluate(
    model,
    dataset, 
    device, 
    metric,
):

    """
    Args:
        - model: trained model or weighted model
        - dataset: train or valid or test
        - device: cuda or cpu
        - metric: a metric to check performance of model about dataset
    """
    start = time.time()
    model = model.to(device)
    model.eval()
    image_list, label_list, output_list = [], [], []
    batch_miou = 0
    pbar = tqdm(dataset, total=len(dataset))
    for batch, dset in enumerate(pbar):
        if len(dset) == 3:
            images, labels = dset[0], dset[1]
            images, labels = images.to(device), labels.to(device)
            
        elif len(dset) == 1:
            images, labels = dset[0], None
            images = images.to(device)

        else:
            raise ValueError('Dataset not found')
            
        if batch == 100:
            break

        _, outputs, _ = model(images)
        
        if images.size(2) != outputs.size(2) or images.size(3) != outputs.size(3):
            outputs = F.interpolate(
                outputs,
                size=(images.size(2), images.size(3)),
                mode='bilinear',
                align_corners=False,
            )
        
        if len(dset) == 3:
            mean_iou = metric(outputs, labels)
            batch_miou += mean_iou.item()
        
        image_list.append(images.detach().cpu())
        label_list.append(labels.detach().cpu() if labels is not None else None)
        output_list.append(outputs.detach().cpu())
        
        del images; del labels; del outputs
        torch.cuda.empty_cache()
        
    end = time.time()
    
    print(f'Inference Time: {end-start:.3f}s')

    if len(dset) == 3:
        print(f'Meawn IOU: {batch_miou/(batch+1)}')
        return {
            'images': torch.cat(image_list, dim=0),
            'labels': torch.cat(label_list, dim=0),
            'outputs': torch.cat(output_list, dim=0),
        }
    
    else:
        return {
            'images': torch.cat(image_list, dim=0),
            'outputs': torch.cat(output_list, dim=0),
        }


def get_args_parser():
    parser = argparse.ArgumentParser(description='Training PIDNet', add_help=False)
    parser.add_argument('--data_dir', type=str, required=True,
                        help='directory where your dataset is located')
    parser.add_argument('--weight', type=str, required=True,
                        help='load weight file of trained model')
    parser.add_argument('--dataset', type=str, required=True,
                        help='select one dataset between train, valid and test')
    parser.add_argument('--model_name', type=str, required=True,
                        help='Name of PIDNet model, select between pidnet_s, pidnet_m and pidnet_l')
    parser.add_argument('--num_classes', type=int, required=True,
                        help='class number of dataset')
    parser.add_argument('--device', type=str, 
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='set device for faster model training')
    return parser

def main(args):
    metric = Metrics(n_classes=args.num_classes, dim=1)

    device = args.device
    
    assert args.dataset in ('train', 'valid', 'test'), \
        'you should select one between train, valid and test set'
    
    if args.dataset=='train':
        dataset = load_cityscapes_dataset(
            path=args.data_dir,
            get_val_set=False,
            batch_size=1,
        )
        
        data_loader = dataset['train_set']

    elif args.dataset=='valid':
        dataset = load_cityscapes_dataset(
            path=args.data_dir,
            get_val_set=True,
            batch_size=1,
        )

        data_loader = dataset['valid_set']
        
    else:
        dataset = load_cityscapes_dataset(
            path=args.data_dir,
            get_test_set=True,
            batch_size=1,
        )
    
        data_loader = dataset['test_set']
    
    pidnet = get_model(
        model_name=args.model_name,
        num_classes=args.num_classes,
        inference_phase=True,
    )
    
    pidnet.load_state_dict(torch.load(args.weight_path))

    evaluate(
        model=pidnet,
        dataset=data_loader,
        device=args.device,
        metric=metric.mean_iou,
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser('PIDNet testing', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)