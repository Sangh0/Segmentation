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


@torch.no_grad()
def evaluate(
    model, 
    weight, 
    dataset, 
    device, 
    cal_miou
):
    start = time.time()
    model.eval()
    image_list, label_list, output_list = [], [], []
    batch_miou = 0
    for batch, (images, labels, _) in tqdm(dataset):
        images, labels = images.to(device), labels.to(device)
        
        _, outputs, _ = model(images)
        
        if labels.size(2) != outputs.size(2) or labels.size(3) != outputs.size(3):
            outputs = F.interpolate(
                outputs,
                size=(labels.size(2), labels.size(3)),
                mode='bilinear',
                align_corners=False,
            )
            
        mean_iou = cal_miou(outputs, labels)
        batch_miou += mean_iou.item()
        
        image_list.append(images)
        label_list.append(labels)
        output_list.append(outputs)
        
        del images; del labels; del outputs
        torch.cuda.empty_cache()
        
    end = time.time()
    
    print(f'Inference Time: {end-start:.3f}s')
    print(f'Mean IOU : {mean_iou/(batch+1)}')
    
    return {
        'images': torch.cat(image_list, dim=0),
        'labels': torch.cat(label_list, dim=0),
        'outputs': torch.cat(output_list, dim=0),
    }


def segmap2image(segmaps):
    labels_info = {
        0: [128,64,128],    # road
        1: [244, 35, 232],  # sidewalk
        2: [70, 70, 70],    # building
        3: [102, 102, 156], # wall
        4: [190, 153, 153], # fence
        5: [153, 153, 153], # pole
        6: [250, 170, 30],  # traffic light
        7: [220, 220, 0],   # traffic sign
        8: [107, 142, 35],  # vegetation
        9: [152, 251, 152], # terrain
        10: [70, 130, 180], # sky
        11: [220, 20, 60],  # person
        12: [255, 0, 0],    # rider
        13: [0, 0, 142],    # car
        14: [0, 0, 70],     # truck
        15: [0, 60, 100],   # bus
        16: [0, 80, 100],   # train
        17: [0, 0, 230],    # motorcycle
        18: [119, 11, 32]   # bicycle
    }

    B, C, H, W = segmaps.size()
    map2image = np.zeros(shape=(B, H, W, 3), dtyupe=np.int32)
    for i in tqdm(range(B)):
        for j in range(H):
            for k in range(W):
                map2image[i,:,j,k] = labels_info[segmaps[i,j,k]]

    del segmaps
    return map2image


def run_on_notebook(images, labels, outputs, num):
    for i in range(num):
        plt.figure(figsize=(25, 8))
        plt.subplot(131)
        plt.imshow(images[i])
        plt.title('Original Image')
        plt.axis('off')
        plt.subplot(132)
        plt.imshow(labels[i])
        plt.title('Label Image')
        plt.axis('off')
        plt.subplot(133)
        plt.imshow(images[i])
        plt.imshow(outputs[i], alpha=0.5)
        plt.title('Predicted Image')
        plt.axis('off')
        plt.show()


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
    cal_miou = Metrics(n_classes=args.num_classes, dim=1)

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
        weight=args.weight,
        dataset=data_loader,
        device=args.device,
        cal_miou=cal_miou,
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser('PIDNet testing', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)