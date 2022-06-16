import time
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn


def test_step(model,
              test_data,
              device,
              loss_func=None,
              miou_func=None,
              subset='test'):
    start = time.time()
    model.eval()
    test_images_list, test_labels_list, test_outputs_list = [], [], []
    with torch.no_grad():
        if subset=='test':
            for tbatch, test_images in tqdm(enumerate(test_data), total=len(test_data)):
                test_images = test_images.to(device)
                
                test_outputs, _, _, _, _ = model(test_images)
                
                test_images_list.append(test_images.detach().cpu().numpy())
                test_outputs_list.append(test_outputs.detach().cpu().numpy())
                
                del test_images; del _; del test_outputs
                torch.cuda.empty_cache()
        
        else:
            batch_loss, batch_miou = 0, 0
            for tbatch, (images, labels) in tqdm(enumerate(test_data), total=len(test_data)):
                images, labels = images.to(device), labels.squeeze().to(device)
                
                outputs, _, _, _, _ = model(images)
                
                loss = loss_func(outputs, labels)
                batch_loss += loss.item()
                miou = miou_func(outputs, labels)
                batch_miou += miou.item()
                
                test_images_list.append(images.detach().cpu().numpy())
                test_outputs_list.append(outputs.detach().cpu().numpy())
                test_labels_list.append(labels.detach().cpu().numpy())
                
                del images; del _; del outputs; del labels
                torch.cuda.empty_cache()
            
        end = time.time()
    
    print(f'Inference Time: {end-start:.3f}s')
    if subset is not 'test':
        print(f'Loss: {batch_loss/(tbatch+1):.3f}\t'
              f'Mean IoU: {batch_miou/(tbatch+1):.3f}')
    
    test_images_list = np.concatenate(test_images_list, axis=0)
    test_outputs_list = np.concatenate(test_outputs_list, axis=0)
    test_labels_list = np.concatenate(test_labels_list, axis=0)
    
    if subset=='test':
        return {
            'test image': test_images_list,
            'test output': test_outputs_list
        }
    else:
        return {
            'image': test_images_list,
            'label': test_labels_list,
            'output': test_outputs_list
        }