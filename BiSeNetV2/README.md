# BiSeNetV2 Implementation 
### BiSeNetV2 link: https://arxiv.org/abs/2004.02147  
### [Paper Review](https://github.com/Sangh0/Segmentation/blob/main/BiSeNetV2/BiSeNetV2_paper_review.ipynb) 
### BiSeNet v2 Architecture  
<img src = "https://github.com/Sangh0/Segmentation/blob/main/BiSeNetV2/figure/figure3.JPG?raw=true" width=700>

## Results
 
<img src = "https://github.com/Sangh0/Segmentation/blob/main/BiSeNetV2/images/output2.png?raw=true" width=800>
<img src = "https://github.com/Sangh0/Segmentation/blob/main/BiSeNetV2/images/output3.png?raw=true" width=800>


## Training
```
usage: main.py [-h] [--data_dir DATA_DIR] [--lr LR] [--epochs EPOCHS] [--batch_size BATCH_SIZE]
               [--weight_decay WEIGHT_DECAY] [--num_classes NUM_CLASSES] [--lr_scheduling LR_SCHEDULING]
               [--check_point CHECK_POINT] [--early_stop EARLY_STOP] [--img_height IMG_HEIGHT] 
               [--img_width IMG_WIDTH]
```