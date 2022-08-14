# PIDNET Implementation 
### PIDNET link : https://arxiv.org/abs/2206.02066  
### [Paper Review](https://github.com/Sangh0/Segmentation/blob/main/PIDNet/pidnet_paper_review.ipynb)  
### Code Implementation Reference: [Official Github](https://github.com/XuJiacong/PIDNet)
### PIDNet Architecture  
<img src = "https://github.com/Sangh0/Segmentation/blob/main/PIDNet/figure/figure4.JPG?raw=true" width=600>

## Training
```
usage: main.py [-h] [--data_dir DATA_DIR] [--model_name MODEL_NAME] [--lr LR] [--epochs EPOCHS]
               [--batch_size BATCH_SIZE] [--weight_decay WEIGHT_DECAY] [--num_classes NUM_CLASSES]
               [--loss_weights LOSS_WEIGHTS] [--t_threshold T_THRESHOLD] [--lr_scheduling LR_SCHEDULING] 
               [--check_point CHECK_POINT] [--early_stop EARLY_STOP] [--img_height IMG_HEIGHT] 
               [--img_width IMG_WIDTH]
```

## Run On Jupyter Notebook
```python
# Load Packages
from torchsummary import summary

from models.pidnet import get_model
from train import TrainModel
from datasets.cityscapes import load_cityscapes_dataset

# Set hyperparameters
Config = {
    'lr': 1e-2,
    'weight_decay': 5e-4,
    'batch_size': 12,
    'width': 1024,
    'height': 1024,
    'epochs': 484,
    'num_classes': 19,
    't_threshold': 0.8,
    'loss_weights': [0.4, 20, 1, 1],
    'lr_scheduling': True,
    'check_point': True,
    'early_stop': False,
    'model_name': 'pidnet_s',
}

# Load Datasets
path = './cityscapes'

dataset = load_cityscapes_dataset(
    path=path,
    height=Config['height'],
    width=Config['width'],
    get_val_set=True,
    batch_size=Config['batch_size'],
)

train_loader, valid_loader = dataset['train_set'], dataset['valid_set']

# Load PIDNet
pidnet = get_model(
    model_name=Config['model_name'], 
    num_classes=Config['num_classes'],
    inference_phase=False,
)

# Check summary of model
summary(pidnet, (3, Config['height'], Config['width']), device='cpu')

# Training model
model = TrainModel(
    model=pidnet,
    lr=Config['lr'],
    epochs=Config['epochs'],
    weight_decay=Config['weight_decay'],
    num_classes=Config['num_classes'],
    t_threshold=Config['t_threshold'],
    loss_weights=Config['loss_weights'],
    lr_scheduling=Config['lr_scheduling'],
    check_point=Config['check_point'],
    early_stop=Config['early_stop'],
)

history = model.fit(train_loader, valid_loader)
```