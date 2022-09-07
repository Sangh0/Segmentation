# PIDNET Implementation 
### Paper Link : https://arxiv.org/abs/2206.02066  
### [Paper Review](https://github.com/Sangh0/Segmentation/blob/main/PIDNet/pidnet_paper_review.ipynb)  
### Code Implementation Reference: [Official Github](https://github.com/XuJiacong/PIDNet)
### PIDNet Architecture  
<img src = "https://github.com/Sangh0/Segmentation/blob/main/PIDNet/figure/figure4.JPG?raw=true" width=600>

## Output of PIDNet in test set
- first sample
<img src = "https://github.com/Sangh0/Segmentation/blob/main/PIDNet/figure/output/test_output1.png?raw=true" width=600>

- second sample
<img src = "https://github.com/Sangh0/Segmentation/blob/main/PIDNet/figure/output/test_output2.png?raw=true" width=600>

## download dataset  
[Cityscapes Dataset](https://www.cityscapes-dataset.com/)     
[CamVid Dataset](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/)    
[ADE20K Dataset](https://groups.csail.mit.edu/vision/datasets/ADE20K/)  

## Train
```
usage: main.py [-h] [--data_dir DATA_DIR] [--model_name MODEL_NAME] [--lr LR] [--epochs EPOCHS] \
               [--batch_size BATCH_SIZE] [--weight_decay WEIGHT_DECAY] [--num_classes NUM_CLASSES] \
               [--loss_weights LOSS_WEIGHTS] [--t_threshold T_THRESHOLD] [--lr_scheduling LR_SCHEDULING] \ 
               [--check_point CHECK_POINT] [--early_stop EARLY_STOP] [--img_height IMG_HEIGHT] \
               [--img_width IMG_WIDTH]

example: python main.py --data_dir ./dataset/cityscapes --model_name pidnet_s --num_classes 19
```

## Evaluate
```
usage: evaluate.py [-h] [--data_dir DATA_DIR] [--weight WEIGHT] [--dataset DATASET] \
                   [--model_name MODEL_NAME] [--num_classes NUM_CLASSES]
    
example: python evaluate.py --data_dir ./dataset/cityscapes --weight ./weights/best_weight.pt \
                            --dataset test --model_name pidnet_s --num_classes 19
```

## Run on Jupyter Notebook for training model
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

## Run on Jupyter Notebook to evaluate model with each dataset
```python
import torch

from util.metrics import Metrics
from models.pidnet import get_model
from datasets.cityscapes import load_cityscapes_dataset
from evaluate import evaluate


# Set parameters
Config = {
    'data_dir': './cityscapes',
    'weight': './pidnet/weights/best.pt',
    'dataset': 'select dataset between train, valid and test',
    'model_name': 'pidnet_s',
    'num_classes': 19,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
}

metric = Metrics(n_classes=Config['num_classes'], dim=1)

# Load Datasets
if Config['dataset']=='train':
    dataset = load_cityscapes_dataset(
        path=Config['data_dir'],
        get_val_set=False,
        batch_size=1,
    )
    data_loader = dataset['train_set']

elif Config['dataset']=='valid':
    dataset = load_cityscapes_dataset(
        path=Config['data_dir'],
        get_val_set=True,
        batch_size=1,
    )
    data_loader = dataset['valid_set']
    
else:
    dataset = load_cityscapes_dataset(
        path=Config['data_dir'],
        get_test_set=True,
        batch_size=1,
    )
    data_loader = dataset['test_set']

# Load PIDNet
pidnet = get_model(
    model_name=Config['model_name'], 
    num_classes=Config['num_classes'],
    inference_phase=False,
)
pidnet.load_state_dict(torch.load(Config['weight']))

# Evaluate PIDNet
evaluate(
    model=pidnet, 
    dataset=data_loader,
    device=Config['device'],
    metric=metric.mean_iou,
)
```

## Visualize outputs on Jupyter Notebook
```python
from visualize import VisualizeOnNotebook

# if using train or valid set
visual = VisualizeCityscapesOnNotebook(
    images=images,
    outputs=outputs,
    labels=labels,
)

visual.visualize(number, 'name of dataset')

# if using test set
visual = VisualizeCityscapesOnNotebook(
    images=images,
    outputs=outputs,
    labels=None,
)

visual.visualize(number, 'test')
```