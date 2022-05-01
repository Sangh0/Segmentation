import time
from tqdm.notebook import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from bisenet.model import BiSeNet

def train(model,
          train_data,
          validation_data,
          epochs,
          learning_rate_scheduler=False,
          check_point=False,
          early_stop=False,
          last_epoch_save_path='./model/last_checkpoint.pt'):

    loss_list, miou_list = [], []
    val_loss_list, val_miou_list = [], []
    
    starting = time.time()
    
    for epoch in tqdm(range(epochs)):
        init_time = time.time()
        batch_loss, batch_miou = 0, 0
        for batch, (train_images, train_labels) in enumerate(train_data):
            model.train()

            train_images = train_images.to(device)
            train_labels = train_labels.to(device)
            
            optimizer.zero_grad()
            
            train_outputs = model(train_images)
            loss = loss_func(train_outputs, train_labels)
            miou = metric.mean_iou(train_outputs, train_labels)
            
            batch_loss += loss.item()
            batch_miou += miou.item()

            loss.backward()
            optimizer.step()

            del train_images; del train_labels; del train_outputs
            torch.cuda.empty_cache()

        loss_list.append(batch_loss/(batch+1))
        miou_list.append(batch_miou/(batch+1))

        model.eval()
        with torch.no_grad():
            vbatch_loss, vbatch_miou = 0, 0
            for vbatch, (val_images, val_labels) in enumerate(validation_data):
                val_images, val_labels = val_images.to(device), val_labels.to(device)

                val_outputs = model(val_images)
                val_loss = loss_func(val_outputs, val_labels)
                val_miou = metric.mean_iou(val_outputs, val_labels)

                vbatch_loss += val_loss.item()
                vbatch_miou += val_miou.item()

                del val_images; del val_labels; del val_outputs
                torch.cuda.empty_cache()

            val_loss_list.append(vbatch_loss/(vbatch+1))
            val_miou_list.append(vbatch_miou/(vbatch+1))

        end_time = time.time()

        print(f'\n[Epoch {epoch+1}/{epochs}]'
              f'  [time: {end_time-init_time:.3f}s]'
              f'  [lr = {optimizer.param_groups[0]["lr"]}]')
        print(f'[train loss: {batch_loss/(batch+1):.3f}]'
              f'  [train mIoU: {batch_miou/(batch+1):.3f}]'
              f'  [valid loss: {vbatch_loss/(vbatch+1):.3f}]'
              f'  [valid mIoU: {vbatch_miou/(vbatch+1):.3f}]')

        if learning_rate_scheduler:
            lr_scheduler.step(epoch+1)

        if check_point:
            checkpoint(vbatch_loss/(vbatch+1), model)

        if early_stop:
            assert check_point==False, 'Choose between Early Stopping and Check Point'
            early_stopping(vbatch_loss/(vbatch+1), model)
            if early_stopping.early_stop:
                print('\n##########################\n'
                      '##### Early Stopping #####\n'
                      '##########################')
                break

    if early_stop==False and check_point==False:
        torch.save(model.state_dict(), last_epoch_save_path)
        print('Saving model of last epoch')
        
    ending = time.time()
    print(f'\nTotal time for training is {ending-starting:.3f}s')
    
    return model, loss_list, miou_list, val_loss_list, val_miou_list


if __name__ == "__main__":
    batch_size = 16
    lr = 2.5e-2
    num_classes = 19
    EPOCH = 500
    device = torch.device('cuda')

    # load data
    from cityscapes import CityscapesDataset

    path = 'C:/Users/user/MY_DL/segmentation/dataset/camvid_low/'
    class_path = 'C:/Users/user/MY_DL/segmentation/dataset/camvid_low/11_class_dict.csv'
    transforms_ = [
        transforms.ToTensor(),
    ]
    
    train_loader = DataLoader(
        CityscapesDataset(path=path,
                          class_path=class_path,
                          transforms_=transforms_, 
                          subset='train'),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    
    valid_loader = DataLoader(
        CityscapesDataset(path=path,
                      class_path=class_path,
                      transforms_=transforms_, 
                      subset='valid'),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    # load callbacks and metrics
    from torchcallback import EarlyStopping, CheckPoint
    from torchscheduler import PolynomialLRDecay
    from torchmetrics import Metrics
    from model import BiSeNet

    model = BiSeNet().to(device)
    loss_func = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.005)
    
    es_save_path = './model/es_checkpoint.pt'
    cp_save_path = './model/cp_checkpoint.pt'

    checkpoint = CheckPoint(verbose=True, path=cp_save_path)
    early_stopping = EarlyStopping(patience=20, verbose=True, path=es_save_path)
    lr_scheduler = PolynomialLRDecay(optimizer, max_decay_steps=EPOCH)

    metric = Metrics(n_classes=num_classes, dim=1)

    model, train_loss, train_miou, train_bf, valid_loss, valid_miou, valid_bf = train(
        model, 
        train_data=train_loader,
        validation_data=valid_loader,
        epochs=EPOCH,
        learning_rate_scheduler=False,
        check_point=True,
        early_stop=False,
    )