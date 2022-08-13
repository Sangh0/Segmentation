import torch
import torch.nn as nn
import torch.nn.functional as F

class Combination(nn.Module):
    
    def __init__(
        self,
        model,
        sem_loss,
        bd_loss,
        metrics,
        ignore_index=255, 
    ):
        super(Combination, self).__init__()
        self.model = model
        self.sem_loss = sem_loss
        self.bd_loss = bd_loss
        self.ignore_index = ignore_index
        self.metrics = metrics
        
    def cal_pixel_acc(self, pred, label):
        return self.metrics.pixel_acc(pred, label)
    
    def cal_mean_iou(self, pred, label):
        return self.metrics.mean_iou(pred, label)
        
    def forward(self, inputs, labels, edges):
        outputs = self.model(inputs)
        
        label_height, label_width = labels.size(2), labels.size(3)
        output_height, output_width = outputs[0].size(2), outputs[0].size(3)
        
        if label_height!=output_height or label_width!=output_width:
            for i in range(len(outputs)):
                outputs[i] = F.interpolate(
                    outputs[i],
                    size=(label_height, label_width),
                    mode='bilinear',
                    align_corners=False,
                )

        pix_acc = self.cal_pixel_acc(outputs[1], labels)
        mean_iou = self.cal_mean_iou(outputs[1], labels)
        semantic_loss = self.sem_loss(outputs[:2], labels.squeeze(dim=1))
        boundary_loss = self.bd_loss(outputs[2], edges.float())
        
        filler = torch.ones_like(labels.squeeze(dim=1)) * self.ignore_index
        bd_labels = torch.where(torch.sigmoid(outputs[2][:,0,:,:])>0.8, labels.squeeze(dim=1), filler)
        sb_loss = self.sem_loss(outputs[1], bd_labels)
        total_loss = semantic_loss + boundary_loss + sb_loss
        
        return {
            'total_loss': torch.unsqueeze(total_loss, dim=0),
            'boundary_loss': boundary_loss,
            'semantic_loss': semantic_loss,
            'pixel_accuracy': pix_acc,
            'mean_iou': mean_iou,
        }