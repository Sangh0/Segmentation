import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_utils import (
    BasicBlock, 
    Bottleneck, 
    SegmentHead, 
    DAPPM, 
    PAPPM, 
    Pag, 
    Bag, 
    LightBag,
)

class PIDNet(nn.Module):
    
    def __init__(
        self,
        num_classes,
        m=2,
        n=3,
        in_dim=3,
        num_filters=64,
        ppm_dim=96,
        head_dim=128,
        cal_auxiliary=True,
    ):
        super(PIDNet, self).__init__()
        self.cal_auxiliary = cal_auxiliary
        
        # I Branch
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim, num_filters, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True),
        )
        self.relu = nn.ReLU(inplace=True)
        self.layer1_common = self._make_layer(BasicBlock, num_filters, num_filters, blocks=m, stride=1)
        self.layer2_common = self._make_layer(BasicBlock, num_filters, num_filters*2, blocks=m, stride=2)
        self.layer3_i = self._make_layer(BasicBlock, num_filters*2, num_filters*4, blocks=n, stride=2)
        self.layer4_i = self._make_layer(BasicBlock, num_filters*4, num_filters*8, blocks=n, stride=2)
        self.layer5_i = self._make_layer(Bottleneck, num_filters*8, num_filters*8, blocks=2, stride=2)
        
        # P Branch
        self.compression3 = nn.Sequential(
            nn.Conv2d(num_filters*4, num_filters*2, 
                      kernel_size=1, stride=1, padding=0, 
                      bias=False),
            nn.BatchNorm2d(num_filters*2),
        )
        self.compression4 = nn.Sequential(
            nn.Conv2d(num_filters*8, num_filters*2,
                      kernel_size=1, stride=1, padding=0,
                      bias=False),
            nn.BatchNorm2d(num_filters*2),
        )
        self.pag3 = Pag(num_filters*2, num_filters)
        self.pag4 = Pag(num_filters*2, num_filters)
        
        self.layer3_p = self._make_layer(BasicBlock, num_filters*2, num_filters*2, blocks=m, stride=1)
        self.layer4_p = self._make_layer(BasicBlock, num_filters*2, num_filters*2, blocks=m, stride=1)
        self.layer5_p = self._make_layer(Bottleneck, num_filters*2, num_filters*2, blocks=1, stride=1)
        
        # D Branch
        if m==2:
            self.layer3_d = self._make_single_layer(BasicBlock, num_filters*2, num_filters)
            self.layer4_d = self._make_layer(Bottleneck, num_filters, num_filters, blocks=1)
            self.diff3 = nn.Sequential(
                nn.Conv2d(num_filters*4, num_filters, 
                          kernel_size=3, stride=1, padding=1, 
                          bias=False),
                nn.BatchNorm2d(num_filters),
            )
            self.diff4 = nn.Sequential(
                nn.Conv2d(num_filters*8, num_filters*2,
                          kernel_size=3, stride=1, padding=1,
                          bias=False),
                nn.BatchNorm2d(num_filters*2),
            )
            self.ppm = PAPPM(num_filters*16, ppm_dim, num_filters*4)
            self.bag = LightBag(num_filters*4, num_filters*4)
        
        else:
            self.layer3_d = self._make_single_layer(BasicBlock, num_filters*2, num_filters*2)
            self.layer4_d = self._make_single_layer(BasicBlock, num_filters*2, num_filters*2)
            self.diff3 = nn.Sequential(
                nn.Conv2d(num_filters*4, num_filters*2,
                          kernel_size=3, stride=1, padding=1,
                          bias=False),
                nn.BatchNorm2d(num_filters*2),
            )
            self.diff4 = nn.Sequential(
                nn.Conv2d(num_filters*8, num_filters*2,
                          kernel_size=3, stride=1, padding=1,
                          bias=False),
                nn.BatchNorm2d(num_filters*2),
            )
            self.ppm = DAPPM(num_filters*16, ppm_dim, num_filters*4)
            self.bag = Bag(num_filters*4, num_filters*4)
            
        self.layer5_d = self._make_layer(Bottleneck, num_filters*2, num_filters*2, 1)
        
        if self.cal_auxiliary:
            self.seghead_p = SegmentHead(num_filters*2, head_dim, num_classes)
            self.seghead_d = SegmentHead(num_filters*2, num_filters, 1)
            
        self.final_layer = SegmentHead(num_filters*4, head_dim, num_classes)
        
        self._init_weight_()
        
    def _make_layer(self, block, in_dim, mid_dim, blocks, stride=1):
        downsample = None
        if stride != 1 or in_dim != mid_dim * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_dim, mid_dim * block.expansion, 
                          kernel_size=1, stride=stride, padding=0, 
                          bias=False),
                nn.BatchNorm2d(mid_dim * block.expansion),
            )
            
        layers = []
        layers.append(block(in_dim, mid_dim, stride, downsample))
        in_dim = mid_dim * block.expansion
        for i in range(1, blocks):
            if i == (blocks-1):
                layers.append(block(in_dim, mid_dim, stride=1, apply_relu=False))
            else:
                layers.append(block(in_dim, mid_dim, stride=1, apply_relu=True))
                
        return nn.Sequential(*layers)
    
    def _make_single_layer(self, block, in_dim, mid_dim, stride=1):
        downsample = None
        if stride != 1 or in_dim != mid_dim * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_dim, mid_dim*block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(mid_dim*block.expansion),
            )
        
        layer = block(in_dim, mid_dim, stride, downsample, apply_relu=False)
        return layer
    
    def _init_weight_(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        width = x.shape[-1] // 8
        height = x.shape[-2] // 8
        
        # Stage 0
        x = self.conv1(x)
        
        # Stage 1
        x = self.layer1_common(x)
        
        # Stage 2
        x = self.relu(self.layer2_common(self.relu(x))) # 1/8
        
        # Stage 3
        x_p = self.layer3_p(x)
        x_d = self.layer3_d(x)
        x_i = self.relu(self.layer3_i(x)) # 1/16
        x_p = self.pag3(x_p, self.compression3(x_i))
        x_d = x_d + F.interpolate(
            self.diff3(x_i),
            size=(height, width),
            mode='bilinear',
            align_corners=False,
        )
        
        if self.cal_auxiliary:
            aux_p = x_p
            
        # Stage 4
        x_i = self.relu(self.layer4_i(x_i))
        x_p = self.layer4_p(self.relu(x_p))
        x_d = self.layer4_d(self.relu(x_d))
        x_p = self.pag4(x_p, self.compression4(x_i))
        x_d = x_d + F.interpolate(
            self.diff4(x_i),
            size=(height, width),
            mode='bilinear',
            align_corners=False,
        )
        
        if self.cal_auxiliary:
            aux_d = x_d
            
        # Stage 5
        x_p = self.layer5_p(self.relu(x_p))
        x_d = self.layer5_d(self.relu(x_d))
        
        # Stage 6
        x_i = F.interpolate(
            self.ppm(self.layer5_i(x_i)),
            size=(height, width),
            mode='bilinear',
            align_corners=False,
        )
        
        out = self.final_layer(self.bag(x_p, x_i, x_d))
        
        if self.cal_auxiliary:
            extra_p = self.seghead_p(aux_p)
            extra_d = self.seghead_d(aux_d)
            return extra_p, out, extra_d
        else:
            return out
        

def get_model(model_name: str, num_classes: int, inference_phase: bool=False):
    assert model_name in ('pidnet_s', 'pidnet_m', 'pidnet_l'), \
        f'{model_name} does not exist, you have to select between pidnet_s, pidnet_m and pidnet_l'
    
    cal_auxiliary = False if inference_phase else True

    if model_name=='pidnet_s':
        model = PIDNet(num_classes, m=2, n=3, num_filters=32, 
                       ppm_dim=96, head_dim=128, cal_auxiliary=cal_auxiliary)
        
    elif model_name=='pidnet_m':
        model = PIDNet(num_classes, m=2, n=3, num_filters=64, 
                       ppm_dim=96, head_dim=128, cal_auxiliary=cal_auxiliary)
    
    else:
        model = PIDNet(num_classes, m=3, n=4, num_filters=64, 
                       ppm_dim=128, head_dim=256, cal_auxiliary=cal_auxiliary)
                       
    return model