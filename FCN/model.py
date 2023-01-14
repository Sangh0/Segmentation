from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16_bn, VGG16_BN_Weights


class Encoder(nn.Module):
    
    def __init__(self):
        super(Encoder, self).__init__()
        vgg = vgg16_bn(weights=VGG16_BN_Weights)
        
        self.block1 = self._make_block_module_(vgg, [0, 7])
        self.block2 = self._make_block_module_(vgg, [7, 14])
        self.block3 = self._make_block_module_(vgg, [14, 24])
        self.block4 = self._make_block_module_(vgg, [24, 34])
        self.block5 = self._make_block_module_(vgg, [34, 44])
        self.block6 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

    def _make_block_module_(self, vgg: nn.Module, indices: List[int, int]):
        layers = []
        for i in range(indices[0], indices[1]):
            layers.append(self.vgg.features[i])

        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        x5 = self.block5(x4)
        x6 = self.block6(x5)
        return x3, x4, x6


class FCN32s(nn.Module):

    def __init__(self, num_classes: int):
        super(FCN8s, self).__init__()
        self.encoder = Encoder()

        self.upsample32x = nn.Upsample(scale_factor=32, mode='bilinear')
        self.block = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        _, _, conv7 = self.encoder(x)
        x = self.upsample32x(conv7)
        x = self.block(x)
        return x


class FCN16s(nn.Module):

    def __init__(self, num_classes: int):
        super(FCN16s, self).__init__()
        self.encoder = Encoder()

        self.upsample2x = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample16x = nn.Upsample(scale_factor=16, mode='bilinear')

        self.block = nn.Sequential(
            nn.Conv2d(512+256, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        _, pool4, conv7 = self.encoder(x)
        conv7x2 = self.upsample2x(conv7)
        out = torch.cat([conv7x2, pool4], dim=1)
        out = self.block(out)
        out = self.upsample16x(out)
        return out


class FCN8s(nn.Module):

    def __init__(self, num_classes: int):
        super(FCN8s, self).__init__()
        self.encoder = Encoder()

        self.upsample2x = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample4x = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsample8x = nn.Upsample(scale_factor=8, mode='bilinear')

        self.block = nn.Sequential(
            nn.Conv2d(256+512+512, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        pool3, pool4, conv7 = self.encoder(x)
        conv7x4 = self.upsample4x(conv7)
        pool4x2 = self.upsample2x(pool4)
        out = torch.cat([conv7x4, pool4x2, pool3], dim=1)
        out = self.block(out)
        out = self.upsample8x(out)
        return out


def get_fcn(num_classes: int, model_type: str='fcn_32s'):
    assert model_type in ('fcn_32s', 'fcn_16s', 'fcn_8s'), \
        f'The type {model_type} does not exists'

    if model_type == 'fcn_32s':
        model = FCN32s(num_classes)

    elif model_type == 'fcn_16s':
        model = FCN16s(num_classes)

    else:
        model = FCN8s(num_classes)

    return model