import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(
        self,
        in_dim,
        out_dim,
        stride=1,
        downsample=None,
        apply_relu=False,
    ):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_dim)
        self.downsample = downsample
        self.apply_relu = apply_relu
        
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
            
        out += residual
        
        if self.apply_relu:
            return self.relu(out)
        
        else:
            return out


class Bottleneck(nn.Module):
    expansion = 2
    def __init__(
        self,
        in_dim,
        out_dim,
        stride=1,
        downsample=None,
        apply_relu=False,
    ):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=1, 
                               stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_dim)
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_dim)
        self.conv3 = nn.Conv2d(out_dim, out_dim * self.expansion, kernel_size=1, 
                               stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_dim * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.apply_relu = apply_relu
        
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
            
        out += residual
        
        if self.apply_relu:
            return self.relu(out)
        else:
            return out


class SegmentHead(nn.Module):
    
    def __init__(
        self,
        in_dim,
        mid_dim,
        out_dim,
        scale_factor=None,
    ):
        super(SegmentHead, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_dim)
        self.conv1 = nn.Conv2d(in_dim, mid_dim, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_dim, out_dim, kernel_size=1, 
                               stride=1, padding=0, bias=True)
        self.scale_factor = scale_factor
        
    def forward(self, x):
        x = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(x)))
        
        if self.scale_factor is not None:
            height = x.shape[-2] * self.scale_factor
            width = x.shape[-1] * self.scale_factor
            out = F.interpolate(
                out, 
                size=[height, width], 
                mode='bilinear', 
                align_corners=False,
            )
            
        return out

    
class DAPPM(nn.Module):
    
    def __init__(
        self,
        in_dim,
        mid_dim,
        out_dim,
    ):
        super(DAPPM, self).__init__()
        self.scale0 = nn.Sequential(
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim, mid_dim, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.scale1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim, mid_dim, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.scale2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim, mid_dim, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.scale3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim, mid_dim, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.scale4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1,1)),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim, mid_dim, kernel_size=1, stride=1, padding=0, bias=False),
        )
        
        self.process1 = nn.Sequential(
            nn.BatchNorm2d(mid_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_dim, mid_dim, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.process2 = nn.Sequential(
            nn.BatchNorm2d(mid_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_dim, mid_dim, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.process3 = nn.Sequential(
            nn.BatchNorm2d(mid_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_dim, mid_dim, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.process4 = nn.Sequential(
            nn.BatchNorm2d(mid_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_dim, mid_dim, kernel_size=3, stride=1, padding=1, bias=False),
        )
        
        self.compression = nn.Sequential(
            nn.BatchNorm2d(mid_dim*5),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_dim*5, out_dim, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.shortcut = nn.Sequential(
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=False),
        )
        
    def forward(self, x):
        width = x.shape[-1]
        height = x.shape[-2]
        x_list = []
        
        x_list.append(self.scale0(x))
        x_list.append(self.process1(F.interpolate(
            self.scale1(x),
            size=(height, width), 
            mode='bilinear',
            align_corners=False) + x_list[-1])
        )
        x_list.append(self.process2(F.interpolate(
            self.scale2(x),
            size=(height, width),
            mode='bilinear',
            align_corners=False) + x_list[-1])
        )
        x_list.append(self.process3(F.interpolate(
            self.scale3(x),
            size=(height, width),
            mode='bilinear',
            align_corners=False) + x_list[-1])
        )
        x_list.append(self.process4(F.interpolate(
            self.scale4(x),
            size=(height, width),
            mode='bilinear',
            align_corners=False) + x_list[-1])
        )
        
        out = self.compression(torch.cat(x_list, dim=1)) + self.shortcut(x)
        return out
        
        
class PAPPM(nn.Module):
    
    def __init__(
        self,
        in_dim,
        mid_dim,
        out_dim,
    ):
        super(PAPPM, self).__init__()
        self.scale0 = nn.Sequential(
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim, mid_dim, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.scale1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim, mid_dim, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.scale2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim, mid_dim, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.scale3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim, mid_dim, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.scale4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1,1)),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim, mid_dim, kernel_size=1, stride=1, padding=0, bias=False),
        )
        
        self.scale_process = nn.Sequential(
            nn.BatchNorm2d(mid_dim*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_dim*4, mid_dim*4, kernel_size=3, stride=1, padding=1, groups=4, bias=False),
        )
        self.compression = nn.Sequential(
            nn.BatchNorm2d(mid_dim*5),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_dim*5, out_dim, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.shortcut = nn.Sequential(
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=False),
        )
        
    def forward(self, x):
        width = x.shape[-1]
        height = x.shape[-2]
        scale_list = []
        
        x_ = self.scale0(x)
        scale_list.append(F.interpolate(
            self.scale1(x),
            size=(height, width),
            mode='bilinear',
            align_corners=False) + x_)
        scale_list.append(F.interpolate(
            self.scale2(x),
            size=(height, width),
            mode='bilinear',
            align_corners=False) + x_)
        scale_list.append(F.interpolate(
            self.scale3(x),
            size=(height, width),
            mode='bilinear',
            align_corners=False) + x_)
        scale_list.append(F.interpolate(
            self.scale4(x),
            size=(height, width),
            mode='bilinear',
            align_corners=False) + x_)
        
        scale_out = self.scale_process(torch.cat(scale_list, dim=1))
        out = self.compression(torch.cat([x_, scale_out], dim=1)) + self.shortcut(x)
        return out
    

class Pag(nn.Module):
    
    def __init__(
        self,
        in_dim,
        mid_dim,
        after_relu=False,
        with_channel=False,
    ):
        super(Pag, self).__init__()
        self.after_relu = after_relu
        self.with_channel = with_channel

        self.f_x = nn.Sequential(
            nn.Conv2d(in_dim, mid_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(mid_dim),
        )
        self.f_y = nn.Sequential(
            nn.Conv2d(in_dim, mid_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(mid_dim),
        )
        if with_channel:
            self.up = nn.Sequential(
                nn.Conv2d(mid_dim, in_dim, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(in_dim),
            )
        if after_relu:
            self.relu = nn.ReLU(inplace=True)            

    def forward(self, x, y):
        size = x.size()
        if self.after_relu:
            y = self.relu(y)
            x = self.relu(x)
        
        y_q = self.f_y(y)
        y_q = F.interpolate(
            y_q, size=(size[2], size[3]), mode='bilinear', align_corners=False
        )
        
        x_k = self.f_x(x)
        
        if self.with_channel:
            feat_att = torch.sigmoid(self.up(x_k * y_q))
        else:
            feat_att = torch.sigmoid(torch.sum(x_k * y_q, dim=1).unsqueeze(dim=1))
            
        y = F.interpolate(
            y, size=(size[2], size[3]), mode='bilinear', align_corners=False
        )
        
        out = (1 - feat_att) * x + feat_att * y
        return out
    

class LightBag(nn.Module):
    
    def __init__(
        self,
        in_dim,
        out_dim,
    ):
        super(LightBag, self).__init__()
        self.conv_p = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_dim),
        )
        self.conv_i = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_dim),
        )
        
    def forward(self, P, I, D):
        fea_att = torch.sigmoid(D)
        
        p_add = self.conv_p((1 - fea_att) * I + P)
        i_add = self.conv_i(fea_att * P + I)
        
        out = p_add + i_add
        return out
    
    
class Bag(nn.Module):
    
    def __init__(
        self,
        in_dim,
        out_dim,
    ):
        super(Bag, self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=False),
        )
        
    def forward(self, P, I, D):
        fea_att = torch.sigmoid(D)
        out = self.conv(P * fea_att + (1-fea_att) * I)
        return out