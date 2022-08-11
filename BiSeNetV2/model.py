import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

# common block : conv + batch normalization + relu
class ConvBlock(nn.Module):
    def __init__(self, 
                 in_dim,
                 out_dim,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 groups=1,
                 bias=False):
        
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_dim, out_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

# Detail Branch
class DetailBranch(nn.Module):
    def __init__(self, in_dim=3, num_filters=64):
        super(DetailBranch, self).__init__()
        self.S1 = nn.Sequential(
            ConvBlock(in_dim, num_filters, stride=2),
            ConvBlock(num_filters, num_filters, stride=1),
        )
        self.S2 = nn.Sequential(
            ConvBlock(num_filters, num_filters, stride=2),
            ConvBlock(num_filters, num_filters, stride=1),
            ConvBlock(num_filters, num_filters, stride=1),
        )
        self.S3 = nn.Sequential(
            ConvBlock(num_filters, num_filters*2, stride=2),
            ConvBlock(num_filters*2, num_filters*2, stride=1),
            ConvBlock(num_filters*2, num_filters*2, stride=1),
        )
        
    def forward(self, x):
        x = self.S1(x)
        x = self.S2(x)
        x = self.S3(x)
        return x

# Stem Block
class StemBlock(nn.Module):
    def __init__(self, in_dim=3, num_filters=16):
        super(StemBlock, self).__init__()
        self.conv_head = ConvBlock(in_dim, num_filters, kernel_size=3, stride=2, padding=1)
        self.left_block = nn.Sequential(
            ConvBlock(num_filters, num_filters//2, kernel_size=1, stride=1, padding=0),
            ConvBlock(num_filters//2, num_filters, kernel_size=3, stride=2, padding=1),
        )
        self.right_block = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)
        self.conv_tail = ConvBlock(num_filters*2, num_filters, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        x = self.conv_head(x)
        x_left = self.left_block(x)
        x_right = self.right_block(x)
        x = torch.cat((x_left, x_right), dim=1)
        x = self.conv_tail(x)
        return x

# Context Embedding Block
class CEBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(CEBlock, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.bn = nn.BatchNorm2d(in_dim)
        self.gap_conv = ConvBlock(in_dim, out_dim, kernel_size=1, stride=1, padding=0)
        self.out_conv = ConvBlock(out_dim, out_dim, 
                                  kernel_size=3, stride=1, padding=1)
        
    def forward(self, inputs):
        x = self.gap(inputs)
        x = self.bn(x)
        x = self.gap_conv(x)
        x = torch.add(inputs, x)
        x = self.out_conv(x)
        return x

# Gather-and-Expansion Layer 1
class GELayer1(nn.Module):
    def __init__(self, in_dim, out_dim, ratio=6):
        super(GELayer1, self).__init__()
        expansion_dim = in_dim * ratio
        self.conv1 = ConvBlock(in_dim, in_dim, kernel_size=3, stride=1)
        self.dwconv = nn.Sequential(
            nn.Conv2d(in_dim, expansion_dim, 
                      kernel_size=3, stride=1, padding=1,
                      groups=in_dim, bias=False),
            nn.BatchNorm2d(expansion_dim),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(expansion_dim, out_dim, 
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_dim),
        )
        self.conv2[1].last_bn = True
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.dwconv(x)
        x = self.conv2(x)
        x = torch.add(inputs, x)
        x = self.relu(x)
        return x

# Gather-and-Expansion Layer 2
class GELayer2(nn.Module):
    def __init__(self, in_dim, out_dim, ratio=6):
        super(GELayer2, self).__init__()
        expansion_dim = in_dim * ratio
        # left block
        self.left_conv1 = ConvBlock(in_dim, in_dim, kernel_size=3, stride=1, padding=1)
        self.left_dwconv1 = nn.Sequential(
            nn.Conv2d(in_dim, expansion_dim,
                      kernel_size=3, stride=2, padding=1,
                      groups=in_dim, bias=False),
            nn.BatchNorm2d(expansion_dim),
        )
        self.left_dwconv2 = nn.Sequential(
            nn.Conv2d(expansion_dim, expansion_dim, 
                      kernel_size=3, stride=1, padding=1,
                      groups=expansion_dim, bias=False),
            nn.BatchNorm2d(expansion_dim),
            nn.ReLU(inplace=True),
        )
        self.left_conv2 = nn.Sequential(
            nn.Conv2d(expansion_dim, out_dim, 
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_dim),
        )
        self.left_conv2[1].last_bn = True
        # right block
        self.right_dwconv1 = nn.Sequential(
            nn.Conv2d(in_dim, in_dim,
                      kernel_size=3, stride=2, padding=1,
                      groups=in_dim, bias=False),
            nn.BatchNorm2d(in_dim),
        )
        self.right_conv1 = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_dim),
        )
        # last relu activation layer
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, inputs):
        # left block operation
        left_x = self.left_conv1(inputs)
        left_x = self.left_dwconv1(left_x)
        left_x = self.left_dwconv2(left_x)
        left_x = self.left_conv2(left_x)
        # right block operation
        right_x = self.right_dwconv1(inputs)
        right_x = self.right_conv1(right_x)
        # output operation
        sum_x = torch.add(left_x, right_x)
        out = self.relu(sum_x)
        return out

# Semantic Branch
class SemanticBranch(nn.Module):
    def __init__(self, num_filters=16):
        super(SemanticBranch, self).__init__()
        # stage 1 and stage 2
        self.stage1_2 = StemBlock()
        # stage 3
        self.stage3 = nn.Sequential(
            GELayer2(num_filters, num_filters*2),
            GELayer1(num_filters*2, num_filters*2),
        )
        # stage 4
        self.stage4 = nn.Sequential(
            GELayer2(num_filters*2, num_filters*4),
            GELayer1(num_filters*4, num_filters*4),
        )
        # stage 5
        self.stage5 = nn.Sequential(
            GELayer2(num_filters*4, num_filters*8),
            GELayer1(num_filters*8, num_filters*8),
            GELayer1(num_filters*8, num_filters*8),
            GELayer1(num_filters*8, num_filters*8),
        )
        # last : Context Embedding
        self.ce = CEBlock(num_filters*8, num_filters*8)
        
    def forward(self, x):
        s2 = self.stage1_2(x)
        s3 = self.stage3(s2)
        s4 = self.stage4(s3)
        s5 = self.stage5(s4)
        out = self.ce(s5)
        return s2, s3, s4, s5, out

class BGALayer(nn.Module):
    def __init__(self, in_dim=128):
        super(BGALayer, self).__init__()
        # This block receives output of detail branch
        self.detail_branch_keep = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 
                      kernel_size=3, stride=1, padding=1,
                      groups=in_dim, bias=False),
            nn.BatchNorm2d(in_dim),
            nn.Conv2d(in_dim, in_dim, 
                      kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.detail_branch_down = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 
                      kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(in_dim),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False),
        )
        # This block receives output of semantic branch
        self.semantic_branch_up = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_dim),
            nn.Upsample(scale_factor=4., mode='bilinear', align_corners=True),
            nn.Sigmoid(),
        )
        self.semantic_branch_keep = nn.Sequential(
            nn.Conv2d(in_dim, in_dim,
                      kernel_size=3, stride=1, padding=1,
                      groups=in_dim, bias=False),
            nn.BatchNorm2d(in_dim),
            nn.Conv2d(in_dim, in_dim, 
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid(),
        )
        # up-sample detail branch down + semantic branch keep
        self.upsample = nn.Upsample(scale_factor=4., mode='bilinear')
        # last layer: conv
        self.last_conv = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, detail_output, semantic_output):
        # receive output of detail branch
        detail_keep = self.detail_branch_keep(detail_output)
        detail_down = self.detail_branch_down(detail_output)
        # receive output of semantic branch
        semantic_up = self.semantic_branch_up(semantic_output)
        semantic_keep = self.semantic_branch_keep(semantic_output)
        # detail keep + semantic up
        left = torch.mul(detail_keep, semantic_up)
        # detail down + semantic keep
        right = torch.mul(detail_down, semantic_keep)
        right = self.upsample(right)
        # left + right
        last_sum = torch.add(left, right)
        # output
        out = self.last_conv(last_sum)
        
        return out

# Seg Head in booster
class SegHead(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim):
        super(SegHead, self).__init__()
        self.convblock = ConvBlock(in_dim, mid_dim, kernel_size=3, stride=1, padding=1)
        self.drop = nn.Dropout(0.1)
        self.conv_out = nn.Conv2d(mid_dim, out_dim, 
                                  kernel_size=1, stride=1, padding=0, bias=True)
        
    def forward(self, x, size):
        x = self.convblock(x)
        x = self.drop(x)
        x = self.conv_out(x)
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=True)
        return x

class BiSeNetV2(nn.Module):
    def __init__(self, num_classes, phase='train'):
        super(BiSeNetV2, self).__init__()
        assert phase in ('train', 'test')
        # load each branches and layer
        self.phase = phase
        self.detail = DetailBranch()
        self.semantic = SemanticBranch()
        self.aggregation = BGALayer()
        
        self.out_head = SegHead(128, 1024, num_classes)
        
        if self.phase=='train':
            self.s2_head = SegHead(16, 128, num_classes)
            self.s3_head = SegHead(32, 128, num_classes)
            self.s4_head = SegHead(64, 128, num_classes)
            self.s5_head = SegHead(128, 128, num_classes)
            
        # initialize weights
        self._init_weights_()
        
    def forward(self, inputs):
        size = inputs.size()[2:]
        detail_out = self.detail(inputs)
        s2, s3, s4, s5, semantic_out = self.semantic(inputs)
        out = self.aggregation(detail_out, semantic_out)
        
        if self.phase=='train':
            # up-sampling for seg head output
            s2 = self.s2_head(s2, size)
            s3 = self.s3_head(s3, size)
            s4 = self.s4_head(s4, size)
            s5 = self.s5_head(s5, size)
            out = self.out_head(out, size)
            return out, s2, s3, s4, s5
        
        else:
            out = self.out_head(out, size)
            return out
        
    def _init_weights_(self):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, mode='fan_out')
                if not module.bias is None: nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                if hasattr(module, 'last_bn') and module.last_bn:
                    nn.init.zeros_(module.weight)
                else:
                    nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)