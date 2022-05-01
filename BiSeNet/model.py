import torch
import torch.nn as nn
import torchvision.models as models

num_classes = 19

# Conv + Batch Normalization + ReLU
class ConvBlock(nn.Module):
    def __init__(self, 
                 in_dim, 
                 out_dim, 
                 kernel_size=3, 
                 stride=2, 
                 padding=1, 
                 bias=False):
        
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 
                      kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x):
        return self.block(x)

# Define Spatial Path
class SpatialPath(nn.Module):
    def __init__(self, in_dim=3, num_filters=64):
        super(SpatialPath, self).__init__()
        self.convblock1 = ConvBlock(in_dim, num_filters)
        self.convblock2 = ConvBlock(num_filters, num_filters*2)
        self.convblock3 = ConvBlock(num_filters*2, num_filters*4)
        
    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        return x

# Define Attention Refinement Module
class AttentionRefinementModule(nn.Module):
    def __init__(self, out_dim):
        super(AttentionRefinementModule, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.conv = nn.Conv2d(out_dim, out_dim, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_dim)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, inputs):
        x = self.gap(inputs)
        x = self.conv(x)
        x = self.bn(x)
        x = self.sigmoid(x)
        out = torch.mul(inputs, x)
        return out

# Define ResNet
class ResNet(nn.Module):
    def __init__(self, pretrained_model='resnet18'):
        super(ResNet, self).__init__()
        
        # load pre-trained model called lightweight model in context path
        if pretrained_model=='resnet18':
            features = models.resnet18(pretrained=True)
        elif pretrained_model=='resnet101':
            features = models.resnet101(pretrained=True)
        else:
            raise ValueError('You should be select resnet18 or resnet101')
            
        self.conv = features.conv1
        self.bn = features.bn1
        self.relu = features.relu
        self.maxpool = features.maxpool
        # 4x down sampling
        self.layer1 = features.layer1
        # 8x down sampling
        self.layer2 = features.layer2
        # 16x down sampling
        self.layer3 = features.layer3
        # 32x down sampling
        self.layer4 = features.layer4
        
    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)
        feat4 = self.layer1(x)
        feat8 = self.layer2(feat4)
        feat16 = self.layer3(feat8)
        feat32 = self.layer4(feat16)
        return feat16, feat32

# Define Context Path
class ContextPath(nn.Module):
    def __init__(self, out_dim=num_classes, pretrained_model='resnet18'):
        super(ContextPath, self).__init__()
        
        # load pre-trained model 
        self.resnet = ResNet(pretrained_model=pretrained_model)
        
        self.gap = nn.AdaptiveAvgPool2d(output_size=(1,1))
        
        # using attention refinement module
        if pretrained_model=='resnet18':
            self.arm16 = AttentionRefinementModule(256)
            self.arm32 = AttentionRefinementModule(512)
            
        elif pretrained_model=='resnet101':
            self.arm16 = AttentionRefinementModule(1024)
            self.arm32 = AttentionRefinementModule(2048)
        
    def forward(self, inputs):
        # load pre-trained resnet
        feat16, feat32 = self.resnet(inputs)
        # global average pooling operation
        tail = self.gap(feat32)    
        # operate feature 32 features layer
        feat32 = self.arm32(feat32)        
        # operate feature 16 features layer
        feat16 = self.arm16(feat16)
        
        return feat16, feat32, tail

# Define Feature Fusion Module
class FeatureFusionModule(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FeatureFusionModule, self).__init__()
        self.convblock = ConvBlock(in_dim, out_dim, kernel_size=1, stride=1, padding=0)
        self.gap = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.conv1 = nn.Conv2d(out_dim, out_dim, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, input_1, input_2):
        x = torch.cat((input_1, input_2), dim=1)
        features = self.convblock(x)
        x = self.gap(features)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.sigmoid(x)
        x = torch.mul(features, x)
        x = torch.add(features, x)
        return x

class BiSeNetOutput(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim, output_size):
        super(BiSeNetOutput, self).__init__()
        
        self.convblock = ConvBlock(in_dim, mid_dim, kernel_size=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_dim, out_dim, kernel_size=1, bias=True)
        self.up = nn.Upsample(size=output_size, mode='bilinear')
        
    def forward(self, x):
        x = self.convblock(x)
        x = self.conv_out(x)
        x = self.up(x)
        return x

# build BiSeNet
class BiSeNet(nn.Module):
    def __init__(self, output_size, num_classes=num_classes, pretrained_model='resnet18', mode='train'):
        super(BiSeNet, self).__init__()
        # load Spatial Path
        self.spatial_path = SpatialPath()
        # load Context Path
        self.context_path = ContextPath(pretrained_model=pretrained_model)
        
        self.up16 = nn.Upsample(size=(45,60), mode='bilinear')
        self.up32 = nn.Upsample(size=(23,30), mode='bilinear')
        
        # load feature fusion module
        if pretrained_model=='resnet18':
            self.feature_fusion = FeatureFusionModule(256+512+256, 256)
            self.conv_out32 = BiSeNetOutput(512, 128, num_classes, output_size=output_size)
            self.conv_out16 = BiSeNetOutput(256, 128, num_classes, output_size=output_size)
            self.bisenet_output = BiSeNetOutput(256, 64, num_classes, output_size=output_size)
            
        elif pretrained_model=='resnet101':
            self.feature_fusion = FeatureFusionModule(1024+2048+256, 512)
            self.conv_out32 = BiSeNetOutput(2048, 128, num_classes, output_size=output_size)
            self.conv_out16 = BiSeNetOutput(1024, 128, num_classes, output_size=output_size)
            self.bisenet_output = BiSeNetOutput(512, 128, num_classes, output_size=output_size)
            
        else:
            raise ValueError('You should be select resnet18 or resnet101')
        
        # initialize weights
        self._init_weight_()
        
    def forward(self, inputs):
        # Spatial Path
        sx = self.spatial_path(inputs)
        
        # Context Path
        feat16, feat32, tail = self.context_path(inputs)
        feat32_gap = torch.mul(feat32, tail)
        feat32_up = self.up32(feat32_gap)
        cx = torch.cat((feat16, feat32_up), dim=1)
        cx = self.up16(cx)
        
        # FFM
        ffm_output = self.feature_fusion(sx, cx)
        
        # upsampling
        output = self.bisenet_output(ffm_output)
        
        # extract arm16 output and amr32 output for calculate auxiliary loss 
        out32 = self.conv_out32(feat32)
        out16 = self.conv_out16(feat16)
        
        return output, out16, out32
    
    def _init_weight_(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if not m.bias is None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

if __name__ == "__main__":
    from torchsummary import summary
    summary(BiSeNet(), (3, 1024, 2048), device='cpu')