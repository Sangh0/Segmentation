import torch
import torch.nn as nn
import torchvision.models as models
from torchsummary import summary

num_classes = 12

# Encoder Block
class EncoderBlock(nn.Module):
    def __init__(self, index):
        super(EncoderBlock, self).__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.block = nn.Sequential(
            vgg16.features[index],
            nn.BatchNorm2d(vgg16.features[index].out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.block(x)

# Decoder Block
class DecoderBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(DecoderBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.block(x)

# SegNet
class SegNet(nn.Module):
    def __init__(self, num_filters=64, in_dim=3, out_dim=num_classes):
        super(SegNet, self).__init__()
        # define convolution layer indices of vgg16 
        vgg_conv_index = [0,2,5,7,10,12,14,17,19,21,24,26,28]
        
        ### stage 1, 2 has 2 of Conv layer + Batch Norm layer + ReLU activation layer
        ### stage 3, 4, 5 has 3 of Conv layer + Batch Norm layer + ReLU activation layer
        
        ######################### ENCODER #########################
        # first encoding
        self.encoder1 = EncoderBlock(vgg_conv_index[0])
        self.encoder2 = EncoderBlock(vgg_conv_index[1])
        self.down1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        # second encoding
        self.encoder3 = EncoderBlock(vgg_conv_index[2])
        self.encoder4 = EncoderBlock(vgg_conv_index[3])
        self.down2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        # third encoding
        self.encoder5 = EncoderBlock(vgg_conv_index[4])
        self.encoder6 = EncoderBlock(vgg_conv_index[5])
        self.encoder7 = EncoderBlock(vgg_conv_index[6])
        self.down3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        # fourth encoding
        self.encoder8 = EncoderBlock(vgg_conv_index[7])
        self.encoder9 = EncoderBlock(vgg_conv_index[8])
        self.encoder10 = EncoderBlock(vgg_conv_index[9])
        self.down4 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        # fifth encoding
        self.encoder11 = EncoderBlock(vgg_conv_index[10])
        self.encoder12 = EncoderBlock(vgg_conv_index[11])
        self.encoder13 = EncoderBlock(vgg_conv_index[12])
        self.down5 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        ###########################################################
        ######################## SYMMETRIC ########################
        ###########################################################
        
        ######################### DECODER #########################
        # first decoding
        self.up1 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.decoder1 = DecoderBlock(num_filters*8, num_filters*8)
        self.decoder2 = DecoderBlock(num_filters*8, num_filters*8)
        self.decoder3 = DecoderBlock(num_filters*8, num_filters*8)
        # second decoding
        self.up2 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.decoder4 = DecoderBlock(num_filters*8, num_filters*8)
        self.decoder5 = DecoderBlock(num_filters*8, num_filters*8)
        self.decoder6 = DecoderBlock(num_filters*8, num_filters*4)
        # third decoding
        self.up3 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.decoder7 = DecoderBlock(num_filters*4, num_filters*4)
        self.decoder8 = DecoderBlock(num_filters*4, num_filters*4)
        self.decoder9 = DecoderBlock(num_filters*4, num_filters*2)
        # fourth decoding
        self.up4 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.decoder10 = DecoderBlock(num_filters*2, num_filters*2)
        self.decoder11 = DecoderBlock(num_filters*2, num_filters)
        # fifth decoding
        self.up5 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.decoder12 = DecoderBlock(num_filters, num_filters)
        self.decoder13 = DecoderBlock(num_filters, out_dim)
        
    def forward(self, x):
        ######################### ENCODING #########################
        # stage 1
        size1 = x.size()
        x = self.encoder1(x)
        x = self.encoder2(x)
        x, idx1 = self.down1(x)
        # stage 2
        size2 = x.size()
        x = self.encoder3(x)
        x = self.encoder4(x)
        x, idx2 = self.down2(x)
        # stage 3
        size3 = x.size()
        x = self.encoder5(x)
        x = self.encoder6(x)
        x = self.encoder7(x)
        x, idx3 = self.down3(x)
        # stage 4
        size4 = x.size()
        x = self.encoder8(x)
        x = self.encoder9(x)
        x = self.encoder10(x)
        x, idx4 = self.down4(x)
        # stage 5
        size5 = x.size()
        x = self.encoder11(x)
        x = self.encoder12(x)
        x = self.encoder13(x)
        x, idx5 = self.down5(x)
        
        ######################### DECODING #########################
        # stage 5
        x = self.up1(x, idx5, output_size=size5)
        x = self.decoder1(x)
        x = self.decoder2(x)
        x = self.decoder3(x)
        # stage 4
        x = self.up2(x, idx4, output_size=size4)
        x = self.decoder4(x)
        x = self.decoder5(x)
        x = self.decoder6(x)
        # stage 3
        x = self.up3(x, idx3, output_size=size3)
        x = self.decoder7(x)
        x = self.decoder8(x)
        x = self.decoder9(x)
        # stage 2
        x = self.up4(x, idx2, output_size=size2)
        x = self.decoder10(x)
        x = self.decoder11(x)
        # stage 1
        x = self.up5(x, idx1, output_size=size1)
        x = self.decoder12(x)
        out = self.decoder13(x)
        
        return out
    
if __name__ == "__main__":
    summary(SegNet(), (3, 720, 960), device='cpu')
