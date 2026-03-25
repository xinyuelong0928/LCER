
import torch
import torch.nn as nn
import torch.nn.functional as F

from .InceptionEncoder import IncepEncoder, generation_init_weights
from utils.configs import Parser


argp = Parser()
arg = argp.get_args()
arg_dict = vars(arg)

class Adapter(nn.Module):
    def __init__(self, adapter_dim, embed_dim):
        super(Adapter, self).__init__()
        self.layer_norm = nn.BatchNorm2d(embed_dim)  
        self.down_project = nn.Conv2d(embed_dim, adapter_dim, kernel_size=1, bias=False)
        self.up_project = nn.Conv2d(adapter_dim, embed_dim, kernel_size=1, bias=False)

    def forward(self, z):
        normalized_z = self.layer_norm(z)
        h = F.relu(self.down_project(normalized_z))
        return self.up_project(h) + z

from torchvision.models import resnet50

class EncoderResNet50(nn.Module):
    def __init__(self):
        super(EncoderResNet50, self).__init__()
        resnet = resnet50(pretrained=True)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])

    def forward(self, x):
        return self.encoder(x)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(num_parameters=out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(num_parameters=out_channels)
            )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(

            nn.MaxPool2d(kernel_size=2,stride=2),
            DoubleConv(in_channels, out_channels))

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            pass

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class SEPool(nn.Module):
    def __init__(self, channels):
        super(SEPool, self).__init__()
        self.block = nn.Sequential(
            SEBlock(channels),
            nn.Conv2d(channels, channels, kernel_size=3, stride=4, padding=1),
            nn.BatchNorm2d(channels),
            nn.PReLU()
        )

    def forward(self, x):
        return self.block(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        _SCALE_ = 2

        self.encoder = EncoderResNet50()
        encoder_channels = 2048


        self.incepEncoder = IncepEncoder(use_inception=True, repeat_per_module=1, middel_layer_size=encoder_channels)
        self.Conv4Inception = DoubleConv(encoder_channels, encoder_channels)

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)
        self.down5 = Down(512, 1024)
        self.down6 = Down(1024, 2048)

        self.up1 = Up(4096, 1024, bilinear)
        self.up2 = Up(2048, 512, bilinear)
        self.up3 = Up(1024, 256, bilinear)
        self.up4 = Up(512, 128, bilinear)
        self.up5 = Up(256, 64, bilinear)
        self.up6 = Up(128, 32, bilinear)
        self.up7 = Up(64, 32, bilinear)
        self.outc = OutConv(32, n_classes)

        self.pool = SEPool(encoder_channels)
        self.adapter = Adapter(adapter_dim=128, embed_dim=2048) 

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x7 = self.down6(x6)

        x_encoder = self.encoder(x)
        x_encoder = self.incepEncoder(x_encoder)
        x_encoder = self.adapter(x_encoder)
        x_encoder= self.Conv4Inception(x_encoder)
        x_encoder = self.pool(x_encoder)

        x = self.up1(x_encoder, x7)
        x = self.up2(x, x6)
        x = self.up3(x, x5)
        x = self.up4(x, x4)
        x = self.up5(x, x3)
        x = self.up6(x, x2)
        x = self.up7(x, x1)

        logits = self.outc(x)
        return logits

    def init_weights(self):
        generation_init_weights(self)