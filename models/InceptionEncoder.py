import torch.nn as nn
import torch.nn.functional as F

from .lora_layers import LoRA_Conv2d
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

def generation_init_weights(module):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):

            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.normal_(m.weight, 0.0, 0.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    module.apply(init_func)

class create_encoder_single_conv(nn.Module):
    def __init__(self, in_chs, out_chs, kernel, r, lora_alpha, lora_dropout=0):
        super().__init__()
        assert kernel % 2 == 1

        self.single_Conv = nn.Sequential(
            LoRA_Conv2d(in_chs, out_chs, kernel_size=kernel, padding=(kernel - 1) // 2, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout),
            nn.BatchNorm2d(out_chs),
            nn.PReLU(num_parameters=out_chs)
            )

    def forward(self, x):
        out = self.single_Conv(x)
        return out

class EncoderInceptionModuleSignle(nn.Module):
    def __init__(self, channels, adapter_dim=16):
        super().__init__()
        bn_ch = channels // 2

        self.bottleneck = create_encoder_single_conv(channels, bn_ch, 1, r=0, lora_alpha=1)

        if arg_dict['mode'] == 'pretrain':
            self.conv1 = create_encoder_single_conv(bn_ch, channels, 1, r=0, lora_alpha=1)
            self.conv3 = create_encoder_single_conv(bn_ch, channels, 3, r=0, lora_alpha=1)
            self.conv5 = create_encoder_single_conv(bn_ch, channels, 5, r=0, lora_alpha=1)
            self.conv7 = create_encoder_single_conv(bn_ch, channels, 7, r=0, lora_alpha=1)
        else:
            self.conv1 = create_encoder_single_conv(bn_ch, channels, 1, r=0, lora_alpha=1)
            self.adapter_conv1 = Adapter(adapter_dim, channels)
            self.conv3 = create_encoder_single_conv(bn_ch, channels, 3, r=8, lora_alpha=16, lora_dropout = 0.5)
            self.conv5 = create_encoder_single_conv(bn_ch, channels, 5, r=0, lora_alpha=1)
            self.adapter_conv5 = Adapter(adapter_dim, channels)
            self.conv7 = create_encoder_single_conv(bn_ch, channels, 7, r=8, lora_alpha=16, lora_dropout = 0.5)

        self.pool3 = nn.MaxPool2d(3, stride=1, padding=1)
        self.pool5 = nn.MaxPool2d(5, stride=1, padding=2)

    def forward(self, x):
        bn = self.bottleneck(x)

        if arg_dict['mode'] == 'pretrain':
            conv1 = self.conv1(bn)
            conv3 = self.conv3(bn)
            conv5 = self.conv5(bn)
            conv7 = self.conv7(bn)
        else:
            conv1 = self.adapter_conv1(self.conv1(bn))
            conv3 = self.conv3(bn)
            conv5 = self.adapter_conv5(self.conv5(bn))
            conv7 = self.conv7(bn)

        pool3 = self.pool3(x)
        pool5 = self.pool5(x)

        out = conv1 + conv3 + conv5 + conv7 + pool3 + pool5

        return out

class EncoderModule(nn.Module):
    def __init__(self, chs, repeat_num, use_inception):
        super().__init__()
        if use_inception:
            layers = [EncoderInceptionModuleSignle(chs) for i in range(repeat_num)]
        else:
            layers = [create_encoder_single_conv(chs, chs, 3) for i in range(repeat_num)]
        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        return self.convs(x)

class IncepEncoder(nn.Module):
    def __init__(self,use_inception, repeat_per_module, middel_layer_size=256 ):
        super().__init__()
        self.encoderPart = EncoderModule(middel_layer_size, repeat_per_module, use_inception)

    def forward(self, x):
        out = self.encoderPart(x)
        return out

    def init_weights(self):
        generation_init_weights(self)
