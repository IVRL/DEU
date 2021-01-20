import torch.nn.functional as F
import os
import os.path
import numpy as np
import random
import torch
import cv2
import glob
from PIL import Image
import matplotlib.pyplot as plt
from scipy.fftpack import fft,ifft
from scipy.fftpack import dct,idct
from torchsummary import summary
from torchvision import transforms, datasets
import torch
import torchvision
from os import listdir
from os.path import isfile, join
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from skimage import metrics
import torch.nn as nn
import torch.optim as optim
import math

from RIDmodel import ops
from RIDmodel import common

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True

####### the pre-trained denoise model########

class DnCNN(nn.Module):
    def __init__(self, channels, num_of_layers=17):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        noise = self.dncnn(x)
        return noise

class DnCNN_RL(nn.Module):
    def __init__(self, channels, num_of_layers=3):
        super(DnCNN_RL, self).__init__()

        self.dncnn = DnCNN(channels=channels, num_of_layers=num_of_layers)

    def forward(self, x):
        noisy_input = x
        noise = self.dncnn(x)
        return noisy_input - noise
    
    
class MemNet(nn.Module):
    def __init__(self, in_channels, channels, num_memblock, num_resblock):
        super(MemNet, self).__init__()
        self.feature_extractor = BNReLUConv(in_channels, channels)
        self.reconstructor = BNReLUConv(channels, in_channels)
        self.dense_memory = nn.ModuleList(
            [MemoryBlock(channels, num_resblock, i+1) for i in range(num_memblock)]
        )

    def forward(self, x):
        # x = x.contiguous()
        residual = x
        out = self.feature_extractor(x)
        ys = [out]
        for memory_block in self.dense_memory:
            out = memory_block(out, ys)
        out = self.reconstructor(out)
        out = out + residual
        
        return out


class MemoryBlock(nn.Module):
    """Note: num_memblock denotes the number of MemoryBlock currently"""
    def __init__(self, channels, num_resblock, num_memblock):
        super(MemoryBlock, self).__init__()
        self.recursive_unit = nn.ModuleList(
            [ResidualBlock(channels) for i in range(num_resblock)]
        )
        self.gate_unit = BNReLUConv((num_resblock+num_memblock) * channels, channels, 1, 1, 0)

    def forward(self, x, ys):
        """ys is a list which contains long-term memory coming from previous memory block
        xs denotes the short-term memory coming from recursive unit
        """
        xs = []
        residual = x
        for layer in self.recursive_unit:
            x = layer(x)
            xs.append(x)
        
        gate_out = self.gate_unit(torch.cat(xs+ys, 1))
        ys.append(gate_out)
        return gate_out


class ResidualBlock(nn.Module):
    """ResidualBlock
    x - Relu - Conv - Relu - Conv - x
    """

    def __init__(self, channels, k=3, s=1, p=1):
        super(ResidualBlock, self).__init__()
        self.relu_conv1 = BNReLUConv(channels, channels, k, s, p)
        self.relu_conv2 = BNReLUConv(channels, channels, k, s, p)
        
    def forward(self, x):
        residual = x
        out = self.relu_conv1(x)
        out = self.relu_conv2(out)
        out = out + residual
        return out

class BNReLUConv(nn.Module):
    def __init__(self, in_channels, channels, k=3, s=1, p=1, inplace=True):
        super(BNReLUConv, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=inplace)
        self.conv = nn.Conv2d(in_channels, channels, k, s, p, bias=False)
    
    def forward(self,x):
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        return x
    
    
def make_model(args, parent=False):
    return RIDNET(args)



class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.c1 = ops.BasicBlock(channel , channel // reduction, 1, 1, 0)
        self.c2 = ops.BasicBlockSig(channel // reduction, channel , 1, 1, 0)

    def forward(self, x):
        y = self.avg_pool(x)
        y1 = self.c1(y)
        y2 = self.c2(y1)
        return x * y2

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, group=1):
        super(Block, self).__init__()

        self.r1 = ops.Merge_Run_dual(in_channels, out_channels)
        self.r2 = ops.ResidualBlock(in_channels, out_channels)
        self.r3 = ops.EResidualBlock(in_channels, out_channels)
        #self.g = ops.BasicBlock(in_channels, out_channels, 1, 1, 0)
        self.ca = CALayer(in_channels)

    def forward(self, x):
        
        r1 = self.r1(x)            
        r2 = self.r2(r1)       
        r3 = self.r3(r2)
        #g = self.g(r3)
        out = self.ca(r3)

        return out
        


class RIDNET(nn.Module):
    def __init__(self, in_channels):
        super(RIDNET, self).__init__()
        
        n_feats = 64
        kernel_size = 3
        reduction = 16
        rgb_range = 1
        
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        if in_channels == 1:
            rgb_mean = sum(rgb_mean)/3.0
            rgb_std = 1.0
        

        # Global avg pooling needed layers
        self.sub_mean = common.MeanShift(rgb_range, rgb_mean, rgb_std, in_channels=in_channels)       
        self.add_mean = common.MeanShift(rgb_range, rgb_mean, rgb_std, sign=1, in_channels=in_channels)

        self.head = ops.BasicBlock(in_channels, n_feats, kernel_size, 1, 1)

        self.b1 = Block(n_feats, n_feats)
        self.b2 = Block(n_feats, n_feats)
        self.b3 = Block(n_feats, n_feats)
        self.b4 = Block(n_feats, n_feats)

        self.tail = nn.Conv2d(n_feats, in_channels, kernel_size, 1, 1, 1)

    def forward(self, x):

        s = self.sub_mean(x)
        h = self.head(s)

        b1 = self.b1(h)
        b2 = self.b2(b1)
        b3 = self.b3(b2)
        b_out = self.b4(b3)

        res = self.tail(b_out)
        out = self.add_mean(res)
        f_out = out + x 

        return f_out 
    
###### the ensemble network of spatial domain#########


#### ensemble method---helpers class#########

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025,0.025)
        nn.init.constant_(m.bias.data, 0.0)

    
class ResBlock(nn.Module):
    def __init__(self,layers_num,mode_num,in_channels):
        super(ResBlock, self).__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels=mode_num*in_channels, out_channels=64, kernel_size=3, padding=1,bias=False))
        layers.append(nn.BatchNorm2d(64))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(layers_num):
            layers.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1,bias=False))
            layers.append(nn.BatchNorm2d(64))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=64, out_channels=mode_num*in_channels, kernel_size=3, padding=1,bias=False))
        layers.append(nn.BatchNorm2d(mode_num*in_channels))
        layers.append(nn.ReLU(inplace=True))
        
        self.resblocks = nn.Sequential(*layers)
        
    def forward(self, x):
        y=self.resblocks(x)
        return y
    
    
class SE_Block(nn.Module):
    def __init__(self, ch_in,ch_out):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_out),
            nn.ReLU(inplace=True),
            nn.Linear(ch_out, ch_in),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Ensemble_fusion(nn.Module):
    def __init__(self,mode_num,in_channels):
        super(Ensemble_fusion,self).__init__()
        self.resblocks = ResBlock(layers_num=3, mode_num=mode_num,in_channels=in_channels)
        self.seblock = SE_Block(ch_in=mode_num*in_channels,ch_out=mode_num*in_channels*2)
        
        self.fusion = nn.Conv2d(in_channels=mode_num*2*in_channels, out_channels=in_channels,kernel_size=1,bias=False)
                                  
    def forward(self,x):
        
        spatial_attention = x * F.softmax(self.resblocks(x), 1)
        
        channel_attention = self.seblock(x)
        return self.fusion(torch.cat((spatial_attention, channel_attention), 1))


class Spatial_attention(nn.Module):
    # use for the gray models
    def __init__(self,mode_num):
        super(Spatial_attention,self).__init__()
        self.resblocks = ResBlock(layers_num=3, mode_num=mode_num,in_channels=1) 
    def forward(self,x):
        spatial_attention = x * F.softmax(self.resblocks(x), 1)
        return spatial_attention.sum(dim=1, keepdim=True)


class Channel_attention(nn.Module):
    #use for gray models
    def __init__(self,mode_num):
        super(Channel_attention,self).__init__()
        self.seblock = SE_Block(ch_in=mode_num,ch_out=mode_num*2)
    def forward(self,x):
        channel_attention = self.seblock(x)
        return channel_attention.sum(dim=1, keepdim=True)