# -*- coding: utf-8 -*-
"""
GAN for performing style transfer of VEMON images to GTA images
Created on Fri Apr 17 12:28:51 2020

@author: delgallegon
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import logger


def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

def clamp(value, max):
    if(value > max):
        return max
    else:
        return value
            
class Generator(nn.Module):
    
    #Nblocks = number of layers (conv and upconv separate)
    #Filter_size = num channels of each layer
    #Expansion = Multiplier factor for each layer
    def __init__(self, nblocks = 4, filter_size = 64, expansion = 2, max_filter_size = 2048):
        super(Generator, self).__init__()
        
        self.conv_blocks = []
        self.upconv_blocks = []
        
        self.conv_blocks += nn.Sequential(nn.Conv2d(in_channels = 3, out_channels = filter_size, kernel_size=4, stride=2, padding=1),
                                   nn.BatchNorm2d(filter_size),
                                   nn.ReLU(True))
        
        new_size = filter_size
        for i in range(nblocks - 1):
            in_size = new_size
            out_size = clamp(new_size * expansion, max_filter_size)
            self.conv_blocks += nn.Sequential(nn.Conv2d(in_size, out_channels = out_size, kernel_size=4, stride=2, padding=1),
                                   nn.BatchNorm2d(out_size),
                                   nn.ReLU(True))
            
            new_size = out_size

        for i in range(nblocks - 1):
            in_size = new_size
            out_size = int(new_size / expansion)
            self.upconv_blocks += nn.Sequential(nn.ConvTranspose2d(in_channels = in_size, out_channels = out_size, kernel_size=4, stride=2, padding=1, bias=False),
                                        nn.BatchNorm2d(out_size),
                                        nn.ReLU(True))
            
            new_size = out_size
            
        
        self.upconv_blocks += nn.Sequential(nn.ConvTranspose2d(in_channels = new_size, out_channels = 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh())
        
        self.model = nn.Sequential(*self.conv_blocks, *self.upconv_blocks)
        self.apply(weights_init)
        
       
    def forward(self, input):
        #no skip connection
        return self.model(input)

class Discriminator(nn.Module):
    def __init__(self, n_blocks = 6, filter_size = 256, expansion = 2):
        super(Discriminator, self).__init__()
        
        self.conv_blocks = []
        self.conv_blocks += nn.Sequential(nn.Conv2d(in_channels = 6, out_channels = filter_size, kernel_size=4, stride=2, padding=1),
                                   nn.LeakyReLU(0.2, inplace = True))
        
        for i in range(n_blocks - 2):
            in_size = filter_size
            out_size = filter_size * expansion
            self.conv_blocks += nn.Sequential(nn.Conv2d(in_channels = in_size, out_channels = out_size, kernel_size=4, stride=2, padding=1),
                                   nn.BatchNorm2d(out_size),
                                   nn.LeakyReLU(0.2, inplace = True),
                                   nn.Dropout(0.5))
            filter_size = out_size
        
        
        self.conv_blocks += nn.Sequential(nn.Conv2d(in_channels = filter_size, out_channels = 1, kernel_size=4, stride=1, padding=0),
                                        nn.Sigmoid())
        
        self.model = nn.Sequential(*self.conv_blocks)
        self.apply(weights_init)

    def forward(self, clean_like, clean_tensor):
        x = torch.cat([clean_like, clean_tensor], 1)
        for i in range(len(self.conv_blocks)):
            x = self.conv_blocks[i](x)
        
        return x