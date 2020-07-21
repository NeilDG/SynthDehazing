# -*- coding: utf-8 -*-
"""
Div2k GAN that adds features from GTA Denoise GAN
Created on Mon Jun 29 14:30:24 2020

@author: delgallegon
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class Generator(nn.Module):
    def __init__(self, input_nc=3):
        super(Generator, self).__init__()
        self.conv1 = [nn.ReflectionPad2d(2),
                    nn.Conv2d(input_nc, 64, 8),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True)]
        
        # Downsampling
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            self.conv1 += [  nn.Conv2d(in_features, out_features, 4, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2
            
        self.conv1 = nn.Sequential(*self.conv1)
        
        #3 residual blocks by default
        self.res1 = ResidualBlock(in_features)
        
        in_features = in_features + 256 #to account for concat
        self.res2 = ResidualBlock(in_features)
        
        in_features = in_features + 256 # to account for concat
        self.res3 = ResidualBlock(in_features)
        
        # Upsampling
        in_features = in_features + 256 # to account for concat
        out_features = in_features//2
        self.conv5 = []
        for _ in range(2):
            self.conv5 += [  nn.ConvTranspose2d(in_features, out_features, 4, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2
        
        # Output layer
        self.conv5 += [  nn.ReflectionPad2d(4),
                    nn.Conv2d(256, 3, 8),
                    nn.Tanh() ]
        
        self.conv5 = nn.Sequential(*self.conv5)
        
    
    def forward(self, x, feature_getter):
        x = self.conv1(x)
        x = self.res1(x)
        x = self.res2(torch.cat([x, feature_getter[0]], 1))
        x = self.res3(torch.cat([x, feature_getter[1]], 1))
        
        return self.conv5(torch.cat([x, feature_getter[2]], 1))
        