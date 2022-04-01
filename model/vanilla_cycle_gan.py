# -*- coding: utf-8 -*-
"""
Vanilla Cycle GAN
Created on Mon Jun 29 14:30:24 2020

@author: delgallegon
"""


import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, input_nc=3, output_nc=3, downsampling_blocks = 2, n_residual_blocks=6, has_dropout = True):
        super(Generator, self).__init__()

        # Initial convolution block       
        model = [   nn.ReflectionPad2d(2),
                    nn.Conv2d(input_nc, 64, 8),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True) ]

        # Downsampling
        in_features = 64
        out_features = in_features*2
        for _ in range(downsampling_blocks):
            model += [  nn.Conv2d(in_features, out_features, 4, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True)
                    ]

            if(has_dropout):
                model +=[nn.Dropout2d(p = 0.1)]
            in_features = out_features
            out_features = clamp(in_features*2, 8192)

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features//2
        for _ in range(downsampling_blocks):
            model += [  nn.ConvTranspose2d(in_features, out_features, 4, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True)]

            if (has_dropout):
                model += [nn.Dropout2d(p=0.1)]
            in_features = out_features
            out_features = in_features//2

        # Output layer
        model += [  nn.ReflectionPad2d(4),
                    nn.Conv2d(64, output_nc, 8),
                    nn.Tanh() ]

        self.model = nn.Sequential(*model)
        self.model.apply(weights_init)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, input_nc=3, use_bce=0):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                 nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(64, 128, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(128),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(128, 256, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(256),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(256, 512, 4, padding=1),
                  nn.InstanceNorm2d(512),
                  nn.LeakyReLU(0.2, inplace=True)]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]
        if (use_bce == 1):
            model += [nn.Sigmoid()]

        self.model = nn.Sequential(*model)
        self.model.apply(weights_init)

    def forward(self, x):
        x = self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)

class FeatureDiscriminator(nn.Module):
    def __init__(self, in_channels, n_blocks = 3, expansion = 2, max_filter_size = 2048):
        super(FeatureDiscriminator, self).__init__()
        
        filter_size = in_channels * expansion
        
        self.conv_blocks = []
        self.conv_blocks += nn.Sequential(nn.Conv2d(in_channels = in_channels, out_channels = filter_size, kernel_size=4, stride=2, padding=1),
                                   nn.LeakyReLU(0.2, inplace = True))
        
        for i in range(n_blocks - 2):
            in_size = filter_size
            out_size = clamp(filter_size * expansion, max_filter_size)
            self.conv_blocks += nn.Sequential(nn.Conv2d(in_channels = in_size, out_channels = out_size, kernel_size=4, stride=2, padding=1),
                                   nn.BatchNorm2d(out_size),
                                   nn.LeakyReLU(0.2, inplace = True),
                                   nn.Dropout(0.5))
            filter_size = out_size
        
        
        self.conv_blocks += nn.Sequential(nn.Conv2d(in_channels = filter_size, out_channels = 1, kernel_size=4, stride=2, padding=1),
                                        nn.Sigmoid())
        
        self.model = nn.Sequential(*self.conv_blocks)
        self.apply(weights_init)

    def forward(self, feature_tensor):
        return self.model(feature_tensor)