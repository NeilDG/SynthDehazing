# -*- coding: utf-8 -*-
"""
GAN for performing style transfer of VEMON images to GTA images
Created on Fri Apr 17 12:28:51 2020

@author: delgallegon
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

print = logging.info

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
    
    def __init__(self, input_nc = 3, output_nc = 3, filter_size = 64, n_residual_blocks = 10):
        super(Generator, self).__init__()
        
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels = input_nc, out_channels = filter_size, kernel_size=4, stride=2, padding=1),
                                   nn.InstanceNorm2d(filter_size),
                                   nn.ReLU(True))
        
        in_size = filter_size
        out_size = in_size * 2
        
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels = in_size, out_channels = out_size, kernel_size=4, stride=2, padding=1),
                                   nn.InstanceNorm2d(out_size),
                                   nn.ReLU(True),
                                   nn.Dropout2d(0.5))
        
        in_size = out_size
        out_size = in_size * 2
        
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels = in_size, out_channels = out_size, kernel_size=4, stride=2, padding=1),
                                   nn.InstanceNorm2d(out_size),
                                   nn.ReLU(True),
                                   nn.Dropout2d(0.5))
        
        in_size = out_size
        out_size = in_size * 2
        
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels = in_size, out_channels = out_size, kernel_size=4, stride=2, padding=1),
                                   nn.InstanceNorm2d(out_size),
                                   nn.ReLU(True),
                                   nn.Dropout2d(0.5))
        
        
        #4 residual blocks
        # Residual blocks
        residuals = []
        for _ in range(n_residual_blocks):
            residuals += [ResidualBlock(out_size)]

        self.residuals = nn.Sequential(*residuals)
        # self.res1 = ResidualBlock(out_size)
        # self.res2 = ResidualBlock(out_size)
        # self.res3 = ResidualBlock(out_size)
        # self.res4 = ResidualBlock(out_size)
        # self.res5 = ResidualBlock(out_size)
        # self.res6 = ResidualBlock(out_size)
        # self.res7 = ResidualBlock(out_size)
        
        in_size = out_size
        out_size = int(in_size / 2)
        
        self.upconv1 = nn.Sequential(nn.ConvTranspose2d(in_channels = in_size, out_channels = out_size, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(True),
            nn.Dropout2d(0.5))
    
        in_size = out_size
        out_size = int(in_size / 2)
        
        self.upconv2 = nn.Sequential(nn.ConvTranspose2d(in_channels = in_size, out_channels = out_size, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(True),
            nn.Dropout2d(0.5))
        
        in_size = out_size
        out_size = int(in_size / 2)
        
        self.upconv3 = nn.Sequential(nn.ConvTranspose2d(in_channels = in_size, out_channels = out_size, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(True),
            nn.Dropout2d(0.5))
        
        in_size = out_size
        
        self.upconv4 = nn.Sequential(nn.ConvTranspose2d(in_channels = in_size, out_channels = output_nc, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh())
        
        self.apply(weights_init)
        
       
    def forward(self, input):
       x1 = self.conv1(input)
       x2 = self.conv2(x1)
       x3 = self.conv3(x2)
       x4 = self.conv4(x3)

       x4 = self.residuals(x4)
       # x4 = self.res1(x4)
       # x4 = self.res2(x4)
       # x4 = self.res3(x4)
       # x4 = self.res4(x4)
       # x4 = self.res5(x4)
       # x4 = self.res6(x4)
       # x4 = self.res7(x4)

       y1 = self.upconv1(x4)
       y2 = self.upconv2(y1 + x3)
       y3 = self.upconv3(y2 + x2)
       y4 = self.upconv4(y3)
       #y4 = self.upconv4(y3 + x1)

       #print("\tIn Model: input size: %s", input.size())
       
       return y4

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        filter_size = 256
        
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels = 3, out_channels = filter_size, kernel_size=4, stride=2, padding=1),
                                   nn.LeakyReLU(0.2, inplace = True))
        
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels = filter_size, out_channels = filter_size, kernel_size=4, stride=2, padding=1),
                                   nn.BatchNorm2d(filter_size),
                                   nn.LeakyReLU(0.2, inplace = True),
                                   nn.Dropout(0.5))
        
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels = filter_size, out_channels = filter_size, kernel_size=4, stride=2, padding=1),
                                   nn.BatchNorm2d(filter_size),
                                   nn.LeakyReLU(0.2, inplace = True),
                                   nn.Dropout(0.5))
        
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels = filter_size, out_channels = filter_size, kernel_size=4, stride=2, padding=1),
                                   nn.BatchNorm2d(filter_size),
                                   nn.LeakyReLU(0.2, inplace = True),
                                   nn.Dropout(0.5))
        
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels = filter_size, out_channels = filter_size, kernel_size=4, stride=2, padding=1),
                                   nn.BatchNorm2d(filter_size),
                                   nn.LeakyReLU(0.2, inplace = True))
        
        self.disc_layer = nn.Sequential(nn.Conv2d(in_channels = filter_size, out_channels = 1, kernel_size=4, stride=1, padding=0),
                                        nn.Sigmoid())
        
        self.apply(weights_init)

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.disc_layer(x)
        
        return x