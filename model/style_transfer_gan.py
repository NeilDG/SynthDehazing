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
            
class Generator(nn.Module):
    
    def __init__(self):
        super(Generator, self).__init__()
        
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels = 3, out_channels = 512, kernel_size=4, stride=2, padding=1),
                                   nn.BatchNorm2d(512),
                                   nn.ReLU(True))
        
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size=4, stride=2, padding=1),
                                   nn.BatchNorm2d(512),
                                   nn.ReLU(True),
                                   nn.Dropout(0.5))
        
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size=4, stride=2, padding=1),
                                   nn.BatchNorm2d(512),
                                   nn.ReLU(True),
                                   nn.Dropout(0.5))
        
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels = 512, out_channels = 128, kernel_size=4, stride=2, padding=1),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU(True),
                                   nn.Dropout(0.5))
        
        self.upconv1 = nn.Sequential(nn.ConvTranspose2d(in_channels = 128, out_channels = 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True))
    
        
        self.upconv2 = nn.Sequential(nn.ConvTranspose2d(in_channels = 512, out_channels = 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True))
        
        self.upconv3 = nn.Sequential(nn.ConvTranspose2d(in_channels = 512, out_channels = 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True))
        
        self.upconv4 = nn.Sequential(nn.ConvTranspose2d(in_channels = 512, out_channels = 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh())
        
        self.apply(weights_init)
        
       
    def forward(self, input):
       x1 = self.conv1(input)
       x2 = self.conv2(x1)
       x3 = self.conv3(x2)
       x4 = self.conv4(x3)
       
       y1 = self.upconv1(x4)
       y2 = self.upconv2(y1 + x3)
       y3 = self.upconv3(y2 + x2)
       y4 = self.upconv4(y3 + x1)
       
       #print("\tIn Model: input size: %s", input.size())
       
       return y4

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels = 3, out_channels = 256, kernel_size=4, stride=2, padding=1),
                                   nn.LeakyReLU(0.2, inplace = True))
        
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size=4, stride=2, padding=1),
                                   nn.BatchNorm2d(256),
                                   nn.LeakyReLU(0.2, inplace = True),
                                   nn.Dropout(0.5))
        
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size=4, stride=2, padding=1),
                                   nn.BatchNorm2d(256),
                                   nn.LeakyReLU(0.2, inplace = True),
                                   nn.Dropout(0.5))
        
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size=4, stride=2, padding=1),
                                   nn.BatchNorm2d(256),
                                   nn.LeakyReLU(0.2, inplace = True),
                                   nn.Dropout(0.5))
        
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size=4, stride=2, padding=1),
                                   nn.BatchNorm2d(256),
                                   nn.LeakyReLU(0.2, inplace = True))
        
        self.disc_layer = nn.Sequential(nn.Conv2d(in_channels = 256, out_channels = 1, kernel_size=4, stride=1, padding=0),
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