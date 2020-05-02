# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 12:28:51 2020

@author: delgallegon
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import global_vars as gv
import numpy as np
from torchvision import models

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
        
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size=4, stride=2, padding=1),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(True))
        
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels = 64, out_channels = 256, kernel_size=4, stride=2, padding=1),
                                   nn.BatchNorm2d(256),
                                   nn.ReLU(True),
                                   nn.Dropout(0.5))
        
        self.adder_conv2 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size=1, stride=2, padding=0)
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size=4, stride=2, padding=1),
                                   nn.BatchNorm2d(256),
                                   nn.ReLU(True),
                                   nn.Dropout(0.5))
        
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels = 256, out_channels = 64, kernel_size=4, stride=2, padding=1),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(True),
                                   nn.Dropout(0.5))
        
        # self.conv5 = nn.Sequential(nn.Conv2d(in_channels = 256, out_channels = 128, kernel_size=4, stride=2, padding=1),
        #                            nn.BatchNorm2d(128),
        #                            nn.ReLU(True))
        
        self.upconv1 = nn.Sequential(nn.ConvTranspose2d(in_channels = 64, out_channels = 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True))
        
        self.adder_upconv1 = nn.Sequential(nn.ConvTranspose2d(in_channels = 256, out_channels = 256, kernel_size=1, stride=2, padding=0, bias=False),
                                           nn.ZeroPad2d((1, 0, 1, 0)))
        
        self.upconv2 = nn.Sequential(nn.ConvTranspose2d(in_channels = 256, out_channels = 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True))
        
        self.upconv3 = nn.Sequential(nn.ConvTranspose2d(in_channels = 256, out_channels = 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True))
        
        # self.upconv4 = nn.Sequential(nn.ConvTranspose2d(in_channels = 896, out_channels = 896, kernel_size=4, stride=2, padding=1, bias=False),
        #     nn.BatchNorm2d(896),
        #     nn.ReLU(True))
        
        # self.upconv5 = nn.Sequential(nn.ConvTranspose2d(in_channels = 896, out_channels = 896, kernel_size=2, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(896),
        #     nn.ReLU(True))
        
        self.upconv4 = nn.Sequential(nn.ConvTranspose2d(in_channels = 512, out_channels = 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh())
        
        self.apply(weights_init)
        
       
    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        
        x4 = self.conv4((self.adder_conv2(x2) + x3))
        
        x5 = self.upconv1(x4)
        x6 = self.upconv2(x5)
        
        x7 = self.upconv3(self.adder_upconv1(x5) + x6)
        x8 = self.upconv4(x7)
        
        return x8

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels = 6, out_channels = 128, kernel_size=4, stride=2, padding=1),
                                   nn.LeakyReLU(0.2, inplace = True))
        
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size=4, stride=2, padding=1),
                                   nn.BatchNorm2d(128),
                                   nn.LeakyReLU(0.2, inplace = True),
                                   nn.Dropout(0.5))
        
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size=4, stride=2, padding=1),
                                   nn.BatchNorm2d(128),
                                   nn.LeakyReLU(0.2, inplace = True),
                                   nn.Dropout(0.5))
        
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size=4, stride=2, padding=1),
                                   nn.BatchNorm2d(128),
                                   nn.LeakyReLU(0.2, inplace = True),
                                   nn.Dropout(0.5))
        
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size=4, stride=2, padding=1),
                                   nn.BatchNorm2d(128),
                                   nn.LeakyReLU(0.2, inplace = True))
        
        # self.conv6 = nn.Sequential(nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size=4, stride=2, padding=1),
        #                            nn.BatchNorm2d(512),
        #                            nn.LeakyReLU(0.2, inplace = True))
        
        self.disc_layer = nn.Sequential(nn.Conv2d(in_channels = 128, out_channels = 1, kernel_size=4, stride=1, padding=0),
                                        nn.Sigmoid())
        
        self.apply(weights_init)

    def forward(self, normal_tensor, topdown_tensor):
        #print("Normal shape: ", np.shape(normal_tensor), " Topdown shape: ", np.shape(topdown_tensor))
        input = torch.cat([normal_tensor, topdown_tensor], 1)
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        #x = self.conv6(x)
        x = self.disc_layer(x)
        
        return x