# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 16:39:17 2020

@author: delgallegon
"""
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
        #input = torch.cat([clean_like, clean_tensor], 1)
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.disc_layer(x)
        
        return x