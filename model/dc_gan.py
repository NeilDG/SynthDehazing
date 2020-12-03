# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 12:28:51 2020

@author: delgallegon
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    
    def __init__(self, num_channels, input_latent_size, gen_feature_size):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( input_latent_size, gen_feature_size * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(gen_feature_size * 8),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(gen_feature_size * 8, gen_feature_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(gen_feature_size * 4),
            nn.ReLU(True),
            
            nn.ConvTranspose2d( gen_feature_size * 4, gen_feature_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(gen_feature_size * 2),
            nn.ReLU(True),
            
            nn.ConvTranspose2d( gen_feature_size * 2, gen_feature_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(gen_feature_size),
            nn.ReLU(True),
            
            nn.ConvTranspose2d( gen_feature_size, num_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        
        self.apply(weights_init)
        
       
    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, num_channels, disc_feature_size):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x H x W.  Opposite of generator
            nn.Conv2d(num_channels, disc_feature_size, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(disc_feature_size, disc_feature_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(disc_feature_size * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(disc_feature_size * 2, disc_feature_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(disc_feature_size * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(disc_feature_size * 4, disc_feature_size * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(disc_feature_size * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(disc_feature_size * 8, 1, 4, 1, 0, bias=False)
        )
        
        self.fc_block = nn.Sequential(
            #nn.Linear(disc_feature_size * 128, disc_feature_size * 8),
            #nn.Linear(disc_feature_size * 8, disc_feature_size * 4),
            #nn.Linear(disc_feature_size * 4, disc_feature_size * 2),
            #nn.Linear(disc_feature_size * 2, disc_feature_size),
            #nn.Linear(disc_feature_size, 1),
            nn.Sigmoid())
        
        self.apply(weights_init)

    def forward(self, input):
        x = self.main(input)
        #x = torch.flatten(x, 1)
        x = self.fc_block(x)
        
        return x