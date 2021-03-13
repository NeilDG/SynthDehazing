# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 16:39:17 2020

@author: delgallegon
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
            
class Discriminator(nn.Module):
    def __init__(self, input_nc = 3):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                 nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(64, 128, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(128),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(128, 256, 2, stride=2, padding=1),
                  nn.InstanceNorm2d(256),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(256, 512, 2, padding=1),
                  nn.InstanceNorm2d(512),
                  nn.LeakyReLU(0.2, inplace=True)]

        # FCN classification layer
        model += nn.Sequential(nn.Conv2d(in_channels = 512, out_channels = 1, kernel_size=4, stride=1, padding=0),
                                        nn.Sigmoid())

        self.model = nn.Sequential(*model)
        self.apply(weights_init)

    def forward(self, x):
        return self.model(x)


class AirlightEstimator(nn.Module):
    def __init__(self, input_nc = 3, num_layers = 3):
        super(AirlightEstimator, self).__init__()

        # A bunch of convolutions one after another
        model = [nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                 nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(64, 128, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(128),
                  nn.LeakyReLU(0.2, inplace=True)]

        in_filters = 128
        out_filters = in_filters * 2

        for i in range(0, num_layers):
            model += [nn.Conv2d(in_filters, out_filters, 2, stride=2, padding=1),
                      nn.InstanceNorm2d(out_filters),
                      nn.LeakyReLU(0.2, inplace=True)]

            in_filters = out_filters
            out_filters = in_filters * 2


        out_filters = int(in_filters / 2)

        for i in range(0, num_layers):
            model += [nn.Conv2d(in_filters, out_filters, 2, stride=2, padding=1),
                      nn.InstanceNorm2d(out_filters),
                      nn.LeakyReLU(0.2, inplace=True)]

            in_filters = out_filters
            out_filters = int(in_filters / 2)

        # FCN classification layer
        model += nn.Sequential(nn.Conv2d(in_channels = in_filters, out_channels = 64, kernel_size=4, stride=1, padding=0),
                               nn.Sigmoid(),
                               nn.Flatten(),
                               nn.Linear(in_features= 64 * 2 * 2, out_features=32),#the feature map size after sigmoid
                               nn.Linear(in_features=32, out_features=1))

        self.model = nn.Sequential(*model)
        self.apply(weights_init)

    def forward(self, x):
        y = self.model(x)
        #print(np.shape(y))

        return y