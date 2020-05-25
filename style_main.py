# -*- coding: utf-8 -*-
"""
Main entry for GAN training
Created on Sun Apr 19 13:22:06 2020

@author: delgallegon
"""

from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from torch.utils.tensorboard import SummaryWriter
from IPython.display import HTML

from model import sample_gan
from loaders import dataset_loader
from trainers import style_gan_trainer
import constants
     
def main():
    manualSeed = random.randint(1, 10000) # use if you want new results
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    writer = SummaryWriter('train_plot')
    
    gt = style_gan_trainer.GANTrainer(constants.STYLE_GAN_VERSION, constants.STYLE_ITERATION, device, writer)
    start_epoch = 0
    
    if(True): 
        checkpoint = torch.load(constants.STYLE_CHECKPATH)
        start_epoch = checkpoint['epoch'] + 1          
        gt.load_saved_state(checkpoint, constants.GENERATOR_KEY, constants.DISCRIMINATOR_KEY, constants.OPTIMIZER_KEY)
 
        print("Loaded checkpt ",constants.STYLE_CHECKPATH, "Current epoch: ", start_epoch)
        print("===================================================")
    
    # Create the dataloader
    dataloader = dataset_loader.load_style_dataset(constants.batch_size, 50000)
    
    # Plot some training images
    name_batch, vemon_batch, homog_batch, gta_batch = next(iter(dataloader))
    plt.figure(figsize=(constants.FIG_SIZE,constants.FIG_SIZE))
    plt.axis("off")
    plt.title("Training - Normal Images")
    plt.imshow(np.transpose(vutils.make_grid(vemon_batch.to(device)[:constants.batch_size], nrow = 8, padding=2, normalize=True).cpu(),(1,2,0)))
    plt.show()
    
    plt.figure(figsize=(constants.FIG_SIZE,constants.FIG_SIZE))
    plt.axis("off")
    plt.title("Training - Homog Images")
    plt.imshow(np.transpose(vutils.make_grid(homog_batch.to(device)[:constants.batch_size], nrow = 8, padding=2, normalize=True).cpu(),(1,2,0)))
    plt.show()
    
    plt.figure(figsize=(constants.FIG_SIZE,constants.FIG_SIZE))
    plt.axis("off")
    plt.title("Training - Topdown Images")
    plt.imshow(np.transpose(vutils.make_grid(gta_batch.to(device)[:constants.batch_size], nrow = 8, padding=2, normalize=True).cpu(),(1,2,0)))
    plt.show()
    
    print("Starting Training Loop...")
    for epoch in range(start_epoch, constants.num_epochs):
        # For each batch in the dataloader
        for i, (name, vemon_batch, homog_batch, gta_batch) in enumerate(dataloader, 0):
            vemon_tensor = vemon_batch.to(device)
            homog_tensor = homog_batch.to(device)
            gta_tensor = gta_batch.to(device)
            gt.train(vemon_tensor, gta_tensor, i)
        
        name_batch, vemon_batch, homog_batch, gta_batch = next(iter(dataloader))
        gt.verify(vemon_batch.to(device), gta_batch.to(device)) #produce image from first batch
        gt.report(epoch)
        
        #save every X epoch
        gt.save_states(epoch, constants.STYLE_CHECKPATH, constants.GENERATOR_KEY, constants.DISCRIMINATOR_KEY, constants.OPTIMIZER_KEY)

#FIX for broken pipe num_workers issue.
if __name__=="__main__": 
    main()

