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
from trainers import gan_trainer
import constants

def main():
    # Set random seed for reproducibility
    manualSeed = 999
    
    # Number of training epochs
    num_epochs = 2000
    
    # Batch size during training
    batch_size = 16
    
    #manualSeed = random.randint(1, 10000) # use if you want new results
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    writer = SummaryWriter('train_plot')
    
    GAN_VERSION = "gan_v1.02"
    GAN_ITERATION = "2"
    OPTIMIZER_KEY = "optimizer"
    CHECKPATH = 'checkpoint/' + GAN_VERSION +'.pt'
    GENERATOR_KEY = "generator"
    DISCRIMINATOR_KEY = "discriminator"
    
    gt = gan_trainer.GANTrainer(GAN_VERSION, GAN_ITERATION, device, writer, batch_size)
    start_epoch = 0
    
    if(True): 
        checkpoint = torch.load(CHECKPATH)
        start_epoch = checkpoint['epoch'] + 1          
        gt.load_saved_state(checkpoint, GENERATOR_KEY, DISCRIMINATOR_KEY, OPTIMIZER_KEY)
 
        print("Loaded checkpt ",CHECKPATH, "Current epoch: ", start_epoch)
        print("===================================================")
    
    # Create the dataloader
    dataloader = dataset_loader.load_dataset(batch_size, -1)
    
    # Plot some training images
    file_name_batch, real_batch = next(iter(dataloader))
    plt.figure(figsize=(constants.FIG_SIZE,constants.FIG_SIZE))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch.to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
    plt.show()
    
    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(start_epoch, num_epochs):
        # For each batch in the dataloader
        for i, (file_name, data) in enumerate(dataloader, 0):
            real_gpu = data.to(device)
            gt.train(real_gpu, i)
        
        gt.verify()
        gt.report(epoch)
        
        #save every X epoch
        gt.save_states(epoch, CHECKPATH, GENERATOR_KEY, DISCRIMINATOR_KEY, OPTIMIZER_KEY)



#FIX for broken pipe num_workers issue.
if __name__=="__main__": 
    main()

