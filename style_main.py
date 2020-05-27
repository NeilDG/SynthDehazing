# -*- coding: utf-8 -*-
"""
Main entry for GAN training
Created on Sun Apr 19 13:22:06 2020

@author: delgallegon
"""

from __future__ import print_function
import os
import sys
import logging
from optparse import OptionParser
import random
import torch
import torch.nn.parallel
import torch.utils.data
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from loaders import dataset_loader
from trainers import style_gan_trainer
import constants
     
parser = OptionParser()
parser.add_option('--coare', type=int, help="Is running on COARE?", default=0)
parser.add_option('--img_to_load', type=int, help="Image to load?", default=-1)
logging.basicConfig(level=logging.INFO)
print = logging.info

#Update config if on COARE
def update_config():
    if(constants.is_coare == 1):
        print("Using COARE configuration.")
        constants.batch_size = constants.batch_size * 16
        
        constants.DATASET_BIRD_NORMAL_PATH = "/scratch1/scratch2/neil.delgallego/VEMON Dataset/pending/frames/"
        constants.DATASET_BIRD_HOMOG_PATH = "/scratch1/scratch2/neil.delgallego/VEMON Dataset/pending/homog_frames/"
        constants.DATASET_BIRD_GROUND_TRUTH_PATH = "/scratch1/scratch2/neil.delgallego/VEMON Dataset/pending/topdown_frames/"
        #DATASET_BIRD_ALTERNATIVE_PATH = "E:/GTA Bird Dataset/raw/"
        
        constants.DATASET_VEMON_FRONT_PATH = "/scratch1/scratch2/neil.delgallego/VEMON Dataset/frames/"
        constants.DATASET_VEMON_HOMOG_PATH = "/scratch1/scratch2/neil.delgallego/VEMON Dataset/homog_frames/" 

def main(argv):
    (opts, args) = parser.parse_args(argv)
    constants.is_coare = opts.coare
    
    print("=========BEGIN============")
    print("Is Coare? %d Has GPU available? %d Count: %d", constants.is_coare, torch.cuda.is_available(), torch.cuda.device_count())
    print("Torch CUDA version: %s" ,torch.version.cuda)
    update_config()
    
    manualSeed = random.randint(1, 10000) # use if you want new results
    print("Random Seed: %d", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    print("Device: %s", device)
    writer = SummaryWriter('train_plot')
    
    gt = style_gan_trainer.GANTrainer(constants.STYLE_GAN_VERSION, constants.STYLE_ITERATION, device, writer)
    start_epoch = 0
    
    if(False): 
        checkpoint = torch.load(constants.STYLE_CHECKPATH)
        start_epoch = checkpoint['epoch'] + 1          
        gt.load_saved_state(checkpoint, constants.GENERATOR_KEY, constants.DISCRIMINATOR_KEY, constants.OPTIMIZER_KEY)
 
        print("Loaded checkpt: %s Current epoch: %d", constants.STYLE_CHECKPATH, start_epoch)
        print("===================================================")
    
    # Create the dataloader
    dataloader = dataset_loader.load_style_dataset(constants.batch_size, opts.img_to_load)
    
    # Plot some training images
    if(constants.is_coare == 0):
        name_batch, vemon_batch, gta_batch = next(iter(dataloader))
        plt.figure(figsize=(constants.FIG_SIZE,constants.FIG_SIZE))
        plt.axis("off")
        plt.title("Training - Normal Images")
        plt.imshow(np.transpose(vutils.make_grid(vemon_batch.to(device)[:constants.batch_size], nrow = 8, padding=2, normalize=True).cpu(),(1,2,0)))
        plt.show()
        
        plt.figure(figsize=(constants.FIG_SIZE,constants.FIG_SIZE))
        plt.axis("off")
        plt.title("Training - Topdown Images")
        plt.imshow(np.transpose(vutils.make_grid(gta_batch.to(device)[:constants.batch_size], nrow = 8, padding=2, normalize=True).cpu(),(1,2,0)))
        plt.show()
    
    print("Starting Training Loop...")
    for epoch in range(start_epoch, constants.num_epochs):
        # For each batch in the dataloader
        for i, (name, vemon_batch, gta_batch) in enumerate(dataloader, 0):
            vemon_tensor = vemon_batch.to(device)
            gta_tensor = gta_batch.to(device)
            gt.train(vemon_tensor, gta_tensor, i)
        
        name_batch, vemon_batch, gta_batch = next(iter(dataloader))
        gt.verify(vemon_batch.to(device), gta_batch.to(device)) #produce image from first batch
        gt.report(epoch)
        
        #save every X epoch
        gt.save_states(epoch, constants.STYLE_CHECKPATH, constants.GENERATOR_KEY, constants.DISCRIMINATOR_KEY, constants.OPTIMIZER_KEY)

#FIX for broken pipe num_workers issue.
if __name__=="__main__": 
    main(sys.argv)

