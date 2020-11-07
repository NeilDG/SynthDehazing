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
from loaders import dataset_loader
from trainers import denoise_net_trainer
import constants
from utils import logger
     
parser = OptionParser()
parser.add_option('--coare', type=int, help="Is running on COARE?", default=0)
parser.add_option('--img_to_load', type=int, help="Image to load?", default=-1)
parser.add_option('--load_previous', type=int, help="Load previous?", default=0)
parser.add_option('--iteration', type=int, help="Style version?", default="1")
parser.add_option('--identity_weight', type=float, help="Weight", default="1.0")
parser.add_option('--adv_weight', type=float, help="Weight", default="1.0")
parser.add_option('--likeness_weight', type=float, help="Weight", default="1.0")
parser.add_option('--gen_blocks', type=int, help="Weight", default="3")
#parser.add_option('--disc_blocks', type=int, help="Weight", default="3")
print = logger.log

#--img_to_load=72296 --identity_weight=10.0 --likeness_weight=5.0 --adv_weight=1.0 --load_previous=0
#Update config if on COARE
def update_config(opts):
    constants.is_coare = opts.coare
    
    if(constants.is_coare == 1):
        print("Using COARE configuration.")
        constants.batch_size = 512
        
        constants.ITERATION = str(opts.iteration)
        constants.CHECKPATH = 'checkpoint/' + constants.VERSION + "_" + constants.ITERATION +'.pt'
        
        constants.DATASET_NOISY_GTA_PATH = "/scratch1/scratch2/neil.delgallego/Noisy GTA/noisy/"
        constants.DATASET_CLEAN_GTA_PATH = "/scratch1/scratch2/neil.delgallego/Noisy GTA/clean/"
        constants.DATASET_VEMON_PATH_COMPLETE = "/scratch1/scratch2/neil.delgallego/VEMON Dataset/frames/"
        
        constants.num_workers = 4
        
def main(argv):
    (opts, args) = parser.parse_args(argv)
    logger.clear_log()
    update_config(opts)
    print("=========BEGIN============")
    print("Is Coare? %d Has GPU available? %d Count: %d" % (constants.is_coare, torch.cuda.is_available(), torch.cuda.device_count()))
    print("Torch CUDA version: %s" % torch.version.cuda)
    
    manualSeed = random.randint(1, 10000) # use if you want new results
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    print("Device: %s" % device)
    
    gt = denoise_net_trainer.DenoiseTrainer(constants.VERSION, constants.ITERATION, device, opts.gen_blocks)
    gt.update_penalties(opts.adv_weight, opts.identity_weight, opts.likeness_weight)
    start_epoch = 0
    iteration = 0
    
    if(opts.load_previous): 
        checkpoint = torch.load(constants.CHECKPATH)
        start_epoch = checkpoint['epoch'] + 1   
        iteration = checkpoint['iteration'] + 1
        gt.load_saved_state(iteration, checkpoint, constants.GENERATOR_KEY, constants.DISCRIMINATOR_KEY, constants.OPTIMIZER_KEY)
 
        print("Loaded checkpt: %s Current epoch: %d" % (constants.CHECKPATH, start_epoch))
        print("===================================================")
    
    # Create the dataloader
    train_loader = dataset_loader.load_train_dataset(constants.DATASET_NOISY_GTA_PATH, constants.DATASET_CLEAN_GTA_PATH, constants.batch_size, opts.img_to_load)
    test_loader = dataset_loader.load_test_dataset(constants.DATASET_NOISY_GTA_PATH, constants.DATASET_CLEAN_GTA_PATH, constants.display_size, 500)
    index = 0
    
    # Plot some training images
    if(constants.is_coare == 0):
        _, noisy_batch, clean_batch = next(iter(train_loader))
        
        plt.figure(figsize=constants.FIG_SIZE)
        plt.axis("off")
        plt.title("Training - Dirty Images")
        plt.imshow(np.transpose(vutils.make_grid(noisy_batch.to(device)[:constants.batch_size], nrow = 8, padding=2, normalize=True).cpu(),(1,2,0)))
        plt.show()
        
        plt.figure(figsize=constants.FIG_SIZE)
        plt.axis("off")
        plt.title("Training - Clean Images")
        plt.imshow(np.transpose(vutils.make_grid(clean_batch.to(device)[:constants.batch_size], nrow = 8, padding=2, normalize=True).cpu(),(1,2,0)))
        plt.show()
    
    print("Starting Training Loop...")
    if(constants.is_coare == 0):
        for epoch in range(start_epoch, constants.num_epochs):
            # For each batch in the dataloader
            for i, (name, dirty_batch, clean_batch) in enumerate(train_loader, 0):
                dirty_tensor = dirty_batch.to(device)
                clean_tensor = clean_batch.to(device)
                gt.train(dirty_tensor, clean_tensor)
                if(i % 70 == 0):
                    view_batch, view_dirty_batch, view_clean_batch = next(iter(test_loader))
                    view_dirty_batch = view_dirty_batch.to(device)
                    view_clean_batch = view_clean_batch.to(device)
                    gt.visdom_report(iteration, dirty_tensor, clean_tensor, view_dirty_batch, view_clean_batch)
                    iteration = iteration + 1
                    index = (index + 1) % len(test_loader)
                    if(index == 0):
                      test_loader = dataset_loader.load_test_dataset(constants.DATASET_NOISY_GTA_PATH, constants.DATASET_CLEAN_GTA_PATH, constants.batch_size, 500)
              
            gt.save_states(epoch, iteration, constants.CHECKPATH, constants.GENERATOR_KEY, constants.DISCRIMINATOR_KEY, constants.OPTIMIZER_KEY)
    else:
        for i, (name, dirty_batch, clean_batch) in enumerate(train_loader, 0):
                dirty_tensor = dirty_batch.to(device)
                clean_tensor = clean_batch.to(device)
                gt.train(dirty_tensor, clean_tensor)
                
        
        #save every X epoch
        gt.save_states(start_epoch, iteration, constants.CHECKPATH, constants.GENERATOR_KEY, constants.DISCRIMINATOR_KEY, constants.OPTIMIZER_KEY)

#FIX for broken pipe num_workers issue.
if __name__=="__main__": 
    main(sys.argv)

