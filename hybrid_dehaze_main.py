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
from trainers import dehaze_trainer
import constants
from utils import logger
     
parser = OptionParser()
parser.add_option('--coare', type=int, help="Is running on COARE?", default=0)
parser.add_option('--img_to_load', type=int, help="Image to load?", default=-1)
parser.add_option('--load_previous', type=int, help="Load previous?", default=0)
parser.add_option('--iteration', type=int, help="Style version?", default="1")
parser.add_option('--clarity_weight', type=float, help="Weight", default="100.0")
parser.add_option('--adv_weight', type=float, help="Weight", default="1.0")
parser.add_option('--cycle_weight', type=float, help="Weight", default="100.0")
parser.add_option('--gen_blocks', type=int, help="Weight", default="5")
#parser.add_option('--disc_blocks', type=int, help="Weight", default="3")
#print = logger.log

#--img_to_load=-1 --load_previous=0
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
        constants.DATASET_VEMON_PATH = "/scratch1/scratch2/neil.delgallego/VEMON Dataset/frames/"
        
        constants.DATASET_HAZY_PATH = "/scratch1/scratch2/neil.delgallego/Synth Hazy/hazy/"
        constants.DATASET_CLEAN_PATH = "/scratch1/scratch2/neil.delgallego/Synth Hazy/clean/"
        
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
    
    gt = dehaze_trainer.DehazeTrainer(constants.VERSION, constants.ITERATION, device, opts.gen_blocks)
    gt.update_penalties(opts.adv_weight, opts.clarity_weight, opts.cycle_weight)
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
    synth_train_loader = dataset_loader.load_noise_dataset(constants.DATASET_HAZY_PATH, constants.DATASET_CLEAN_PATH, constants.batch_size, opts.img_to_load)
    #real_train_loader = dataset_loader.load_noise_dataset(constants.DATASET_VEMON_PATH, constants.DATASET_CLEAN_PATH, constants.batch_size, opts.img_to_load)
    
    #synth_test_loader = dataset_loader.load_test_dataset(constants.DATASET_HAZY_PATH, constants.DATASET_CLEAN_PATH, constants.display_size, 500)
    real_test_loader = dataset_loader.load_test_dataset(constants.DATASET_VEMON_PATH, constants.DATASET_CLEAN_PATH, constants.display_size, 500)
    
    synth_dark_test_loader = dataset_loader.load_dark_channel_test_dataset(constants.DATASET_HAZY_PATH, constants.DATASET_CLEAN_PATH, constants.display_size, 500)
    real_dark_test_loader = dataset_loader.load_dark_channel_test_dataset(constants.DATASET_VEMON_PATH, constants.DATASET_CLEAN_PATH, constants.display_size, 500)
    
    index = 0
    
    # Plot some training images
    if(constants.is_coare == 0):
        _, synth_noisy_batch, synth_clean_batch = next(iter(synth_train_loader))
        #_, real_noisy_batch , real_clean_batch = next(iter(real_train_loader))
        
        plt.figure(figsize=constants.FIG_SIZE)
        plt.axis("off")
        plt.title("Training - Dirty Images")
        plt.imshow(np.transpose(vutils.make_grid(synth_noisy_batch.to(device)[:constants.batch_size], nrow = 8, padding=2, normalize=True).cpu(),(1,2,0)))
        plt.show()
        
        plt.figure(figsize=constants.FIG_SIZE)
        plt.axis("off")
        plt.title("Training - Clean Images")
        plt.imshow(np.transpose(vutils.make_grid(synth_clean_batch.to(device)[:constants.batch_size], nrow = 8, padding=2, normalize=True).cpu(),(1,2,0)))
        plt.show()
    
    print("Starting Training Loop...")
    if(constants.is_coare == 0):
        for epoch in range(start_epoch, constants.num_epochs):
            # For each batch in the dataloader
            for i, (synth_data) in enumerate(synth_train_loader):
                _, synth_dirty_batch, synth_clean_batch = synth_data
                #_, real_dirty_batch, real_clean_batch = real_data
                 
                probability_real = 0.0
                chance = random.random()
                synth_dirty_tensor = synth_dirty_batch.to(device)
                synth_clean_tensor = synth_clean_batch.to(device)
                
                #real_dirty_tensor = real_dirty_batch.to(device)
                #real_clean_tensor = real_clean_batch.to(device)
                
                if(chance >= probability_real): 
                    gt.train(synth_dirty_tensor, synth_clean_tensor)
                else:
                    print("Training real")
                    #gt.train(real_dirty_tensor, real_clean_tensor, False)
                
                if(i % 200 == 0):
                    _, synth_dark_dirty_batch, synth_dark_clean_batch = next(iter(synth_dark_test_loader))
                    _, real_dark_dirty_batch, real_dark_clean_batch = next(iter(real_dark_test_loader))
                    _, real_rgb_dirty_batch, _ = next(iter(real_test_loader))
                    
                    synth_dark_dirty_batch = synth_dark_dirty_batch.to(device)
                    synth_dark_clean_batch = synth_dark_clean_batch.to(device)
                    real_dark_dirty_batch = real_dark_dirty_batch.to(device)
                    real_rgb_dirty_batch = real_rgb_dirty_batch.to(device)
                    
                    gt.visdom_report(iteration, synth_dark_dirty_batch, synth_dark_clean_batch, real_dark_dirty_batch, real_rgb_dirty_batch)
                    iteration = iteration + 1
                    
                    index = (index + 1) % len(synth_dark_test_loader)
                    if(index == 0):
                      synth_dark_test_loader = dataset_loader.load_dark_channel_test_dataset(constants.DATASET_HAZY_PATH, constants.DATASET_CLEAN_PATH, constants.display_size, 500)
                      real_dark_test_loader = dataset_loader.load_dark_channel_test_dataset(constants.DATASET_VEMON_PATH, constants.DATASET_CLEAN_PATH, constants.display_size, 500)
                      real_test_loader = dataset_loader.load_test_dataset(constants.DATASET_VEMON_PATH, constants.DATASET_CLEAN_PATH, constants.display_size, 500)
              
            gt.save_states(epoch, iteration, constants.CHECKPATH, constants.GENERATOR_KEY, constants.DISCRIMINATOR_KEY, constants.OPTIMIZER_KEY)
    else:
        for i, (_, synth_dirty_batch, synth_clean_batch) in enumerate(synth_train_loader, 0):
                
                probability_real = 0.0
                chance = random.random()
                synth_dirty_tensor = synth_dirty_batch.to(device)
                synth_clean_tensor = synth_clean_batch.to(device)
                
                if(chance > probability_real): 
                    #print("Training synth")
                    gt.train(synth_dirty_tensor, synth_clean_tensor)
                else:
                    print("Training real")
                    #gt.train(real_dirty_tensor, real_clean_tensor, False)     
        
        #save every X epoch
        gt.save_states(start_epoch, iteration, constants.CHECKPATH, constants.GENERATOR_KEY, constants.DISCRIMINATOR_KEY, constants.OPTIMIZER_KEY)

#FIX for broken pipe num_workers issue.
if __name__=="__main__": 
    main(sys.argv)

