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
parser.add_option('--style_iteration', type=int, help="Style version?", default="1")
parser.add_option('--identity_weight', type=float, help="Weight", default="1.0")
parser.add_option('--cycle_weight', type=float, help="Weight", default="10.0")
parser.add_option('--adv_weight', type=float, help="Weight", default="1.0")
parser.add_option('--tv_weight', type=float, help="Weight", default="1.0")
parser.add_option('--gen_blocks', type=int, help="Weight", default="4")
parser.add_option('--disc_blocks', type=int, help="Weight", default="6")
parser.add_option('--gen_skips', type=int, default="1")
parser.add_option('--disc_skips', type=int, default="1")
print = logger.log

#--img_to_load=72000 --gen_skips=1 --disc_skips=1 --cycle_weight=10.0 --identity_weight=1.0 --tv_weight=3.0 --adv_weight=50.0 --load_previous=0
#Update config if on COARE
def update_config(opts):
    constants.is_coare = opts.coare
    
    if(constants.is_coare == 1):
        print("Using COARE configuration.")
        constants.batch_size = 128
        
        constants.STYLE_ITERATION = str(opts.style_iteration)
        constants.STYLE_CHECKPATH = 'checkpoint/' + constants.STYLE_GAN_VERSION + "_" + constants.STYLE_ITERATION +'.pt'
        
        constants.DATASET_GTA_PATH = "/scratch1/scratch2/neil.delgallego/VEMON Dataset/pending/frames/"
        constants.DATASET_PLACES_PATH = "/scratch1/scratch2/neil.delgallego/Places Dataset/"
        constants.DATASET_VEMON_PATH = "/scratch1/scratch2/neil.delgallego/VEMON Dataset/frames/"
        
        constants.num_workers = 0
        
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
    
    gt = denoise_net_trainer.GANTrainer(constants.STYLE_GAN_VERSION, constants.STYLE_ITERATION, device, opts.gen_blocks, opts.disc_blocks)
    gt.update_penalties(opts.identity_weight, opts.cycle_weight, opts.adv_weight, opts.tv_weight, opts.gen_skips, opts.disc_skips)
    start_epoch = 0
    
    if(opts.load_previous): 
        checkpoint = torch.load(constants.STYLE_CHECKPATH)
        start_epoch = checkpoint['epoch'] + 1   
        iteration = checkpoint['iteration'] + 1
        gt.load_saved_state(iteration, checkpoint, constants.GENERATOR_KEY, constants.DISCRIMINATOR_KEY, constants.OPTIMIZER_KEY)
 
        print("Loaded checkpt: %s Current epoch: %d" % (constants.STYLE_CHECKPATH, start_epoch))
        print("===================================================")
    
    # Create the dataloader
    dataloader = dataset_loader.load_msg_dataset(constants.batch_size, opts.img_to_load)
    
    # Plot some training images
    if(constants.is_coare == 0):
        name_batch, vemon_batch_orig, gta_batch_orig = next(iter(dataloader))
        plt.figure(figsize=(constants.FIG_SIZE,constants.FIG_SIZE))
        plt.axis("off")
        plt.title("Training - Normal Images")
        plt.imshow(np.transpose(vutils.make_grid(vemon_batch_orig.to(device)[:constants.batch_size], nrow = 8, padding=2, normalize=True).cpu(),(1,2,0)))
        plt.show()
        
        plt.figure(figsize=(constants.FIG_SIZE,constants.FIG_SIZE))
        plt.axis("off")
        plt.title("Training - Topdown Images")
        plt.imshow(np.transpose(vutils.make_grid(gta_batch_orig.to(device)[:constants.batch_size], nrow = 8, padding=2, normalize=True).cpu(),(1,2,0)))
        plt.show()
    
        vemon_orig_tensor = vemon_batch_orig.to(device)
        gta_orig_tensor = gta_batch_orig.to(device)
    
    print("Starting Training Loop...")
    for epoch in range(start_epoch, constants.num_epochs):
        # For each batch in the dataloader
        for i, (name, vemon_batch, gta_batch) in enumerate(dataloader, 0):
            vemon_tensor = vemon_batch.to(device)
            gta_tensor = gta_batch.to(device)
            gt.train(vemon_tensor, gta_tensor)
            gt.visdom_report(vemon_orig_tensor, gta_orig_tensor) #use same batch for visualization and debugging
            
        
        #save every X epoch
        gt.save_states(epoch, constants.STYLE_CHECKPATH, constants.GENERATOR_KEY, constants.DISCRIMINATOR_KEY, constants.OPTIMIZER_KEY)

#FIX for broken pipe num_workers issue.
if __name__=="__main__": 
    main(sys.argv)

