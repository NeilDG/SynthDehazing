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
from trainers import multistyle_net_trainer
import constants
from utils import logger
     
parser = OptionParser()
parser.add_option('--coare', type=int, help="Is running on COARE?", default=0)
parser.add_option('--img_to_load', type=int, help="Image to load?", default=-1)
parser.add_option('--load_previous', type=int, help="Load previous?", default=0)
parser.add_option('--style_iteration', type=int, help="Style version?", default="1")
parser.add_option('--content_weight', type=float, help="Content weight", default="1.0")
parser.add_option('--style_weight', type=float, help="Content weight", default="5.0")
print = logger.log

#Update config if on COARE
def update_config(opts):
    constants.is_coare = opts.coare
    
    if(constants.is_coare == 1):
        print("Using COARE configuration.")
        constants.batch_size = constants.batch_size * 4
        
        constants.STYLE_ITERATION = opts.style_iteration
        constants.DATASET_BIRD_NORMAL_PATH = "/scratch1/scratch2/neil.delgallego/VEMON Dataset/synth_gta/"
        constants.DATASET_BIRD_HOMOG_PATH = "/scratch1/scratch2/neil.delgallego/VEMON Dataset/pending/homog_frames/"
        constants.DATASET_BIRD_GROUND_TRUTH_PATH = "/scratch1/scratch2/neil.delgallego/VEMON Dataset/pending/topdown_frames/"
        
        constants.DATASET_VEMON_FRONT_PATH = "/scratch1/scratch2/neil.delgallego/VEMON Dataset/frames/"
        constants.DATASET_VEMON_HOMOG_PATH = "/scratch1/scratch2/neil.delgallego/VEMON Dataset/homog_frames/"
        
        constants.num_workers = 2

def main(argv):
    (opts, args) = parser.parse_args(argv)
    logger.clear_log()
    print("=========BEGIN============")
    print("Is Coare? %d Has GPU available? %d Count: %d" % (opts.coare, torch.cuda.is_available(), torch.cuda.device_count()))
    print("Torch CUDA version: %s" % torch.version.cuda)
    update_config(opts)
    
    manualSeed = random.randint(1, 10000) # use if you want new results
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    print("Device: %s" % device)
    writer = SummaryWriter('train_plot')
    
    gt = multistyle_net_trainer.MultiStyleTrainer(constants.STYLE_GAN_VERSION, constants.STYLE_ITERATION, device, writer)
    gt.update_weight(opts.content_weight, opts.style_weight)
    start_epoch = 0
    
    if(opts.load_previous == 1): 
        checkpoint = torch.load(constants.STYLE_CHECKPATH)
        start_epoch = checkpoint['epoch'] + 1          
        gt.load_saved_state(checkpoint, constants.GENERATOR_KEY, constants.OPTIMIZER_KEY)
 
        print("Loaded checkpt: %s Current epoch: %d" % (constants.STYLE_CHECKPATH, start_epoch))
        print("===================================================")
    
    # Create the dataloader
    dataloader = dataset_loader.load_msg_dataset(constants.batch_size, opts.img_to_load)
    
    print("Starting Training Loop...")
    # For each batch in the dataloader
    for i, (name, vemon_batch, gta_batch) in enumerate(dataloader, 0):
        vemon_tensor = vemon_batch.to(device)
        gta_tensor = gta_batch.to(device)
        gt.train(vemon_tensor, gta_tensor, i)
        
    #save every X epoch
    gt.save_states(start_epoch, constants.STYLE_CHECKPATH, constants.GENERATOR_KEY, constants.OPTIMIZER_KEY)

#FIX for broken pipe num_workers issue.
if __name__=="__main__": 
    main(sys.argv)

