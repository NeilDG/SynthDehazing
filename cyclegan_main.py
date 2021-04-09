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
from trainers import cyclegan_trainer
import constants
     
parser = OptionParser()
parser.add_option('--coare', type=int, help="Is running on COARE?", default=0)
parser.add_option('--img_to_load', type=int, help="Image to load?", default=-1)
parser.add_option('--load_previous', type=int, help="Load previous?", default=0)
parser.add_option('--iteration', type=int, help="Style version?", default="1")
parser.add_option('--identity_weight', type=float, help="Weight", default="1.0")
parser.add_option('--adv_weight', type=float, help="Weight", default="1.0")
parser.add_option('--likeness_weight', type=float, help="Weight", default="10.0")
parser.add_option('--smoothness_weight', type=float, help="Weight", default="1.0")
parser.add_option('--cycle_weight', type=float, help="Weight", default="10.0")
parser.add_option('--brightness_enhance', type=float, help="Weight", default="1.00") 
parser.add_option('--contrast_enhance', type=float, help="Weight", default="1.00")
parser.add_option('--g_lr', type=float, help="LR", default="0.0002")
parser.add_option('--d_lr', type=float, help="LR", default="0.0002")
parser.add_option('--comments', type=str, help="comments for bookmarking", default = "Vanilla CycleGAN. Paired learning for extracing albedo from a lit image and vice versa.")

#--img_to_load=-1 --load_previous=1
#Update config if on COARE
def update_config(opts):
    constants.is_coare = opts.coare
    constants.brightness_enhance = opts.brightness_enhance
    constants.contrast_enhance = opts.contrast_enhance
    
    if(constants.is_coare == 1):
        print("Using COARE configuration.")
        constants.batch_size = constants.batch_size * 2
        
        constants.ITERATION = str(opts.iteration)
        constants.COLOR_TRANSFER_CHECKPATH = 'checkpoint/' + constants.COLOR_TRANSFER_VERSION + "_" + constants.ITERATION + '.pt'

        constants.DATASET_VEMON_PATH_COMPLETE = "/scratch1/scratch2/neil.delgallego/VEMON Dataset/frames/"
        constants.DATASET_HAZY_PATH_COMPLETE = "/scratch1/scratch2/neil.delgallego/Synth Hazy - Depth/hazy/"
        constants.DATASET_CLEAN_PATH_COMPLETE = "/scratch1/scratch2/neil.delgallego/Synth Hazy - Depth/clean/"
        constants.DATASET_PLACES_PATH = "/scratch1/scratch2/neil.delgallego/Places Dataset/"

        constants.DATASET_CLEAN_PATH_COMPLETE_STYLED_3 = "/scratch1/scratch2/neil.delgallego/Synth Hazy 3/clean/"
        constants.DATASET_ALBEDO_PATH_COMPLETE_3 = "/scratch1/scratch2/neil.delgallego/Synth Hazy 3/albedo/"

def show_images(img_tensor, caption):
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    plt.figure(figsize=constants.FIG_SIZE)
    plt.axis("off")
    plt.title(caption)
    plt.imshow(np.transpose(vutils.make_grid(img_tensor.to(device)[:constants.batch_size], nrow = 8, padding=2, normalize=True).cpu(),(1,2,0)))
    plt.show()

def main(argv):
    (opts, args) = parser.parse_args(argv)
    update_config(opts)
    print("=====================BEGIN============================")
    print("Is Coare? %d Has GPU available? %d Count: %d" % (constants.is_coare, torch.cuda.is_available(), torch.cuda.device_count()))
    print("Torch CUDA version: %s" % torch.version.cuda)
    
    manualSeed = random.randint(1, 10000) # use if you want new results
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    print("Device: %s" % device)
    
    gt = cyclegan_trainer.CycleGANTrainer(device, opts.g_lr, opts.d_lr)
    gt.update_penalties(opts.adv_weight, opts.identity_weight, opts.likeness_weight, opts.cycle_weight, opts.smoothness_weight, opts.comments)
    start_epoch = 0
    iteration = 0
    
    if(opts.load_previous): 
        checkpoint = torch.load(constants.COLOR_TRANSFER_CHECKPATH)
        start_epoch = checkpoint['epoch'] + 1   
        iteration = checkpoint['iteration'] + 1
        gt.load_saved_state(checkpoint)
 
        print("Loaded checkpt: %s Current epoch: %d" % (constants.COLOR_TRANSFER_CHECKPATH, start_epoch))
        print("===================================================")
    
    # Create the dataloader
    #train_loader = dataset_loader.load_color_train_dataset(constants.DATASET_CLEAN_PATH_COMPLETE, constants.DATASET_PLACES_PATH, constants.batch_size, opts.img_to_load)
    #test_loader = dataset_loader.load_test_dataset(constants.DATASET_CLEAN_PATH_COMPLETE, constants.DATASET_PLACES_PATH, constants.display_size, 500)
    train_loader = dataset_loader.load_color_albedo_train_dataset(constants.DATASET_CLEAN_PATH_COMPLETE_STYLED_3, constants.DATASET_ALBEDO_PATH_COMPLETE_3, constants.batch_size, opts.img_to_load)
    test_loader_1 = dataset_loader.load_color_albedo_test_dataset(constants.DATASET_CLEAN_PATH_COMPLETE_STYLED_3, constants.DATASET_ALBEDO_PATH_COMPLETE_3, constants.batch_size, 500)
    test_loader_2 = dataset_loader.load_color_albedo_test_dataset(constants.DATASET_OHAZE_HAZY_PATH_COMPLETE, constants.DATASET_ALBEDO_PATH_COMPLETE_3, constants.batch_size, 500)
    index = 0
    
    # Plot some training images
    if(constants.is_coare == 0):
        _, noisy_batch, clean_batch = next(iter(train_loader))

        show_images(noisy_batch, "Training - A Images")
        show_images(clean_batch, "Training - B Images")
    
    print("Starting Training Loop...")

    if(constants.is_coare == 0):
        for epoch in range(start_epoch, constants.num_epochs):
            # For each batch in the dataloader
            for i, train_data in enumerate(train_loader, 0):
                _, dirty_batch, clean_batch = train_data
                dirty_tensor = dirty_batch.to(device)
                clean_tensor = clean_batch.to(device)

                gt.train(dirty_tensor, clean_tensor)
                if(i % 100 == 0 and constants.is_coare == 0):
                    view_batch, view_dirty_batch, view_clean_batch = next(iter(test_loader_1))
                    view_dirty_batch = view_dirty_batch.to(device)
                    view_clean_batch = view_clean_batch.to(device)
                    gt.visdom_report(iteration, dirty_tensor, clean_tensor, view_dirty_batch, view_clean_batch)

                    view_batch, view_dirty_batch, _ = next(iter(test_loader_2))
                    view_dirty_batch = view_dirty_batch.to(device)
                    gt.visdom_infer(view_dirty_batch, "O-Haze Hazy", "O-Haze Albedo")

                    iteration = iteration + 1
                    index = (index + 1) % len(test_loader_1)
                    if(index == 0):
                        test_loader_1 = dataset_loader.load_color_albedo_test_dataset(constants.DATASET_CLEAN_PATH_COMPLETE_STYLED_3, constants.DATASET_ALBEDO_PATH_COMPLETE_3, constants.batch_size, 500)
                        test_loader_2 = dataset_loader.load_color_albedo_test_dataset(constants.DATASET_OHAZE_HAZY_PATH_COMPLETE, constants.DATASET_ALBEDO_PATH_COMPLETE_3, constants.batch_size, 500)

                    gt.save_states(epoch, iteration)
    else:
        for i, (name, dirty_batch, clean_batch) in enumerate(train_loader, 0):
            dirty_tensor = dirty_batch.to(device)
            clean_tensor = clean_batch.to(device)
            gt.train(dirty_tensor, clean_tensor)
            if(i % 100 == 0):
                print("Iterating %d " % i)

        #save every X epoch
        gt.save_states(start_epoch, iteration)

#FIX for broken pipe num_workers issue.
if __name__=="__main__": 
    main(sys.argv)

