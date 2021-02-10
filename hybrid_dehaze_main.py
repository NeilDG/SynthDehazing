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
from utils import tensor_utils
import numpy as np
import matplotlib.pyplot as plt
from loaders import dataset_loader
from trainers import dehaze_trainer
from trainers import correction_trainer
import constants
     
parser = OptionParser()
parser.add_option('--coare', type=int, help="Is running on COARE?", default=0)
parser.add_option('--img_to_load', type=int, help="Image to load?", default=-1)
parser.add_option('--load_previous', type=int, help="Load previous?", default=0)
parser.add_option('--iteration', type=int, help="Style version?", default="1")
parser.add_option('--adv_weight', type=float, help="Weight", default="1.0")
parser.add_option('--clarity_weight', type=float, help="Weight", default="500.0")
parser.add_option('--color_weight', type=float, help="Weight", default="500.0")
parser.add_option('--cycle_weight', type=float, help="Weight", default="100.0")
parser.add_option('--gen_blocks', type=int, help="Weight", default="5")
parser.add_option('--g_lr', type=float, help="LR", default="0.0005")
parser.add_option('--d_lr', type=float, help="LR", default="0.0002")

#--img_to_load=-1 --load_previous=0
#Update config if on COARE
def update_config(opts):
    constants.is_coare = opts.coare
    
    if(constants.is_coare == 1):
        print("Using COARE configuration.")
        constants.batch_size = 1024
        
        constants.ITERATION = str(opts.iteration)
        constants.DEHAZER_CHECKPATH = 'checkpoint/' + constants.DEHAZER_VERSION + "_" + constants.ITERATION +'.pt'
        constants.COLORIZER_CHECKPATH = 'checkpoint/' + constants.COLORIZER_VERSION + "_" + constants.ITERATION +'.pt'
        
        constants.DATASET_NOISY_GTA_PATH = "/scratch1/scratch2/neil.delgallego/Noisy GTA/noisy/"
        constants.DATASET_CLEAN_GTA_PATH = "/scratch1/scratch2/neil.delgallego/Noisy GTA/clean/"
        constants.DATASET_VEMON_PATH_COMPLETE = "/scratch1/scratch2/neil.delgallego/VEMON Dataset/frames/"
        
        constants.DATASET_HAZY_PATH_COMPLETE = "/scratch1/scratch2/neil.delgallego/Synth Hazy/hazy/"
        constants.DATASET_CLEAN_PATH_COMPLETE = "/scratch1/scratch2/neil.delgallego/Synth Hazy/clean/"

        constants.DATASET_HAZY_PATH_PATCH = "/scratch1/scratch2/neil.delgallego/Synth Hazy - Patch/hazy/"
        constants.DATASET_CLEAN_PATH_PATCH = "/scratch1/scratch2/neil.delgallego/Synth Hazy - Patch/clean/"
        constants.DATASET_VEMON_PATH_PATCH_32 = "/scratch1/scratch2/neil.delgallego/VEMON Dataset/B/"

        constants.num_workers = 4

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
    print("=========BEGIN============")
    print("Is Coare? %d Has GPU available? %d Count: %d" % (constants.is_coare, torch.cuda.is_available(), torch.cuda.device_count()))
    print("Torch CUDA version: %s" % torch.version.cuda)
    
    manualSeed = random.randint(1, 10000) # use if you want new results
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    print("Device: %s" % device)
    
    dehazer = dehaze_trainer.DehazeTrainer(constants.DEHAZER_VERSION, constants.ITERATION, device, opts.gen_blocks, opts.g_lr, opts.d_lr)
    dehazer.update_penalties(opts.clarity_weight, opts.adv_weight)
    
    colorizer = correction_trainer.CorrectionTrainer(constants.COLORIZER_VERSION, constants.ITERATION, device, opts.g_lr, opts.d_lr)
    colorizer.update_penalties(opts.color_weight, opts.cycle_weight, opts.adv_weight)

    start_epoch = 0
    iteration = 0

    #Temp code loading old dehazer
    #dehaze_checkpoint = torch.load(constants.DEHAZER_CHECKPATH)
    #dehazer.load_saved_state(iteration, dehaze_checkpoint, constants.GENERATOR_KEY, constants.DISCRIMINATOR_KEY, constants.OPTIMIZER_KEY)

    if(opts.load_previous): 
        dehaze_checkpoint = torch.load(constants.DEHAZER_CHECKPATH)
        color_checkpoint = torch.load(constants.COLORIZER_CHECKPATH)
        start_epoch = dehaze_checkpoint['epoch'] + 1   
        iteration = dehaze_checkpoint['iteration'] + 1
        dehazer.load_saved_state(iteration, dehaze_checkpoint, constants.GENERATOR_KEY, constants.DISCRIMINATOR_KEY, constants.OPTIMIZER_KEY)
        colorizer.load_saved_state(iteration, color_checkpoint, constants.GENERATOR_KEY, constants.DISCRIMINATOR_KEY, constants.OPTIMIZER_KEY)
        
        print("Loaded checkpt: %s %s Current epoch: %d" % (constants.DEHAZER_CHECKPATH, constants.COLORIZER_CHECKPATH, start_epoch))
        print("===================================================")
    
    # Create the dataloader
    synth_train_loader = dataset_loader.load_dark_channel_dataset(constants.DATASET_HAZY_PATH_PATCH, constants.DATASET_CLEAN_PATH_PATCH, constants.batch_size, opts.img_to_load)
    synth_test_loader = dataset_loader.load_dark_channel_test_dataset(constants.DATASET_HAZY_PATH_COMPLETE, constants.DATASET_CLEAN_PATH_COMPLETE, 4, 500)
    rgb_train_loader = dataset_loader.load_rgb_dataset(constants.DATASET_OHAZE_PATH_PATCH_CLEAN, constants.batch_size, opts.img_to_load)
    rgb_test_loader = dataset_loader.load_color_test_dataset(constants.DATASET_HAZY_TEST_PATH_2, 4, 500)
    index = 0
    
    # Plot some training images
    if(constants.is_coare == 0):
        _, synth_noisy_batch, synth_clean_batch = next(iter(synth_train_loader))
        _, y_batch, yuv_batch = next(iter(rgb_test_loader))
        
        show_images(synth_noisy_batch, "Training - Hazy Images")
        show_images(synth_clean_batch, "Training - Clean Images")
        show_images(y_batch, "Training - Gray Images")
        show_images(tensor_utils.yuv_to_rgb(yuv_batch), "Test - Real Images")
    
    print("Starting Training Loop...")
    if(constants.is_coare == 0):
        for epoch in range(start_epoch, constants.num_epochs):
            # For each batch in the dataloader
            for i, (dehaze_data, colored_data) in enumerate(zip(synth_train_loader, rgb_train_loader)):
                _, hazy_batch, clean_batch = dehaze_data
                _, y_batch, yuv_batch = colored_data
                hazy_tensor = hazy_batch.to(device)
                clean_tensor = clean_batch.to(device)
                gray_tensor = y_batch.to(device)
                colored_tensor = yuv_batch.to(device)
                
                #train dehazing
                dehazer.train(hazy_tensor, clean_tensor)
                #train colorization
                colorizer.train(dehazer.infer_single(gray_tensor), colored_tensor)
                
                if(i % 500 == 0):
                    _, synth_dark_dirty_batch, synth_dark_clean_batch = next(iter(synth_test_loader))
                    _, y_batch, yuv_batch = next(iter(rgb_test_loader))
                    
                    synth_dark_dirty_batch = synth_dark_dirty_batch.to(device)
                    synth_dark_clean_batch = synth_dark_clean_batch.to(device)
                    y_batch = y_batch.to(device)
                    yuv_batch = yuv_batch.to(device)

                    dehazer.visdom_report(iteration, synth_dark_dirty_batch, synth_dark_clean_batch, y_batch)
                    colorizer.visdom_report(iteration, dehazer.infer_single(y_batch), yuv_batch)
                    iteration = iteration + 1
                    
                    index = (index + 1) % len(synth_test_loader)
                    if(index == 0):
                      rgb_test_loader = dataset_loader.load_color_test_dataset(constants.DATASET_VEMON_PATH_COMPLETE, constants.display_size, 500)
                      synth_test_loader = dataset_loader.load_dark_channel_test_dataset(constants.DATASET_HAZY_PATH_COMPLETE, constants.DATASET_CLEAN_PATH_COMPLETE, constants.batch_size, 500)
                if(i % 1500 == 0):
                    dehazer.save_states(epoch, iteration, constants.DEHAZER_CHECKPATH, constants.GENERATOR_KEY, constants.DISCRIMINATOR_KEY, constants.OPTIMIZER_KEY)
                    colorizer.save_states(epoch, iteration, constants.COLORIZER_CHECKPATH, constants.GENERATOR_KEY, constants.DISCRIMINATOR_KEY, constants.OPTIMIZER_KEY)
    else:
        for i, (dehaze_data, colored_data) in enumerate(zip(synth_train_loader, rgb_train_loader)):
                _, hazy_batch, clean_batch = dehaze_data
                _, y_batch, yuv_batch = colored_data
                hazy_tensor = hazy_batch.to(device)
                clean_tensor = clean_batch.to(device)
                gray_tensor = y_batch.to(device)
                rgb_tensor = yuv_batch.to(device)
                
                #train dehazing
                dehazer.train(hazy_tensor, clean_tensor)
                #train colorization
                colorizer.train(dehazer.infer_single(gray_tensor), rgb_tensor)
        
        #save every X epoch
        dehazer.save_states(start_epoch, iteration, constants.DEHAZER_CHECKPATH, constants.GENERATOR_KEY, constants.DISCRIMINATOR_KEY, constants.OPTIMIZER_KEY)
        colorizer.save_states(start_epoch, iteration, constants.COLORIZER_CHECKPATH, constants.GENERATOR_KEY, constants.DISCRIMINATOR_KEY, constants.OPTIMIZER_KEY)

#FIX for broken pipe num_workers issue.
if __name__=="__main__": 
    main(sys.argv)

