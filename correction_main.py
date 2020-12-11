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
from trainers import correction_trainer
import constants
     
parser = OptionParser()
parser.add_option('--coare', type=int, help="Is running on COARE?", default=0)
parser.add_option('--img_to_load', type=int, help="Image to load?", default=-1)
parser.add_option('--load_previous', type=int, help="Load previous?", default=0)
parser.add_option('--iteration', type=int, help="Style version?", default="1")
parser.add_option('--adv_weight', type=float, help="Weight", default="1.0")
parser.add_option('--color_weight', type=float, help="Weight", default="50.0")
parser.add_option('--g_lr', type=float, help="LR", default="0.0002")
parser.add_option('--d_lr', type=float, help="LR", default="0.0002")

#--img_to_load=-1 --load_previous=0
#Update config if on COARE
def update_config(opts):
    constants.is_coare = opts.coare
    
    if(constants.is_coare == 1):
        print("Using COARE configuration.")
        constants.batch_size = 512
        
        constants.ITERATION = str(opts.iteration)
        constants.DEHAZER_CHECKPATH = 'checkpoint/' + constants.DEHAZER_VERSION + "_" + constants.ITERATION +'.pt'
        constants.COLORIZER_CHECKPATH = 'checkpoint/' + constants.COLORIZER_VERSION + "_" + constants.ITERATION +'.pt'
        
        constants.DATASET_DIV2K_PATH_PATCH ="/scratch1/scratch2/neil.delgallego/"
        constants.DATASET_VEMON_PATH_COMPLETE = "/scratch1/scratch2/neil.delgallego/VEMON Dataset/frames/"

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
    
    colorizer = correction_trainer.CorrectionTrainer(constants.COLORIZER_VERSION, constants.ITERATION, device, opts.g_lr, opts.d_lr)
    colorizer.update_penalties(opts.color_weight, opts.adv_weight)

    start_epoch = 0
    iteration = 0

    if(opts.load_previous):
        color_checkpoint = torch.load(constants.COLORIZER_CHECKPATH)
        start_epoch = color_checkpoint['epoch'] + 1
        iteration = color_checkpoint['iteration'] + 1
        colorizer.load_saved_state(iteration, color_checkpoint, constants.GENERATOR_KEY, constants.DISCRIMINATOR_KEY, constants.OPTIMIZER_KEY)
        
        print("Loaded checkpt: %s %s Current epoch: %d" % (constants.DEHAZER_CHECKPATH, constants.COLORIZER_CHECKPATH, start_epoch))
        print("===================================================")
    
    # Create the dataloader
    rgb_train_loader = dataset_loader.load_rgb_dataset(constants.DATASET_DIV2K_PATH_PATCH, constants.batch_size, opts.img_to_load)
    rgb_test_loader_1 = dataset_loader.load_color_test_dataset(constants.DATASET_DIV2K_PATH, 4, 500)
    rgb_test_loader_2 = dataset_loader.load_color_test_dataset(constants.DATASET_OHAZE_HAZY_PATH_COMPLETE, 4, 500)
    rgb_test_loader_3 = dataset_loader.load_color_test_dataset(constants.DATASET_VEMON_PATH_COMPLETE, 4, 500)
    index = 0
    
    # Plot some training images
    if(constants.is_coare == 0):
        _, gray_batch_a, colored_batch_a = next(iter(rgb_test_loader_1))
        show_images(gray_batch_a, "Test - Gray Images A")
        show_images(tensor_utils.yuv_to_rgb(colored_batch_a), "Test - Colored Images A")

    print("Starting Training Loop...")
    if(constants.is_coare == 0):
        for epoch in range(start_epoch, constants.num_epochs):
            # For each batch in the dataloader
            for i, (_, gray_batch_a, colored_batch_a) in enumerate(rgb_train_loader):
                gray_tensor = gray_batch_a.to(device)
                colored_tensor_a = colored_batch_a.to(device)

                #train colorization
                colorizer.train(gray_tensor, colored_tensor_a, colored_tensor_a)

                if(i % 500 == 0):
                    _, gray_batch_a, colored_batch_a = next(iter(rgb_test_loader_1))
                    _, gray_batch_b, colored_batch_b = next(iter(rgb_test_loader_2))
                    _, gray_batch_c, colored_batch_c = next(iter(rgb_test_loader_3))
                    gray_batch_a = gray_batch_a.to(device)
                    colored_batch_a = colored_batch_a.to(device)
                    gray_batch_b = gray_batch_b.to(device)
                    colored_batch_b = colored_batch_b.to(device)
                    gray_batch_c = gray_batch_c.to(device)
                    colored_batch_c = colored_batch_c.to(device)

                    colorizer.visdom_report_train(gray_tensor, colored_tensor_a)
                    colorizer.visdom_report(iteration, gray_batch_a, colored_batch_a, 1)
                    colorizer.visdom_report(iteration, gray_batch_b, colored_batch_b, 2)
                    colorizer.visdom_report(iteration, gray_batch_c, colored_batch_c, 3)
                    iteration = iteration + 1

                    index = (index + 1) % len(rgb_test_loader_1)
                    if(index == 0):
                      rgb_test_loader_1 = dataset_loader.load_color_test_dataset(constants.DATASET_VEMON_PATH_COMPLETE, 4, 500)
                      rgb_test_loader_2 = dataset_loader.load_color_test_dataset(constants.DATASET_OHAZE_HAZY_PATH_COMPLETE, 4, 500)
                      rgb_test_loader_3 = dataset_loader.load_color_test_dataset(constants.DATASET_VEMON_PATH_COMPLETE, 4, 500)

            colorizer.save_states(epoch, iteration, constants.COLORIZER_CHECKPATH, constants.GENERATOR_KEY, constants.DISCRIMINATOR_KEY, constants.OPTIMIZER_KEY)
    else:
        for i, (_, gray_batch_a, colored_batch_a, colored_batch_b) in enumerate(rgb_train_loader):
            gray_tensor = gray_batch_a.to(device)
            colored_tensor_a = colored_batch_a.to(device)
            colored_tensor_b = colored_batch_b.to(device)

            # train colorization
            colorizer.train(gray_tensor, colored_tensor_a, colored_tensor_b)

        #save every X epoch
        colorizer.save_states(start_epoch, iteration, constants.COLORIZER_CHECKPATH, constants.GENERATOR_KEY, constants.DISCRIMINATOR_KEY, constants.OPTIMIZER_KEY)

#FIX for broken pipe num_workers issue.
if __name__=="__main__": 
    main(sys.argv)

