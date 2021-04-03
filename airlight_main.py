# -*- coding: utf-8 -*-
"""
Main entry for network training for airlight
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
from trainers import airlight_trainer
from model import vanilla_cycle_gan as cycle_gan
from model import dehaze_discriminator as dh
import constants
import itertools

parser = OptionParser()
parser.add_option('--coare', type=int, help="Is running on COARE?", default=0)
parser.add_option('--img_to_load', type=int, help="Image to load?", default=-1)
parser.add_option('--load_previous', type=int, help="Load previous?", default=0)
parser.add_option('--iteration', type=int, help="Style version?", default="1")
parser.add_option('--airlight_weight', type=float, help="Weight", default="100.0")
parser.add_option('--d_lr', type=float, help="LR", default="0.00005")
parser.add_option('--batch_size', type=int, help="Weight", default="64")
parser.add_option('--image_size', type=int, help="Weight", default="256")
parser.add_option('--comments', type=str, help="comments for bookmarking", default = "Airlight estimation network using same architecture for A and B. \n "
                                                                                     "Light estimates from coord network are included in training. \n"
                                                                                     "New architecture based on CycleGAN.")

#--img_to_load=-1 --load_previous=0
# Update config if on COARE
def update_config(opts):
    constants.is_coare = opts.coare

    if (constants.is_coare == 1):
        print("Using COARE configuration.")
        constants.TEST_IMAGE_SIZE = (opts.image_size, opts.image_size)
        constants.batch_size = opts.batch_size
        constants.ITERATION = str(opts.iteration)
        constants.LIGHTCOORDS_ESTIMATOR_CHECKPATH = 'checkpoint/' + constants.LIGHTS_ESTIMATOR_VERSION + "_" + constants.ITERATION + '.pt'

        constants.DATASET_HAZY_PATH_COMPLETE = "/scratch1/scratch2/neil.delgallego/Synth Hazy 2/hazy/"
        constants.DATASET_DEPTH_PATH_COMPLETE = "/scratch1/scratch2/neil.delgallego/Synth Hazy 2/depth/"
        constants.DATASET_CLEAN_PATH_COMPLETE = "/scratch1/scratch2/neil.delgallego/Synth Hazy 2/clean/"
        constants.DATASET_CLEAN_PATH_COMPLETE_STYLED = "/scratch1/scratch2/neil.delgallego/Synth Hazy 2/clean - styled/"
        constants.DATASET_LIGHTCOORDS_PATH_COMPLETE = "/scratch1/scratch2/neil.delgallego/Synth Hazy 2/light/"
        constants.num_workers = 4

def show_images(img_tensor, caption):
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    plt.figure(figsize=(16, 4))
    plt.axis("off")
    plt.title(caption)
    plt.imshow(np.transpose(
        vutils.make_grid(img_tensor.to(device)[:constants.batch_size], nrow=8, padding=2, normalize=True).cpu(),
        (1, 2, 0)))
    plt.show()

def main(argv):
    (opts, args) = parser.parse_args(argv)
    update_config(opts)
    print("=====================BEGIN============================")
    print("Is Coare? %d Has GPU available? %d Count: %d Torch CUDA version: %s"
          % (constants.is_coare, torch.cuda.is_available(), torch.cuda.device_count(), torch.version.cuda))

    #manualSeed = random.randint(1, 10000)  # use if you want new results
    manualSeed = 1 #set this for experiments and promoting fixed results
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    print("Device: %s" % device)

    # load color transfer
    color_transfer_checkpt = torch.load('checkpoint/color_transfer_v1.11_2.pt')
    color_transfer_gan = cycle_gan.Generator(n_residual_blocks=10).to(device)
    color_transfer_gan.load_state_dict(color_transfer_checkpt[constants.GENERATOR_KEY + "A"])
    color_transfer_gan.eval()
    print("Color transfer GAN model loaded.")
    print("===================================================")

    #load light coords estimation network
    light_coords_checkpt = torch.load('checkpoint/lightcoords_estimator_V1.00_9.pt')
    light_estimator = dh.LightCoordsEstimator_V2(input_nc = 3, num_layers = 4).to(device)
    light_estimator.load_state_dict(light_coords_checkpt[constants.DISCRIMINATOR_KEY])
    light_estimator.eval()
    print("Light estimator network loaded")
    print("===================================================")

    gt = airlight_trainer.AirlightTrainer(device, opts.d_lr)
    gt.update_penalties(opts.airlight_weight, opts.comments)
    start_epoch = 0

    iteration = 0
    if (opts.load_previous):
        checkpoint = torch.load(constants.AIRLIGHT_ESTIMATOR_CHECKPATH)
        start_epoch = checkpoint['epoch'] + 1
        iteration = checkpoint['iteration'] + 1
        gt.load_saved_state(checkpoint)

        print("Loaded checkpt: %s Current epoch: %d" % (constants.AIRLIGHT_ESTIMATOR_CHECKPATH, start_epoch))
        print("===================================================")

    # Create the dataloader
    train_loader = dataset_loader.load_model_based_transmission_dataset(constants.DATASET_CLEAN_PATH_COMPLETE_STYLED, constants.DATASET_DEPTH_PATH_COMPLETE, constants.DATASET_LIGHTCOORDS_PATH_COMPLETE,
                                                                        constants.TEST_IMAGE_SIZE, constants.batch_size, opts.img_to_load)

    test_loader = dataset_loader.load_model_based_transmission_dataset_test(constants.DATASET_CLEAN_PATH_COMPLETE_STYLED_TEST, constants.DATASET_DEPTH_PATH_COMPLETE_TEST, constants.DATASET_LIGHTCOORDS_PATH_COMPLETE_TEST,
                                                                        constants.TEST_IMAGE_SIZE, constants.batch_size, 500)

    # Plot some training images
    if (constants.is_coare == 0):
        _, a, b, light_coords_tensor, _ = next(iter(train_loader))
        show_images(a, "Training - RGB Images")
        show_images(b, "Training - Depth Images")
        print("Training - Lights Tensor", np.shape(light_coords_tensor))
        print("Values: ",light_coords_tensor.numpy())

        _, a, b, _, _ = next(iter(test_loader))
        show_images(a, "Test - RGB Images")
        show_images(b, "Test - Depth Images")

    print("Starting Training Loop...")
    if (constants.is_coare == 0):
        for epoch in range(start_epoch, constants.num_epochs):
            # For each batch in the dataloader
            print("Feeding ground truth light data. Epoch: ", epoch)
            for i, (train_data, test_data) in enumerate(zip(train_loader, itertools.cycle(test_loader))):
                _, rgb_batch, _, light_batch, airlight_batch = train_data
                rgb_tensor = rgb_batch.to(device).float()
                airlight_tensor = airlight_batch.to(device).float()
                lights_coord_tensor = light_batch.to(device).float()

                # perform color transfer first
                #rgb_tensor = color_transfer_gan(rgb_tensor)
                
                gt.train(rgb_tensor, lights_coord_tensor, airlight_tensor, i)

                #check test set
                _, rgb_batch, _, light_batch, airlight_batch = test_data
                rgb_tensor_test = rgb_batch.to(device).float()
                airlight_tensor = airlight_batch.to(device).float()
                lights_coord_tensor = light_batch.to(device).float()
                
                gt.test(rgb_tensor_test, lights_coord_tensor, airlight_tensor, i)

            gt.save_states(epoch, iteration)
            gt.visdom_report(iteration, rgb_tensor, rgb_tensor_test)

            #For each batch, use pretrained light coord network to account for errors
            print("Feeding light estimator training data. Epoch: ", epoch)
            for i, (train_data, test_data) in enumerate(zip(train_loader, itertools.cycle(test_loader))):
                _, rgb_batch, _, _, airlight_batch = train_data
                rgb_tensor = rgb_batch.to(device).float()
                airlight_tensor = airlight_batch.to(device).float()

                light_estimator.eval()
                gt.train(rgb_tensor, light_estimator(rgb_tensor), airlight_tensor, i)

                # check test set
                _, rgb_batch, _, _, airlight_batch = test_data
                rgb_tensor_test = rgb_batch.to(device).float()
                airlight_tensor = airlight_batch.to(device).float()

                gt.test(rgb_tensor_test, light_estimator(rgb_tensor_test), airlight_tensor, i)

            gt.save_states(epoch, iteration)
            gt.visdom_report(iteration, rgb_tensor, rgb_tensor_test)

    else:
        for i, train_data in enumerate(train_loader, 0):
            _, rgb_batch, _, light_batch, airlight_batch = train_data
            rgb_tensor = rgb_batch.to(device).float()
            #airlight_tensor = airlight_batch.to(device).float()
            lights_coord_tensor = light_batch.to(device).float()

            # perform color transfer first
            rgb_tensor = color_transfer_gan(rgb_tensor)
            gt.train(rgb_tensor, lights_coord_tensor)
            if ((i + 1) % 400 == 0):
                print("Iterating %d" % i)

        gt.save_states(start_epoch, iteration)


# FIX for broken pipe num_workers issue.
if __name__ == "__main__":
    main(sys.argv)