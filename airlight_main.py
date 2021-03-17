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
from trainers import lightcoords_trainer
from model import style_transfer_gan as color_gan
from model import vanilla_cycle_gan as cycle_gan
import constants
from utils import dehazing_proper

parser = OptionParser()
parser.add_option('--coare', type=int, help="Is running on COARE?", default=0)
parser.add_option('--img_to_load', type=int, help="Image to load?", default=-1)
parser.add_option('--load_previous', type=int, help="Load previous?", default=0)
parser.add_option('--iteration', type=int, help="Style version?", default="1")
parser.add_option('--airlight_weight', type=float, help="Weight", default="10.0")
parser.add_option('--d_lr', type=float, help="LR", default="0.005")
parser.add_option('--batch_size', type=int, help="Weight", default="64")
parser.add_option('--image_size', type=int, help="Weight", default="256")
parser.add_option('--comments', type=str, help="comments for bookmarking", default = "Patch-based airlight estimation network")

#--img_to_load=-1 --load_previous=0
# Update config if on COARE
def update_config(opts):
    constants.is_coare = opts.coare

    if (constants.is_coare == 1):
        print("Using COARE configuration.")
        constants.TEST_IMAGE_SIZE = (opts.image_size, opts.image_size)
        constants.batch_size = opts.batch_size

        constants.ITERATION = str(opts.iteration)
        constants.AIRLIGHT_ESTIMATOR_CHECKPATH = 'checkpoint/' + constants.AIRLIGHT_VERSION + "_" + constants.ITERATION + '.pt'

        constants.DATASET_HAZY_PATH_COMPLETE = "/scratch1/scratch2/neil.delgallego/Synth Hazy - Depth 2/hazy/"
        constants.DATASET_DEPTH_PATH_COMPLETE = "/scratch1/scratch2/neil.delgallego/Synth Hazy - Depth 2/depth/"
        constants.DATASET_CLEAN_PATH_COMPLETE = "/scratch1/scratch2/neil.delgallego/Synth Hazy - Depth 2/clean/"

        constants.DATASET_OHAZE_HAZY_PATH_COMPLETE = "/scratch1/scratch2/neil.delgallego/Synth Hazy - Depth 2/hazy/"
        constants.DATASET_RESIDE_TEST_PATH_COMPLETE = "/scratch1/scratch2/neil.delgallego/Synth Hazy - Depth 2/depth/"

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
    print("Is Coare? %d Has GPU available? %d Count: %d" % (
        constants.is_coare, torch.cuda.is_available(), torch.cuda.device_count()))
    print("Torch CUDA version: %s" % torch.version.cuda)

    manualSeed = random.randint(1, 10000)  # use if you want new results
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

    #gt = airlight_trainer.AirlightTrainer(constants.AIRLIGHT_VERSION, constants.ITERATION, device, opts.d_lr)
    gt = lightcoords_trainer.LightCoordsTrainer(device, opts.d_lr)
    gt.update_penalties(opts.airlight_weight, opts.comments)
    start_epoch = 0
    iteration = 0
    if (opts.load_previous):
        checkpoint = torch.load(constants.LIGHTCOORDS_ESTIMATOR_CHECKPATH)
        start_epoch = checkpoint['epoch'] + 1
        iteration = checkpoint['iteration'] + 1
        gt.load_saved_state(checkpoint)

        print("Loaded checkpt: %s Current epoch: %d" % (constants.LIGHTCOORDS_ESTIMATOR_CHECKPATH, start_epoch))
        print("===================================================")

    # Create the dataloader
    train_loader = dataset_loader.load_model_based_transmission_dataset(constants.DATASET_CLEAN_PATH_COMPLETE, constants.DATASET_DEPTH_PATH_COMPLETE, constants.DATASET_LIGHTCOORDS_PATH_COMPLETE,
                                                                        constants.TEST_IMAGE_SIZE, constants.batch_size, opts.img_to_load)

    # Plot some training images
    if (constants.is_coare == 0):
        _, a, b, _, lights_tensor = next(iter(train_loader))
        show_images(a, "Training - RGB Images")
        show_images(b, "Training - Depth Images")
        print("Training - Lights Tensor", np.shape(lights_tensor))

    print("Starting Training Loop...")
    if (constants.is_coare == 0):
        for epoch in range(start_epoch, constants.num_epochs):
            # For each batch in the dataloader
            for i, train_data in enumerate(train_loader, 0):
                _, rgb_batch, _, light_batch, airlight_batch = train_data
                rgb_tensor = rgb_batch.to(device).float()
                airlight_tensor = airlight_batch.to(device).float()
                lights_coord_tensor =light_batch.to(device).float()

                # perform color transfer first
                rgb_tensor = color_transfer_gan(rgb_tensor)
                gt.train(rgb_tensor, lights_coord_tensor)

                if ((i + 1) % 400 == 0):
                    gt.save_states(epoch, iteration)
                    gt.visdom_report(iteration, rgb_tensor)

    else:
        for i, train_data in enumerate(train_loader, 0):
            _, rgb_batch, _, airlight_batch = train_data
            rgb_tensor = rgb_batch.to(device).float()
            airlight_tensor = airlight_batch.to(device).float()

            # perform color transfer first
            rgb_tensor = color_transfer_gan(rgb_tensor)
            gt.train(rgb_tensor, airlight_tensor)
            if ((i + 1) % 300 == 0):
                print("Iterating %d" % i)

        gt.save_states(start_epoch, iteration)


# FIX for broken pipe num_workers issue.
if __name__ == "__main__":
    main(sys.argv)