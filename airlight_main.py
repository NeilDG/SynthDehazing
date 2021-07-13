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
parser.add_option('--airlight_weight', type=float, help="Weight", default="10.0")
parser.add_option('--d_lr', type=float, help="LR", default="0.00005")
parser.add_option('--batch_size', type=int, help="batch_size", default="512")
parser.add_option('--comments', type=str, help="comments for bookmarking", default = "Airlight estimation network using same architecture for A and B. \n "
                                                                                     "Accepts albedo input. \n"
                                                                                     "New architecture based on DCGAN. \n"
                                                                                     "Airlight range converted to uniform [0.6 - 0.95]")

#--img_to_load=-1 --load_previous=0
# Update config if on COARE
def update_config(opts):
    constants.is_coare = opts.coare

    if (constants.is_coare == 1):
        print("Using COARE configuration.")
        constants.ITERATION = str(opts.iteration)

        constants.AIRLIGHT_ESTIMATOR_CHECKPATH = 'checkpoint/' + constants.AIRLIGHT_ESTIMATOR_CHECKPATH + "_" + constants.ITERATION + '.pt'

        constants.DATASET_CLEAN_PATH_COMPLETE_STYLED_3 = "/scratch1/scratch2/neil.delgallego/Synth Hazy 3/clean/"
        constants.DATASET_ALBEDO_PATH_COMPLETE_3 = "/scratch1/scratch2/neil.delgallego/Synth Hazy 3/albedo/"
        constants.DATASET_DEPTH_PATH_COMPLETE_3 = "/scratch1/scratch2/neil.delgallego/Synth Hazy 3/depth/"
        constants.DATASET_OHAZE_HAZY_PATH_COMPLETE = "/scratch1/scratch2/neil.delgallego/Hazy Dataset Benchmark/O-HAZE/hazy/"

        constants.DATASET_ALBEDO_PATH_PATCH_3 = "/scratch1/scratch2/neil.delgallego/Synth Hazy 3 - Patch/albedo/"
        constants.DATASET_ALBEDO_PATH_PSEUDO_PATCH_3 = "/scratch1/scratch2/neil.delgallego/Synth Hazy 3 - Patch/albedo - pseudo/"
        constants.DATASET_CLEAN_PATH_PATCH_STYLED_3 = "/scratch1/scratch2/neil.delgallego/Synth Hazy 3 - Patch/clean - styled/"

def show_images(img_tensor, caption):
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    plt.figure(figsize=(16, 4))
    plt.axis("off")
    plt.title(caption)
    plt.imshow(np.transpose(
        vutils.make_grid(img_tensor.to(device)[:constants.display_size], nrow=8, padding=2, normalize=True).cpu(),
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

    gt = airlight_trainer.AirlightTrainer(device, opts.batch_size, opts.d_lr)
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
    train_loaders = [dataset_loader.load_airlight_train_dataset(constants.DATASET_ALBEDO_PATH_COMPLETE_3, constants.DATASET_CLEAN_PATH_COMPLETE_STYLED_3, constants.DATASET_DEPTH_PATH_COMPLETE_3, opts.batch_size, opts.img_to_load),
                    dataset_loader.load_airlight_train_dataset(constants.DATASET_ALBEDO_PATH_PSEUDO_3, constants.DATASET_CLEAN_PATH_COMPLETE_STYLED_3, constants.DATASET_DEPTH_PATH_COMPLETE_3, opts.batch_size, opts.img_to_load)]

    test_loader = dataset_loader.load_airlight_test_dataset(constants.DATASET_ALBEDO_PATH_PSEUDO_TEST, constants.DATASET_CLEAN_PATH_COMPLETE_STYLED_TEST, constants.DATASET_DEPTH_PATH_COMPLETE_TEST, opts.batch_size, 5000)

    # Plot some training images
    if (constants.is_coare == 0):
        _, a, b, airlight_tensor = next(iter(train_loaders[0]))
        show_images(a, "Training - Albedo Images")
        show_images(b, "Training - Styled Images")
        print("Training - Airlight Tensor", np.shape(airlight_tensor))
        print("Values: ",airlight_tensor.numpy())

        _, a, b, airlight_tensor = next(iter(train_loaders[1]))
        show_images(a, "Training - Pseudo Albedo Images")
        show_images(b, "Training - Styled Images")
        print("Training - Pseudo Airlight Tensor", np.shape(airlight_tensor))
        print("Values: ", airlight_tensor.numpy())

        _, a, b, airlight_tensor = next(iter(test_loader))
        show_images(a, "Test - Pseudo Albedo Images")
        show_images(b, "Test - Styled Images")
        print("Test - Pseudo Airlight Tensor", np.shape(airlight_tensor))
        print("Values: ", airlight_tensor.numpy())

    print("Starting Training Loop...")
    for epoch in range(start_epoch, constants.num_epochs):
        # For each batch in the dataloader
        for i, (train_data, pseudo_train_data, test_data) in enumerate(zip(train_loaders[0], train_loaders[1], itertools.cycle(test_loader))):
            _, albedo_batch, styled_batch, airlight_batch = train_data
            albedo_batch = albedo_batch.to(device).float()
            styled_batch = styled_batch.to(device).float()
            airlight_batch= airlight_batch.to(device).float()

            gt.train_a1(styled_batch, airlight_batch)
            gt.train_a2(albedo_batch, styled_batch, airlight_batch)

            _, albedo_batch, styled_batch, airlight_batch = pseudo_train_data
            albedo_batch = albedo_batch.to(device).float()
            styled_batch = styled_batch.to(device).float()
            airlight_batch = airlight_batch.to(device).float()

            gt.train_a1(styled_batch, airlight_batch)
            gt.train_a2(albedo_batch, styled_batch, airlight_batch)

            _, albedo_batch, styled_batch, airlight_batch = test_data
            albedo_batch = albedo_batch.to(device).float()
            styled_batch = styled_batch.to(device).float()
            airlight_batch = airlight_batch.to(device).float()

            gt.test(albedo_batch, styled_batch, airlight_batch)

            iteration = iteration + 1

        gt.save_states(epoch, iteration)
        gt.visdom_report(iteration, albedo_batch, styled_batch)


# FIX for broken pipe num_workers issue.
if __name__ == "__main__":
    main(sys.argv)