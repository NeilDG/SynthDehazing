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
from trainers import airlight_gen_trainer, early_stopper
from trainers import airlight_trainer
from model import style_transfer_gan as color_gan
from model import vanilla_cycle_gan as cycle_gan
import constants

parser = OptionParser()
parser.add_option('--server_config', type=int, help="Is running on COARE?", default=0)
parser.add_option('--cuda_device', type=str, help="CUDA Device?", default="cuda:0")
parser.add_option('--albedo_checkpt', type=str, help="Albedo checkpt?", default="checkpoint/albedo_transfer_v1.04_1.pt")
parser.add_option('--img_to_load', type=int, help="Image to load?", default=-1)
parser.add_option('--load_previous', type=int, help="Load previous?", default=0)
parser.add_option('--iteration', type=int, help="Style version?", default="1")
parser.add_option('--num_layers', type=int, help="num_layers", default="4")
parser.add_option('--batch_size', type=int, help="batch_size", default="1024") #16384 default
parser.add_option('--patch_size', type=int, help="patch_size", default="32")
parser.add_option('--num_workers', type=int, help="Workers", default="12")
parser.add_option('--d_lr', type=float, help="LR", default="0.0002")
parser.add_option('--t_min', type=float, help="", default="0.1")
parser.add_option('--t_max', type=float, help="", default="1.2")
parser.add_option('--a_min', type=float, help="", default="0.1")
parser.add_option('--a_max', type=float, help="", default="0.95")
parser.add_option('--style_transfer_enabled', type=int, help="", default="1")
parser.add_option('--use_lowres', type=int, help="", default="0")
parser.add_option('--comments', type=str, help="comments for bookmarking", default="New airlight estimator network. 32 x 32 patch")


# --img_to_load=-1 --load_previous=0
# Update config if on COARE
def update_config(opts):
    constants.server_config = opts.server_config
    constants.ITERATION = str(opts.iteration)
    constants.AIRLIGHT_ESTIMATOR_CHECKPATH = 'checkpoint/' + constants.AIRLIGHT_VERSION + "_" + constants.ITERATION + '.pt'

    if (constants.server_config == 1):
        constants.num_workers = opts.num_workers

        print("Using COARE configuration. Workers: ", constants.num_workers, "Path: ", constants.AIRLIGHT_ESTIMATOR_CHECKPATH)

        constants.DATASET_CLEAN_PATH_COMPLETE_STYLED_3 = "/scratch1/scratch2/neil.delgallego/Synth Hazy 3/clean - styled/"
        constants.DATASET_ALBEDO_PATH_COMPLETE_3 = "/scratch1/scratch2/neil.delgallego/Synth Hazy 3/albedo/"
        constants.DATASET_ALBEDO_PATH_PSEUDO_3 = "/scratch1/scratch2/neil.delgallego/Synth Hazy 3/albedo - pseudo/"
        constants.DATASET_DEPTH_PATH_COMPLETE_3 = "/scratch1/scratch2/neil.delgallego/Synth Hazy 3/depth/"
        constants.DATASET_OHAZE_HAZY_PATH_COMPLETE = "/scratch1/scratch2/neil.delgallego/Hazy Dataset Benchmark/O-HAZE/hazy/"
        constants.DATASET_OHAZE_CLEAN_PATH_COMPLETE = "/scratch1/scratch2/neil.delgallego/Hazy Dataset Benchmark/O-HAZE/GT/"
        constants.DATASET_RESIDE_TEST_PATH_COMPLETE = "/scratch1/scratch2/neil.delgallego/Hazy Dataset Benchmark/RESIDE-Unannotated/"
        constants.DATASET_STANDARD_PATH_COMPLETE = "/scratch1/scratch2/neil.delgallego/Hazy Dataset Benchmark/RESIDE-Unannotated/"

    elif (constants.server_config == 2):
        constants.num_workers = opts.num_workers
        constants.ALBEDO_CHECKPT = opts.albedo_checkpt

        print("Using CCS configuration. Workers: ", constants.num_workers, "Path: ", constants.AIRLIGHT_ESTIMATOR_CHECKPATH)

        constants.DATASET_CLEAN_STYLED_LOW_PATH = "Synth Hazy - Low/clean - styled/"
        constants.DATASET_DEPTH_LOW_PATH = "Synth Hazy - Low/depth/"
        constants.DATASET_CLEAN_PATH_COMPLETE_STYLED_3 = "clean - styled/"
        constants.DATASET_DEPTH_PATH_COMPLETE_3 = "depth/"
        constants.DATASET_CLEAN_PATH_COMPLETE_STYLED_TEST = "Synth Hazy - Test Set/clean/"
        constants.DATASET_DEPTH_PATH_COMPLETE_TEST = "Synth Hazy - Test Set/depth/"

        constants.DATASET_OHAZE_HAZY_PATH_COMPLETE = "Hazy Dataset Benchmark/O-HAZE/hazy/"
        constants.DATASET_OHAZE_CLEAN_PATH_COMPLETE = "Hazy Dataset Benchmark/O-HAZE/GT/"
        constants.DATASET_STANDARD_PATH_COMPLETE = "Hazy Dataset Benchmark/Standard/"
        constants.DATASET_RESIDE_TEST_PATH_COMPLETE = "Hazy Dataset Benchmark/RESIDE-Unannotated/"


def show_images(img_tensor, caption):
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.title(caption)
    plt.imshow(np.transpose(
        vutils.make_grid(img_tensor.to(device)[:constants.display_size], nrow=8, padding=2, normalize=True).cpu(),
        (1, 2, 0)))
    plt.show()


def main(argv):
    (opts, args) = parser.parse_args(argv)
    update_config(opts)
    print(opts)
    print("=====================BEGIN============================")
    print("Server config? %d Has GPU available? %d Count: %d" % (constants.server_config, torch.cuda.is_available(), torch.cuda.device_count()))
    print("Torch CUDA version: %s" % torch.version.cuda)

    manualSeed = random.randint(1, 10000)  # use if you want new results
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    #
    device = torch.device(opts.cuda_device if (torch.cuda.is_available()) else "cpu")
    print("Device: %s" % device)

    airlight_term_trainer = airlight_trainer.AirlightTrainer(device, opts.batch_size, opts.num_layers, opts.d_lr)
    airlight_term_trainer.update_penalties(1.0, opts.comments)

    early_stopper_l1 = early_stopper.EarlyStopper(40, early_stopper.EarlyStopperMethod.L1_TYPE, 10)

    start_epoch = 0
    iteration = 0

    if (opts.load_previous):
        checkpoint = torch.load(constants.AIRLIGHT_ESTIMATOR_CHECKPATH)
        start_epoch = checkpoint['epoch']+ 1
        iteration = checkpoint['iteration'] + 1
        airlight_term_trainer.load_saved_state(checkpoint)

        print("Loaded airlight estimator checkpt: %s Current epoch: %d" % (constants.AIRLIGHT_ESTIMATOR_CHECKPATH, start_epoch))
        print("===================================================")

    # Create the dataloader
    if(opts.style_transfer_enabled == 1 and opts.use_lowres == 0):
        train_loader = dataset_loader.load_airlight_dataset_train(constants.DATASET_CLEAN_PATH_COMPLETE_STYLED_3, constants.DATASET_DEPTH_PATH_COMPLETE_3, opts, False)
        test_loader = dataset_loader.load_airlight_dataset_train(constants.DATASET_CLEAN_PATH_COMPLETE_STYLED_TEST, constants.DATASET_DEPTH_PATH_COMPLETE_TEST, opts, False)
    elif(opts.use_lowres == 1):
        train_loader = dataset_loader.load_airlight_dataset_train(constants.DATASET_CLEAN_STYLED_LOW_PATH, constants.DATASET_DEPTH_LOW_PATH, opts, False)
        test_loader = dataset_loader.load_airlight_dataset_train(constants.DATASET_CLEAN_STYLED_LOW_PATH, constants.DATASET_DEPTH_LOW_PATH, opts, False)
    else:
        train_loader = dataset_loader.load_airlight_dataset_train(constants.DATASET_CLEAN_PATH_COMPLETE_3, constants.DATASET_DEPTH_PATH_COMPLETE_3, opts, False)
        test_loader = dataset_loader.load_airlight_dataset_train(constants.DATASET_CLEAN_PATH_COMPLETE_TEST, constants.DATASET_DEPTH_PATH_COMPLETE_TEST, opts, False)

    # Plot some training images
    if (constants.server_config == 0):
        _, a, b, c, d = next(iter(train_loader))
        # _, d = next(iter(test_loaders[0]))
        show_images(a, "Training - RGB Images")
        show_images(b, "Training - Transmission Images")
        show_images(c, "Training - 1 - T Images")

    print("Starting Training Loop for Airlight Estimator...", constants.AIRLIGHT_ESTIMATOR_CHECKPATH)
    for epoch in range(start_epoch, constants.num_epochs):
        # For each batch in the dataloader
        for i, (train_data, test_data) in enumerate(zip(train_loader, test_loader)):
            _, rgb_batch, _, _, atmosphere_light = train_data
            rgb_tensor = rgb_batch.to(device).float()
            light_tensor = atmosphere_light.to(device).float()

            airlight_term_trainer.train_a1(rgb_tensor, light_tensor)

            _, rgb_batch, _, _, atmosphere_light = test_data
            rgb_tensor = rgb_batch.to(device).float()
            light_tensor = atmosphere_light.to(device).float()
            airlight_like = airlight_term_trainer.test(rgb_tensor, light_tensor)
            iteration = iteration + 1

            if (early_stopper_l1.test(airlight_term_trainer, epoch, iteration, airlight_like, light_tensor)):
                break

            if ((i) % 5 == 0):
                airlight_term_trainer.save_states_unstable(epoch, iteration)
                # airlight_term_trainer.visdom_report(iteration, rgb_tensor)

        if (early_stopper_l1.test(airlight_term_trainer, epoch, iteration, airlight_like, light_tensor)):
            break


# FIX for broken pipe num_workers issue.
if __name__ == "__main__":
    main(sys.argv)

