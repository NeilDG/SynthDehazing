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
from trainers import airlight_gen_trainer
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
parser.add_option('--adv_weight', type=float, help="Weight", default="1.0")
parser.add_option('--likeness_weight', type=float, help="Weight", default="10.0")
parser.add_option('--edge_weight', type=float, help="Weight", default="1.0")
parser.add_option('--batch_size', type=int, help="batch_size", default="128")
parser.add_option('--g_lr', type=float, help="LR", default="0.0002")
parser.add_option('--d_lr', type=float, help="LR", default="0.0002")
parser.add_option('--is_unet',type=int, help="Is Unet?", default="0")
parser.add_option('--comments', type=str, help="comments for bookmarking", default = "Patch-based transmission estimation network using CycleGAN architecture. \n"
                                                                                     "32 x 32 patch size. \n"
                                                                                     "0.3 - 0.95 = A range")

# --img_to_load=-1 --load_previous=0
# Update config if on COARE
def update_config(opts):
    constants.server_config = opts.server_config

    if (constants.server_config == 1):
        constants.ITERATION = str(opts.iteration)
        #constants.num_workers = opts.num_workers
        constants.AIRLIGHT_GEN_CHECKPATH = 'checkpoint/' + constants.AIRLIGHT_GEN_VERSION + "_" + constants.ITERATION + '.pt'

        print("Using COARE configuration. Workers: ", constants.num_workers, "Path: ", constants.AIRLIGHT_GEN_CHECKPATH)

        constants.DATASET_CLEAN_PATH_COMPLETE_STYLED_3 = "/scratch1/scratch2/neil.delgallego/Synth Hazy 3/clean - styled/"
        constants.DATASET_ALBEDO_PATH_COMPLETE_3 = "/scratch1/scratch2/neil.delgallego/Synth Hazy 3/albedo/"
        constants.DATASET_ALBEDO_PATH_PSEUDO_3 = "/scratch1/scratch2/neil.delgallego/Synth Hazy 3/albedo - pseudo/"
        constants.DATASET_DEPTH_PATH_COMPLETE_3 = "/scratch1/scratch2/neil.delgallego/Synth Hazy 3/depth/"
        constants.DATASET_OHAZE_HAZY_PATH_COMPLETE = "/scratch1/scratch2/neil.delgallego/Hazy Dataset Benchmark/O-HAZE/hazy/"
        constants.DATASET_OHAZE_CLEAN_PATH_COMPLETE = "/scratch1/scratch2/neil.delgallego/Hazy Dataset Benchmark/O-HAZE/GT/"
        constants.DATASET_RESIDE_TEST_PATH_COMPLETE = "/scratch1/scratch2/neil.delgallego/Hazy Dataset Benchmark/RESIDE-Unannotated/"
        constants.DATASET_STANDARD_PATH_COMPLETE = "/scratch1/scratch2/neil.delgallego/Hazy Dataset Benchmark/RESIDE-Unannotated/"

    elif (constants.server_config == 2):
        constants.ITERATION = str(opts.iteration)
        #constants.num_workers = opts.num_workers
        constants.ALBEDO_CHECKPT = opts.albedo_checkpt
        constants.AIRLIGHT_GEN_CHECKPATH = 'checkpoint/' + constants.AIRLIGHT_GEN_VERSION + "_" + constants.ITERATION + '.pt'

        print("Using CCS configuration. Workers: ", constants.num_workers, "Path: ", constants.AIRLIGHT_GEN_CHECKPATH)

        constants.DATASET_CLEAN_PATH_COMPLETE_STYLED_3 = "clean - styled/"
        constants.DATASET_DEPTH_PATH_COMPLETE_3 = "depth/"
        constants.DATASET_CLEAN_PATH_COMPLETE_STYLED_TEST = constants.DATASET_CLEAN_PATH_COMPLETE_STYLED_3
        constants.DATASET_DEPTH_PATH_COMPLETE_TEST = constants.DATASET_DEPTH_PATH_COMPLETE_3

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
    print("=====================BEGIN============================")
    print("Server config? %d Has GPU available? %d Count: %d" % (constants.server_config, torch.cuda.is_available(), torch.cuda.device_count()))
    print("Torch CUDA version: %s" % torch.version.cuda)

    manualSeed = random.randint(1, 10000)  # use if you want new results
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    device = torch.device(opts.cuda_device if (torch.cuda.is_available()) else "cpu")
    print("Device: %s" % device)

    gen_trainer = airlight_gen_trainer.AirlightGenTrainer(device, opts.batch_size, opts.is_unet, opts.g_lr, opts.d_lr)
    gen_trainer.update_penalties(opts.adv_weight, opts.likeness_weight, opts.edge_weight, opts.comments)
    
    airlight_term_trainer = airlight_trainer.AirlightTrainer(device, opts.batch_size, opts.d_lr)
    airlight_term_trainer.update_penalties(1.0, opts.comments)
    
    start_epoch = [0, 0]
    iteration = [0, 0]

    if (opts.load_previous):
        checkpoint = torch.load(constants.AIRLIGHT_GEN_CHECKPATH)
        start_epoch[0] = checkpoint['epoch'] + 1
        iteration[0] = checkpoint['iteration'] + 1
        gen_trainer.load_saved_state(checkpoint)

        print("Loaded airlight gen checkpt: %s Current epoch: %d" % (constants.AIRLIGHT_GEN_CHECKPATH, start_epoch[0]))
        print("===================================================")

        checkpoint = torch.load(constants.AIRLIGHT_ESTIMATOR_CHECKPATH)
        start_epoch[1] = checkpoint['epoch'] + 1
        iteration[1] = checkpoint['iteration'] + 1
        airlight_term_trainer.load_saved_state(checkpoint)

        print("Loaded airlight estimator checkpt: %s Current epoch: %d" % (constants.AIRLIGHT_ESTIMATOR_CHECKPATH, start_epoch[1]))
        print("===================================================")

    # Create the dataloader
    train_loader = dataset_loader.load_airlight_dataset_train(constants.DATASET_CLEAN_PATH_COMPLETE_STYLED_3, constants.DATASET_DEPTH_PATH_COMPLETE_3, False, opts.batch_size, opts.img_to_load)
    test_loader = dataset_loader.load_airlight_dataset_train(constants.DATASET_CLEAN_PATH_COMPLETE_STYLED_TEST, constants.DATASET_DEPTH_PATH_COMPLETE_TEST, False, opts.batch_size, opts.img_to_load)

    # validation_group = [dataset_loader.load_airlight_dataset_test(constants.DATASET_CLEAN_PATH_COMPLETE_STYLED_3, opts.batch_size, 500),
    #                 dataset_loader.load_airlight_dataset_test(constants.DATASET_OHAZE_HAZY_PATH_COMPLETE, opts.batch_size, 500),
    #                 dataset_loader.load_airlight_dataset_test(constants.DATASET_RESIDE_TEST_PATH_COMPLETE, opts.batch_size, 500)]
    # validation_loaders = validation_group
    index = 0

    # Plot some training images
    if (constants.server_config == 0):
        _, a, b, c, d = next(iter(train_loader))
        #_, d = next(iter(test_loaders[0]))
        show_images(a, "Training - RGB Images")
        show_images(b, "Training - Transmission Images")
        show_images(c, "Training - Atmosphere Images")

    # for i, train_data in enumerate(train_loader, 0):
    #     _, rgb_batch, _, atmosphere_batch, _ = train_data
    #     rgb_tensor = rgb_batch.to(device).float()
    #     atmosphere_batch = atmosphere_batch.to(device).float()
    #     gen_trainer.visdom_infer_train(rgb_tensor, atmosphere_batch, i)
    #     break

    print("Starting Training Loop for Airlight Gen...")
    for epoch in range(start_epoch[0], constants.num_epochs):
        # For each batch in the dataloader
        for i, (train_data, test_data) in enumerate(zip(train_loader, test_loader)):
            _, rgb_batch, _, atmosphere_batch, _ = train_data
            rgb_tensor = rgb_batch.to(device).float()
            atmosphere_tensor = atmosphere_batch.to(device).float()

            gen_trainer.train(iteration[0], rgb_tensor, atmosphere_tensor)

            _, rgb_batch, _, atmosphere_batch, _ = test_data
            rgb_tensor = rgb_batch.to(device).float()
            atmosphere_tensor = atmosphere_batch.to(device).float()

            gen_trainer.test(epoch, rgb_tensor, atmosphere_tensor)

            if(gen_trainer.did_stop_condition_met()):
                break

            iteration[0] = iteration[0] + 1
            if ((i) % 300 == 0):
                gen_trainer.save_states(epoch, iteration[0])
                gen_trainer.visdom_report(iteration[0])
                gen_trainer.visdom_infer_train(rgb_tensor, atmosphere_tensor, 0)
                # for j in range(len(validation_loaders)):
                #     _, rgb_batch = next(iter(validation_loaders[j]))
                #     rgb_batch = rgb_batch.to(device)
                #     gen_trainer.visdom_infer_test(rgb_batch, j)
                #
                #     index = (index + 1) % len(validation_loaders[0])
                #     if (index == 0):
                #         validation_loaders = validation_group

    train_loader = dataset_loader.load_airlight_dataset_train(constants.DATASET_CLEAN_PATH_COMPLETE_STYLED_3, constants.DATASET_DEPTH_PATH_COMPLETE_3, False, 8192, opts.img_to_load)
    test_loader = dataset_loader.load_airlight_dataset_train(constants.DATASET_CLEAN_PATH_COMPLETE_STYLED_TEST, constants.DATASET_DEPTH_PATH_COMPLETE_TEST, False, 8192, opts.img_to_load)

    print("Starting Training Loop for Airlight Estimator...")
    for epoch in range(start_epoch[1], constants.num_epochs):
        # For each batch in the dataloader
        for i, (train_data, test_data) in enumerate(zip(train_loader, test_loader)):
            _, rgb_batch, _, _, atmosphere_light = train_data
            rgb_tensor = rgb_batch.to(device).float()
            light_tensor = atmosphere_light.to(device).float()

            airlight_term_trainer.train_a1(rgb_tensor, light_tensor)

            _, rgb_batch, _, _, atmosphere_light = test_data
            rgb_tensor = rgb_batch.to(device).float()
            light_tensor = atmosphere_light.to(device).float()
            airlight_term_trainer.test(epoch, rgb_tensor, light_tensor)

            if (airlight_term_trainer.did_stop_condition_met()):
                break

            iteration[1] = iteration[1] + 1

            if ((i) % 10 == 0):
                airlight_term_trainer.save_states(epoch, iteration[1])
                airlight_term_trainer.visdom_report(iteration[1], rgb_tensor)

# FIX for broken pipe num_workers issue.
if __name__ == "__main__":
    main(sys.argv)

