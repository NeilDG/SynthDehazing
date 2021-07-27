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
parser.add_option('--likeness_weight', type=float, help="Weight", default="100.0")
parser.add_option('--edge_weight', type=float, help="Weight", default="10.0")
parser.add_option('--batch_size', type=int, help="batch_size", default="8")
parser.add_option('--g_lr', type=float, help="LR", default="0.00002")
parser.add_option('--d_lr', type=float, help="LR", default="0.00002")
parser.add_option('--is_unet',type=int, help="Is Unet?", default="0")
parser.add_option('--num_workers', type=int, help="Workers", default="12")
parser.add_option('--comments', type=str, help="comments for bookmarking", default = "Patch-based transmission estimation network using CycleGAN architecture. \n"
                                                                                     "128 x 128 patch size. \n"
                                                                                     "0.3 - 0.95 = A range")

# --img_to_load=-1 --load_previous=0
# Update config if on COARE
def update_config(opts):
    constants.server_config = opts.server_config

    if (constants.server_config == 1):
        constants.ITERATION = str(opts.iteration)
        constants.num_workers = opts.num_workers
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
        constants.num_workers = opts.num_workers
        constants.ALBEDO_CHECKPT = opts.albedo_checkpt
        constants.AIRLIGHT_GEN_CHECKPATH = 'checkpoint/' + constants.AIRLIGHT_GEN_VERSION + "_" + constants.ITERATION + '.pt'

        print("Using CCS configuration. Workers: ", constants.num_workers, "Path: ", constants.AIRLIGHT_GEN_CHECKPATH)

        constants.DATASET_CLEAN_PATH_COMPLETE_STYLED_3 = "clean - styled/"
        constants.DATASET_DEPTH_PATH_COMPLETE_3 = "depth/"
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

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    print("Device: %s" % device)

    gt = airlight_gen_trainer.AirlightGenTrainer(device, opts.batch_size, opts.is_unet, opts.g_lr, opts.d_lr)
    gt.update_penalties(opts.adv_weight, opts.likeness_weight, opts.edge_weight, opts.comments)
    start_epoch = 0
    iteration = 0

    if (opts.load_previous):
        checkpoint = torch.load(constants.AIRLIGHT_GEN_CHECKPATH)
        start_epoch = checkpoint['epoch'] + 1
        iteration = checkpoint['iteration'] + 1
        gt.load_saved_state(checkpoint)

        print("Loaded checkpt: %s Current epoch: %d" % (constants.AIRLIGHT_GEN_CHECKPATH, start_epoch))
        print("===================================================")

    # Create the dataloader
    train_loader = dataset_loader.load_dehazing_dataset(constants.DATASET_CLEAN_PATH_COMPLETE_STYLED_3, constants.DATASET_DEPTH_PATH_COMPLETE_3, False, opts.batch_size, opts.img_to_load)
    # test_loaders = [dataset_loader.load_dehaze_dataset_test(constants.DATASET_CLEAN_PATH_COMPLETE_STYLED_3, opts.batch_size, 500),
    #                 dataset_loader.load_dehaze_dataset_test(constants.DATASET_ALBEDO_PATH_PSEUDO_3,opts.batch_size, 500),
    #                 dataset_loader.load_dehaze_dataset_test(constants.DATASET_OHAZE_HAZY_PATH_COMPLETE, opts.batch_size, 500),
    #                 dataset_loader.load_dehaze_dataset_test(constants.DATASET_RESIDE_TEST_PATH_COMPLETE, opts.batch_size, 500)]
    index = 0

    # Plot some training images
    if (constants.server_config == 0):
        _, a, b, c = next(iter(train_loader))
        #_, d = next(iter(test_loaders[0]))
        show_images(a, "Training - RGB Images")
        show_images(b, "Training - Transmission Images")
        show_images(c, "Training - Atmosphere Images")

    print("Starting Training Loop...")
    # for i, train_data in enumerate(train_loader, 0):
    #     _, rgb_batch, _, atmosphere_batch = train_data
    #     rgb_tensor = rgb_batch.to(device).float()
    #     atmosphere_batch = atmosphere_batch.to(device).float()
    #     gt.visdom_infer_train(rgb_tensor, atmosphere_batch, i)
    #     break

    for epoch in range(start_epoch, constants.num_epochs):
        # For each batch in the dataloader
        for i, train_data in enumerate(train_loader, 0):
            _, rgb_batch, _, atmosphere_batch = train_data
            rgb_tensor = rgb_batch.to(device).float()
            atmosphere_tensor = atmosphere_batch.to(device).float()

            gt.train(iteration, rgb_tensor, atmosphere_tensor)
            iteration = iteration + 1
            if ((i) % 2000 == 0):
                gt.save_states(epoch, iteration)
                # gt.visdom_report(iteration)
                # gt.visdom_infer_train(rgb_tensor, atmosphere_tensor, 0)
                # for j in range(len(test_loaders)):
                #     _, rgb_batch = next(iter(test_loaders[j]))
                #     rgb_batch = rgb_batch.to(device)
                #     gt.visdom_infer_test(rgb_batch, j)
                #
                #     index = (index + 1) % len(test_loaders[0])
                #     if (index == 0):
                #         test_loaders = [dataset_loader.load_dehaze_dataset_test(constants.DATASET_CLEAN_PATH_COMPLETE_STYLED_3, opts.batch_size, 500),
                #                         dataset_loader.load_dehaze_dataset_test(constants.DATASET_ALBEDO_PATH_PSEUDO_3, opts.batch_size, 500),
                #                         dataset_loader.load_dehaze_dataset_test(constants.DATASET_OHAZE_HAZY_PATH_COMPLETE, opts.batch_size, 500),
                #                         dataset_loader.load_dehaze_dataset_test(constants.DATASET_RESIDE_TEST_PATH_COMPLETE, opts.batch_size, 500)]

# FIX for broken pipe num_workers issue.
if __name__ == "__main__":
    main(sys.argv)

