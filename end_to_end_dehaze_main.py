# -*- coding: utf-8 -*-
"""
Main entry for GAN training
Created on Sun Apr 19 13:22:06 2020

@author: delgallegon
"""

from __future__ import print_function

import itertools
import os
import sys
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
from trainers import ffa_trainer, early_stopper
import constants

parser = OptionParser()
parser.add_option('--server_config', type=int, help="Is running on COARE?", default=0)
parser.add_option('--cuda_device', type=str, help="CUDA Device?", default="cuda:0")
parser.add_option('--img_to_load', type=int, help="Image to load?", default=-1)
parser.add_option('--load_previous', type=int, help="Load previous?", default=0)
parser.add_option('--iteration', type=int, help="Style version?", default="1")
parser.add_option('--clarity_weight', type=float, help="Weight", default="1.0")
parser.add_option('--gen_blocks', type=int, help="Weight", default="19")
parser.add_option('--batch_size', type=int, help="batch_size", default="64")
parser.add_option('--num_workers', type=int, help="Workers", default="12")
parser.add_option('--g_lr', type=float, help="LR", default="0.0002")

#--img_to_load=-1 --load_previous=0
#Update config if on COARE
def update_config(opts):
    constants.server_config = opts.server_config

    if(constants.server_config == 1):
        constants.ITERATION = str(opts.iteration)
        constants.num_workers = opts.num_workers
        constants.END_TO_END_CHECKPATH = 'checkpoint/' + constants.END_TO_END_DEHAZER_VERSION + "_" + constants.ITERATION + '.pt'

        print("Using COARE configuration. Workers: ", constants.num_workers, "Path: ", constants.END_TO_END_CHECKPATH)

        constants.DATASET_HAZY_END_TO_END_PATH = "/scratch1/scratch2/neil.delgallego/Synth Hazy - End-to-End/hazy/"
        constants.DATASET_CLEAN_END_TO_END_PATH = "/scratch1/scratch2/neil.delgallego/Synth Hazy - End-to-End/clean/"
        constants.DATASET_HAZY_END_TO_END_PATH_TEST = "/scratch1/scratch2/neil.delgallego/Synth Hazy - End-to-End - Test/hazy/"
        constants.DATASET_CLEAN_END_TO_END_PATH_TEST = "/scratch1/scratch2/neil.delgallego/Synth Hazy - End-to-End - Test/clean/"

    elif(constants.server_config == 2):
        constants.ITERATION = str(opts.iteration)
        constants.num_workers = opts.num_workers
        constants.ALBEDO_CHECKPT = opts.albedo_checkpt
        constants.END_TO_END_CHECKPATH = 'checkpoint/' + constants.END_TO_END_DEHAZER_VERSION + "_" + constants.ITERATION + '.pt'

        print("Using CCS configuration. Workers: ", constants.num_workers, "Path: ", constants.END_TO_END_CHECKPATH)

        constants.DATASET_HAZY_END_TO_END_PATH = "/scratch1/scratch2/neil.delgallego/Synth Hazy - End-to-End/hazy/"
        constants.DATASET_CLEAN_END_TO_END_PATH = "/scratch1/scratch2/neil.delgallego/Synth Hazy - End-to-End/clean/"
        constants.DATASET_HAZY_END_TO_END_PATH_TEST = "/scratch1/scratch2/neil.delgallego/Synth Hazy - End-to-End - Test/hazy/"
        constants.DATASET_CLEAN_END_TO_END_PATH_TEST = "/scratch1/scratch2/neil.delgallego/Synth Hazy - End-to-End - Test/clean/"

def show_images(img_tensor, caption):
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    plt.figure(figsize=constants.FIG_SIZE)
    plt.axis("off")
    plt.title(caption)
    plt.imshow(np.transpose(vutils.make_grid(img_tensor.to(device)[:constants.display_size], nrow = 8, padding=2, normalize=True).cpu(),(1,2,0)))
    plt.show()

def main(argv):
    (opts, args) = parser.parse_args(argv)
    update_config(opts)
    print("=====================BEGIN============================")
    print("Server config? %d Has GPU available? %d Count: %d" % (constants.server_config, torch.cuda.is_available(), torch.cuda.device_count()))
    print("Torch CUDA version: %s" % torch.version.cuda)

    manualSeed = random.randint(1, 10000) # use if you want new results
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    device = torch.device(opts.cuda_device if (torch.cuda.is_available()) else "cpu")
    print("Device: %s" % device)

    dehazer = ffa_trainer.FFATrainer(constants.END_TO_END_DEHAZER_VERSION, constants.ITERATION, device, blocks = opts.gen_blocks, lr = opts.g_lr)
    dehazer.update_penalties(opts.clarity_weight)
    early_stopper_l1 = early_stopper.EarlyStopper(10, early_stopper.EarlyStopperMethod.L1_TYPE)

    start_epoch = 0
    iteration = 0

    if(opts.load_previous):
        dehaze_checkpoint = torch.load(constants.END_TO_END_CHECKPATH)
        start_epoch = dehaze_checkpoint['epoch'] + 1
        iteration = dehaze_checkpoint['iteration'] + 1
        dehazer.load_saved_state(iteration, dehaze_checkpoint)

        print("Loaded checkpt: %s Current epoch: %d" % (constants.END_TO_END_CHECKPATH, start_epoch))
        print("===================================================")

    # Create the dataloader
    train_loader = dataset_loader.load_end_to_end_dehazing_dataset(constants.DATASET_HAZY_END_TO_END_PATH, constants.DATASET_CLEAN_END_TO_END_PATH, opts.batch_size, opts.img_to_load, opts.num_workers)
    test_loaders = [dataset_loader.load_end_to_end_dehazing_dataset(constants.DATASET_HAZY_END_TO_END_PATH_TEST, constants.DATASET_CLEAN_END_TO_END_PATH_TEST, opts.batch_size, opts.img_to_load, 2)]
    index = 0

    # Plot some training images
    if(constants.server_config == 0):
        _, hazy_batch, clear_batch = next(iter(train_loader))
        show_images(hazy_batch, "Training - Hazy Images")
        show_images(clear_batch, "Training - Clean Images")

        _, hazy_batch, clear_batch = next(iter(test_loaders[0]))
        show_images(hazy_batch, "Test - Hazy Images")
        show_images(clear_batch, "Test - Clean Images")

    print("Starting Training Loop...")
    # for i in range(len(unseen_loaders)):
    #     _, hazy_batch = next(iter(unseen_loaders[i]))
    #     hazy_tensor = hazy_batch.to(device)
    #
    #     dehazer.visdom_infer_test(hazy_tensor, i)
    #
    # for i, test_data in enumerate(test_loaders[0], 0):
    #     _, hazy_batch, clear_batch = test_data
    #     hazy_tensor = hazy_batch.to(device)
    #     clear_tensor = clear_batch.to(device)
    #
    #     dehazer.visdom_infer_test_paired(hazy_tensor, clear_tensor, i)
    #     break

    for epoch in range(start_epoch, constants.num_epochs):
        # For each batch in the dataloader
        for i, (train_data, test_data) in enumerate(zip(train_loader, itertools.cycle(test_loaders[0]))):
            _, hazy_batch, clear_batch = train_data
            hazy_tensor = hazy_batch.to(device)
            clear_tensor = clear_batch.to(device)

            dehazer.train(hazy_tensor, clear_tensor)
            iteration = iteration + 1

            _, hazy_batch, clear_batch = test_data
            hazy_tensor = hazy_batch.to(device)
            clear_tensor = clear_batch.to(device)
            clear_like = dehazer.test(hazy_tensor)

            if (early_stopper_l1.test(dehazer, epoch, iteration, clear_like, clear_tensor)):
                break

            # if (i % 100 == 0):
                # dehazer.save_states(epoch, iteration)
                # dehazer.visdom_report(iteration)
                # _, hazy_batch, transmission_batch, clear_batch, atmosphere_batch = train_data
                # hazy_tensor = hazy_batch.to(device)
                # clear_tensor = clear_batch.to(device)
                # transmission_tensor = transmission_batch.to(device).float()
                # atmosphere_tensor = atmosphere_batch.to(device).float()
                #
                # dehazer.visdom_infer_train(hazy_tensor, transmission_tensor, atmosphere_tensor, clear_tensor)
                #
                # _, hazy_batch, clear_batch = test_data
                # hazy_tensor = hazy_batch.to(device)
                # clear_tensor = clear_batch.to(device)
                # dehazer.visdom_infer_test_paired(hazy_tensor, clear_tensor, 0)

        if (early_stopper_l1.test(dehazer, epoch, iteration, clear_like, clear_tensor)):
            break
#FIX for broken pipe num_workers issue.
if __name__=="__main__":
    main(sys.argv)

