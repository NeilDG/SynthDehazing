# -*- coding: utf-8 -*-
"""
Main entry for GAN training
Created on Sun Apr 19 13:22:06 2020

@author: delgallegon
"""

from __future__ import print_function
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
from trainers import dehaze_trainer
from model import ffa_net as ffa_gan
from model import vanilla_cycle_gan as cycle_gan
from model import dehaze_discriminator as dh
import constants

parser = OptionParser()
parser.add_option('--coare', type=int, help="Is running on COARE?", default=0)
parser.add_option('--img_to_load', type=int, help="Image to load?", default=-1)
parser.add_option('--load_previous', type=int, help="Load previous?", default=0)
parser.add_option('--iteration', type=int, help="Style version?", default="1")
parser.add_option('--adv_weight', type=float, help="Weight", default="1.0")
parser.add_option('--likeness_weight', type=float, help="Weight", default="10.0")
parser.add_option('--edge_weight', type=float, help="Weight", default="5.0")
parser.add_option('--is_t_unet',type=int, help="Is Unet?", default="0")
parser.add_option('--t_num_blocks', type=int, help="Num Blocks", default = 10)
parser.add_option('--is_a_unet',type=int, help="Is Unet?", default="0")
parser.add_option('--a_num_blocks', type=int, help="Num Blocks", default = 10)
parser.add_option('--batch_size', type=int, help="batch_size", default="8")
parser.add_option('--g_lr', type=float, help="LR", default="0.0002")
parser.add_option('--d_lr', type=float, help="LR", default="0.0002")
parser.add_option('--dehaze_filter_strength', type=float, help="LR", default="0.5")
parser.add_option('--comments', type=str, help="comments for bookmarking", default = "Cycle Dehazer GAN.")

#--img_to_load=-1 --load_previous=0
#Update config if on COARE
def update_config(opts):
    constants.is_coare = opts.coare

    if(constants.is_coare == 1):
        print("Using COARE configuration.")

        constants.ITERATION = str(opts.iteration)
        constants.DEHAZE_FILTER_STRENGTH = opts.dehaze_filter_strength

        constants.TRANSMISSION_ESTIMATOR_CHECKPATH = 'checkpoint/' + constants.TRANSMISSION_VERSION + "_" + constants.ITERATION + '.pt'

        constants.DATASET_CLEAN_PATH_COMPLETE_STYLED_3 = "/scratch1/scratch2/neil.delgallego/Synth Hazy 3/clean/"
        constants.DATASET_ALBED1O_PATH_COMPLETE_3 = "/scratch1/scratch2/neil.delgallego/Synth Hazy 3/albedo/"
        constants.DATASET_ALBEDO_PATH_PSEUDO_3 = "/scratch1/scratch2/neil.delgallego/Synth Hazy 3/albedo - pseudo/"
        constants.DATASET_DEPTH_PATH_COMPLETE_3 = "/scratch1/scratch2/neil.delgallego/Synth Hazy 3/depth/"
        constants.DATASET_OHAZE_HAZY_PATH_COMPLETE = "/scratch1/scratch2/neil.delgallego/Hazy Dataset Benchmark/O-HAZE/hazy/"
        constants.DATASET_RESIDE_TEST_PATH_COMPLETE = "/scratch1/scratch2/neil.delgallego/Hazy Dataset Benchmark/O-HAZE/hazy/"

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
    print("=========BEGIN============")

    print("Is Coare? %d Has GPU available? %d Count: %d" % (constants.is_coare, torch.cuda.is_available(), torch.cuda.device_count()))
    print("Torch CUDA version: %s" % torch.version.cuda)

    manualSeed = random.randint(1, 10000) # use if you want new results
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    print("Device: %s" % device)

    dehazer = dehaze_trainer.DehazeTrainer(device, opts.g_lr, opts.d_lr, opts.batch_size)
    dehazer.declare_models(opts.t_num_blocks, opts.is_t_unet, opts.a_num_blocks, opts.is_a_unet)
    dehazer.update_penalties(opts.adv_weight, opts.likeness_weight, opts.edge_weight, opts.comments)

    start_epoch = 0
    iteration = 0

    if(opts.load_previous):
        dehaze_checkpoint = torch.load(constants.DEHAZER_CHECKPATH)
        start_epoch = dehaze_checkpoint['epoch'] + 1
        iteration = dehaze_checkpoint['iteration'] + 1
        dehazer.load_saved_state(dehaze_checkpoint)

        print("Loaded checkpt: %s %s Current epoch: %d" % (constants.DEHAZER_CHECKPATH, constants.COLORIZER_CHECKPATH, start_epoch))
        print("===================================================")

    # Create the dataloader
    train_loader = dataset_loader.load_dehazing_dataset(constants.DATASET_CLEAN_PATH_COMPLETE_STYLED_3, constants.DATASET_DEPTH_PATH_COMPLETE_3, True, opts.batch_size, opts.img_to_load)
    test_loaders = [dataset_loader.load_dehaze_dataset_test(constants.DATASET_OHAZE_HAZY_PATH_COMPLETE, opts.batch_size, 500),
                    dataset_loader.load_dehaze_dataset_test(constants.DATASET_RESIDE_TEST_PATH_COMPLETE, opts.batch_size, 500)]

    index = 0

    # Plot some training images
    if(constants.is_coare == 0):
        _, a, b, c, d = next(iter(train_loader))
        show_images(a, "Training - Hazy Images")
        show_images(b, "Training - Transmission Images")
        show_images(c, "Training - Clear Images")
        show_images(d, "Training - Atmosphere Images")

    print("Starting Training Loop...")
    # for i in range(len(test_loaders)):
    #     _, hazy_batch = next(iter(test_loaders[i]))
    #     hazy_tensor = hazy_batch.to(device)
    #
    #     dehazer.visdom_infer_test(hazy_tensor, i)
    #     break
    #
    # for i, train_data in enumerate(train_loader, 0):
    #     _, hazy_batch, transmission_batch, clear_batch, atmosphere_batch = train_data
    #     hazy_tensor = hazy_batch.to(device)
    #     clear_tensor = clear_batch.to(device)
    #     transmission_tensor = transmission_batch.to(device).float()
    #     atmosphere_tensor = atmosphere_batch.to(device).float()
    #
    #     dehazer.visdom_infer_train(hazy_tensor, transmission_tensor, atmosphere_tensor, clear_tensor)
    #     break

    for epoch in range(start_epoch, constants.num_epochs):
        # For each batch in the dataloader
        for i, train_data in enumerate(train_loader, 0):
            _, hazy_batch, transmission_batch, clear_batch, atmosphere_batch = train_data
            hazy_tensor = hazy_batch.to(device)
            clear_tensor = clear_batch.to(device)
            transmission_tensor = transmission_batch.to(device).float()
            atmosphere_tensor = atmosphere_batch.to(device).float()

            dehazer.train(hazy_tensor, transmission_tensor, atmosphere_tensor, clear_tensor)

            if (i % 5000 == 0):
                dehazer.save_states(epoch, iteration)
                dehazer.visdom_report(iteration)
                dehazer.visdom_infer_train(hazy_tensor, transmission_tensor, atmosphere_tensor, clear_tensor)

                iteration = iteration + 1
                for k in range(len(test_loaders)):
                    _, hazy_batch = next(iter(test_loaders[k]))
                    hazy_tensor = hazy_batch.to(device)

                    dehazer.visdom_infer_test(hazy_tensor, k)

                    index = (index + 1) % len(test_loaders[0])

                    if (index == 0):
                        test_loaders = [dataset_loader.load_dehaze_dataset_test(constants.DATASET_OHAZE_HAZY_PATH_COMPLETE, opts.batch_size, 500),
                                        dataset_loader.load_dehaze_dataset_test(constants.DATASET_RESIDE_TEST_PATH_COMPLETE, opts.batch_size, 500)]

#FIX for broken pipe num_workers issue.
if __name__=="__main__":
    main(sys.argv)

