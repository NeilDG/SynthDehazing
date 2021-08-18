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
from trainers import ffa_trainer
import constants

parser = OptionParser()
parser.add_option('--server_config', type=int, help="Is running on COARE?", default=0)
parser.add_option('--cuda_device', type=str, help="CUDA Device?", default="cuda:0")
parser.add_option('--img_to_load', type=int, help="Image to load?", default=-1)
parser.add_option('--load_previous', type=int, help="Load previous?", default=0)
parser.add_option('--iteration', type=int, help="Style version?", default="1")
parser.add_option('--clarity_weight', type=float, help="Weight", default="1.0")
parser.add_option('--gen_blocks', type=int, help="Weight", default="19")
parser.add_option('--g_lr', type=float, help="LR", default="0.0002")

#--img_to_load=-1 --load_previous=0
#Update config if on COARE
def update_config(opts):
    constants.server_config = opts.server_config

    if(constants.server_config == 1):
        constants.ITERATION = str(opts.iteration)
        constants.num_workers =opts.num_workers
        constants.DEHAZER_CHECKPATH = 'checkpoint/' + constants.DEHAZER_VERSION + "_" + constants.ITERATION + '.pt'

        print("Using COARE configuration. Workers: ", constants.num_workers, "Path: ", constants.DEHAZER_CHECKPATH)

        constants.DATASET_CLEAN_PATH_COMPLETE_STYLED_3 = "/scratch1/scratch2/neil.delgallego/Synth Hazy 3/clean - styled/"
        constants.DATASET_ALBEDO_PATH_COMPLETE_3 = "/scratch1/scratch2/neil.delgallego/Synth Hazy 3/albedo/"
        constants.DATASET_ALBEDO_PATH_PSEUDO_3 = "/scratch1/scratch2/neil.delgallego/Synth Hazy 3/albedo - pseudo/"
        constants.DATASET_DEPTH_PATH_COMPLETE_3 = "/scratch1/scratch2/neil.delgallego/Synth Hazy 3/depth/"
        constants.DATASET_OHAZE_HAZY_PATH_COMPLETE = "/scratch1/scratch2/neil.delgallego/Hazy Dataset Benchmark/O-HAZE/hazy/"
        constants.DATASET_OHAZE_CLEAN_PATH_COMPLETE = "/scratch1/scratch2/neil.delgallego/Hazy Dataset Benchmark/O-HAZE/GT/"
        constants.DATASET_RESIDE_TEST_PATH_COMPLETE = "/scratch1/scratch2/neil.delgallego/Hazy Dataset Benchmark/RESIDE-Unannotated/"
        constants.DATASET_STANDARD_PATH_COMPLETE = "/scratch1/scratch2/neil.delgallego/Hazy Dataset Benchmark/RESIDE-Unannotated/"

    elif(constants.server_config == 2):
        constants.ITERATION = str(opts.iteration)
        constants.num_workers = opts.num_workers
        constants.ALBEDO_CHECKPT = opts.albedo_checkpt
        constants.DEHAZER_CHECKPATH = 'checkpoint/' + constants.DEHAZER_VERSION + "_" + constants.ITERATION + '.pt'

        print("Using CCS configuration. Workers: ", constants.num_workers, "Path: ", constants.DEHAZER_CHECKPATH)

        constants.DATASET_CLEAN_PATH_COMPLETE_STYLED_3 = "clean - styled/"
        constants.DATASET_DEPTH_PATH_COMPLETE_3 = "depth/"
        constants.DATASET_OHAZE_HAZY_PATH_COMPLETE = "Hazy Dataset Benchmark/O-HAZE/hazy/"
        constants.DATASET_OHAZE_CLEAN_PATH_COMPLETE = "Hazy Dataset Benchmark/O-HAZE/GT/"
        constants.DATASET_STANDARD_PATH_COMPLETE = "Hazy Dataset Benchmark/Standard/"
        constants.DATASET_RESIDE_TEST_PATH_COMPLETE = "Hazy Dataset Benchmark/RESIDE-Unannotated/"

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

    dehazer = ffa_trainer.FFATrainer(constants.DEHAZER_VERSION, constants.ITERATION, device, blocks = opts.gen_blocks, lr = opts.g_lr)
    dehazer.update_penalties(opts.clarity_weight)

    start_epoch = 0
    iteration = 0

    if(opts.load_previous):
        dehaze_checkpoint = torch.load(constants.DEHAZER_CHECKPATH)
        start_epoch = dehaze_checkpoint['epoch'] + 1
        iteration = dehaze_checkpoint['iteration'] + 1
        dehazer.load_saved_state(iteration, dehaze_checkpoint, constants.GENERATOR_KEY, constants.LATENT_VECTOR_KEY, constants.OPTIMIZER_KEY)

        print("Loaded checkpt: %s %s Current epoch: %d" % (constants.DEHAZER_CHECKPATH, constants.COLORIZER_CHECKPATH, start_epoch))
        print("===================================================")

    # Create the dataloader
    synth_train_loader = dataset_loader.load_dehaze_dataset(constants.DATASET_OHAZE_PATH_PATCH_HAZY, constants.DATASET_OHAZE_PATH_PATCH_CLEAN, constants.batch_size, opts.img_to_load)
    synth_test_loader_hazy = dataset_loader.load_dehaze_dataset_test(constants.DATASET_OHAZE_HAZY_PATH_COMPLETE, constants.batch_size, opts.img_to_load)
    synth_test_loader_clean = dataset_loader.load_dehaze_dataset_test(constants.DATASET_OHAZE_CLEAN_PATH_COMPLETE, constants.batch_size, opts.img_to_load)
    vemon_test_loader = dataset_loader.load_dehaze_dataset_test(constants.DATASET_VEMON_PATH_COMPLETE, constants.batch_size, opts.img_to_load)

    index = 0

    # Plot some training images
    if(constants.is_coare == 0):
        _, synth_hazy_batch, synth_clean_batch = next(iter(synth_train_loader))
        _, rgb_batch = next(iter(vemon_test_loader))

        show_images(synth_hazy_batch, "Training - Hazy Images")
        show_images(synth_clean_batch, "Training - Clean Images")
        show_images(rgb_batch, "Test - Vemon Images")

    print("Starting Training Loop...")
    if(constants.is_coare == 0):
        for epoch in range(start_epoch, constants.num_epochs):
            # For each batch in the dataloader
            for i, (_, hazy_batch, clean_batch) in enumerate(iter(synth_train_loader)):
                hazy_tensor = hazy_batch.to(device)
                clean_tensor = clean_batch.to(device)

                #train dehazing
                dehazer.train(hazy_tensor, clean_tensor)

                if((i + 1) % 500 == 0):
                    _, vemon_batch = next(iter(vemon_test_loader))
                    _, synth_hazy_batch = next(iter(synth_test_loader_hazy))
                    _, synth_clean_batch = next(iter(synth_test_loader_clean))

                    vemon_batch = vemon_batch.to(device)
                    synth_hazy_batch = synth_hazy_batch.to(device)
                    synth_clean_batch = synth_clean_batch.to(device)

                    dehazer.visdom_report(iteration, hazy_tensor, clean_tensor, synth_hazy_batch, synth_clean_batch, vemon_batch)

                    iteration = iteration + 1
                    index = (index + 1) % len(vemon_test_loader)

                    if(index == 0):
                        synth_test_loader_hazy = dataset_loader.load_dehaze_dataset_test(
                            constants.DATASET_OHAZE_HAZY_PATH_COMPLETE, constants.batch_size, opts.img_to_load)
                        synth_test_loader_clean = dataset_loader.load_dehaze_dataset_test(
                            constants.DATASET_OHAZE_CLEAN_PATH_COMPLETE, constants.batch_size, opts.img_to_load)
                        vemon_test_loader = dataset_loader.load_dehaze_dataset_test(
                            constants.DATASET_VEMON_PATH_COMPLETE, constants.batch_size, opts.img_to_load)

                    dehazer.save_states(epoch, iteration, constants.DEHAZER_CHECKPATH, constants.GENERATOR_KEY, constants.LATENT_VECTOR_KEY, constants.OPTIMIZER_KEY)
    else:
        for i, (_, hazy_batch, clean_batch) in enumerate(iter(synth_train_loader)):
                hazy_tensor = hazy_batch.to(device)
                clean_tensor = clean_batch.to(device)

                #train dehazing
                dehazer.train(hazy_tensor, clean_tensor)

        #save every X epoch
        dehazer.save_states(start_epoch, iteration, constants.DEHAZER_CHECKPATH, constants.GENERATOR_KEY, constants.LATENT_VECTOR_KEY, constants.OPTIMIZER_KEY)

#FIX for broken pipe num_workers issue.
if __name__=="__main__":
    main(sys.argv)

