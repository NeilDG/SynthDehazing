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
from model import style_transfer_gan as color_gan
import constants

parser = OptionParser()
parser.add_option('--coare', type=int, help="Is running on COARE?", default=0)
parser.add_option('--img_to_load', type=int, help="Image to load?", default=-1)
parser.add_option('--load_previous', type=int, help="Load previous?", default=0)
parser.add_option('--iteration', type=int, help="Style version?", default="1")
parser.add_option('--adv_weight', type=float, help="Weight", default="1.0")
parser.add_option('--clarity_weight', type=float, help="Weight", default="300.0")
parser.add_option('--gen_blocks', type=int, help="Weight", default="8")
parser.add_option('--brightness_enhance', type=float, help="Weight", default="1.00")
parser.add_option('--contrast_enhance', type=float, help="Weight", default="1.00")
parser.add_option('--g_lr', type=float, help="LR", default="0.0005")
parser.add_option('--d_lr', type=float, help="LR", default="0.0002")

#--img_to_load=-1 --load_previous=0
#Update config if on COARE
def update_config(opts):
    constants.is_coare = opts.coare
    constants.brightness_enhance = opts.brightness_enhance
    constants.contrast_enhance = opts.contrast_enhance

    if(constants.is_coare == 1):
        print("Using COARE configuration.")
        constants.batch_size = 512

        constants.ITERATION = str(opts.iteration)
        constants.DEHAZER_CHECKPATH = 'checkpoint/' + constants.DEHAZER_VERSION + "_" + constants.ITERATION +'.pt'
        constants.COLORIZER_CHECKPATH = 'checkpoint/' + constants.COLORIZER_VERSION + "_" + constants.ITERATION +'.pt'

        constants.DATASET_NOISY_GTA_PATH = "/scratch1/scratch2/neil.delgallego/Noisy GTA/noisy/"
        constants.DATASET_CLEAN_GTA_PATH = "/scratch1/scratch2/neil.delgallego/Noisy GTA/clean/"
        constants.DATASET_VEMON_PATH_COMPLETE = "/scratch1/scratch2/neil.delgallego/VEMON Dataset/frames/"

        constants.DATASET_HAZY_PATH_COMPLETE = "/scratch1/scratch2/neil.delgallego/Synth Hazy/hazy/"
        constants.DATASET_CLEAN_PATH_COMPLETE = "/scratch1/scratch2/neil.delgallego/Synth Hazy/clean/"

        constants.DATASET_HAZY_TEST_PATH_1 = "/scratch1/scratch2/neil.delgallego/VEMON Dataset/frames/"
        constants.DATASET_HAZY_TEST_PATH_2 = "/scratch1/scratch2/neil.delgallego/VEMON Dataset/frames/"

        constants.num_workers = 4

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

    dehazer = dehaze_trainer.DehazeTrainer(constants.DEHAZER_VERSION, constants.ITERATION, device, opts.gen_blocks, opts.g_lr, opts.d_lr)
    dehazer.update_penalties(opts.clarity_weight, opts.adv_weight)

    start_epoch = 0
    iteration = 0

    #load color transfer
    color_transfer_checkpt = torch.load('checkpoint/dehaze_colortransfer_v1.06_10.pt')
    color_transfer_gan = color_gan.Generator().to(device)
    color_transfer_gan.load_state_dict(color_transfer_checkpt[constants.GENERATOR_KEY + "A"])
    print("Color transfer GAN model loaded.")

    if(opts.load_previous):
        dehaze_checkpoint = torch.load(constants.DEHAZER_CHECKPATH)
        start_epoch = dehaze_checkpoint['epoch'] + 1
        iteration = dehaze_checkpoint['iteration'] + 1
        dehazer.load_saved_state(iteration, dehaze_checkpoint, constants.GENERATOR_KEY, constants.DISCRIMINATOR_KEY, constants.OPTIMIZER_KEY)

        print("Loaded checkpt: %s %s Current epoch: %d" % (constants.DEHAZER_CHECKPATH, constants.COLORIZER_CHECKPATH, start_epoch))
        print("===================================================")

    # Create the dataloader
    synth_train_loader = dataset_loader.load_dehaze_dataset(constants.DATASET_HAZY_PATH_COMPLETE, constants.DATASET_CLEAN_PATH_COMPLETE, constants.batch_size, opts.img_to_load)
    vemon_test_loader = dataset_loader.load_dehaze_dataset_test(constants.DATASET_VEMON_PATH_COMPLETE, constants.batch_size, opts.img_to_load)
    hazy_1_loader = dataset_loader.load_dehaze_dataset_test(constants.DATASET_HAZY_TEST_PATH_1, constants.batch_size, opts.img_to_load)
    hazy_2_loader = dataset_loader.load_dehaze_dataset_test(constants.DATASET_HAZY_TEST_PATH_2, constants.batch_size, opts.img_to_load)

    index = 0

    # Plot some training images
    if(constants.is_coare == 0):
        _, synth_hazy_batch, synth_clean_batch = next(iter(synth_train_loader))
        _, rgb_batch = next(iter(vemon_test_loader))
        _, hazy_1_batch = next(iter(hazy_1_loader))
        _, hazy_2_batch = next(iter(hazy_2_loader))

        show_images(synth_hazy_batch, "Training - Hazy Images")
        show_images(synth_clean_batch, "Training - Clean Images")
        show_images(rgb_batch, "Test - Vemon Images")
        show_images(hazy_1_batch, "Test - Hazy 1 Images")
        show_images(hazy_2_batch, "Test - Hazy 2 Images")

    print("Starting Training Loop...")
    if(constants.is_coare == 0):
        for epoch in range(start_epoch, constants.num_epochs):
            # For each batch in the dataloader
            for i, (_, hazy_batch, clean_batch) in enumerate(iter(synth_train_loader)):
                hazy_tensor = hazy_batch.to(device)
                clean_tensor = clean_batch.to(device)

                #color transfer
                with torch.no_grad():
                    hazy_tensor = color_transfer_gan(hazy_tensor)
                    clean_tensor = color_transfer_gan(clean_tensor)

                #train dehazing
                dehazer.train(hazy_tensor, clean_tensor)

                if(i % 100 == 0):
                    _, rgb_batch = next(iter(vemon_test_loader))
                    _, hazy_1_batch = next(iter(hazy_1_loader))
                    _, hazy_2_batch = next(iter(hazy_2_loader))

                    rgb_batch = rgb_batch.to(device)
                    hazy_1_batch = hazy_1_batch.to(device)
                    hazy_2_batch = hazy_2_batch.to(device)

                    dehazer.visdom_report(iteration, hazy_tensor, clean_tensor, rgb_batch, hazy_1_batch, hazy_2_batch)

                    iteration = iteration + 1

                    index = (index + 1) % len(vemon_test_loader)

                    if(index == 0):
                      vemon_test_loader = dataset_loader.load_dehaze_dataset_test(constants.DATASET_VEMON_PATH_COMPLETE, constants.batch_size, opts.img_to_load)
                      hazy_1_loader = dataset_loader.load_dehaze_dataset_test(constants.DATASET_HAZY_TEST_PATH_1, constants.batch_size, opts.img_to_load)
                      hazy_2_loader = dataset_loader.load_dehaze_dataset_test(constants.DATASET_HAZY_TEST_PATH_2, constants.batch_size, opts.img_to_load)
            dehazer.save_states(epoch, iteration, constants.DEHAZER_CHECKPATH, constants.GENERATOR_KEY, constants.DISCRIMINATOR_KEY, constants.OPTIMIZER_KEY)
    else:
        for i, (dehaze_data) in enumerate(zip(synth_train_loader)):
                _, hazy_batch, clean_batch = dehaze_data
                hazy_tensor = hazy_batch.to(device)
                clean_tensor = clean_batch.to(device)

                # color transfer
                with torch.no_grad():
                    hazy_tensor = color_transfer_gan(hazy_tensor)
                    clean_tensor = color_transfer_gan(clean_tensor)

                #train dehazing
                dehazer.train(hazy_tensor, clean_tensor)

        #save every X epoch
        dehazer.save_states(start_epoch, iteration, constants.DEHAZER_CHECKPATH, constants.GENERATOR_KEY, constants.DISCRIMINATOR_KEY, constants.OPTIMIZER_KEY)

#FIX for broken pipe num_workers issue.
if __name__=="__main__":
    main(sys.argv)

