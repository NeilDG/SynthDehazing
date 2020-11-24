# -*- coding: utf-8 -*-
"""
Main entry for GAN training
Created on Sun Apr 19 13:22:06 2020

@author: delgallegon
"""

from __future__ import print_function
import sys
from optparse import OptionParser
import random
import torch
import torch.nn.parallel
import torch.utils.data
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from loaders import dataset_loader
from model import latent_network
from trainers.latent_trainers import latent_trainer_64
import constants

parser = OptionParser()
parser.add_option('--coare', type=int, help="Is running on COARE?", default=0)
parser.add_option('--img_to_load', type=int, help="Image to load?", default=-1)
parser.add_option('--load_previous', type=int, help="Load previous?", default=0)
parser.add_option('--iteration', type=int, help="Style version?", default="1")
parser.add_option('--adv_weight', type=float, help="Weight", default="1.0")
parser.add_option('--likeness_weight', type=float, help="Weight", default="1.0")
parser.add_option('--gen_blocks', type=int, help="Weight", default="19")
parser.add_option('--g_lr', type=float, help="LR", default="0.0005")
parser.add_option('--d_lr', type=float, help="LR", default="0.0002")

#--img_to_load=-1 --load_previous=0
#Update config if on COARE
def update_config(opts):
    constants.is_coare = opts.coare

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

        constants.DATASET_HAZY_TEST_PATH_1_HAZY = "/scratch1/scratch2/neil.delgallego/VEMON Dataset/frames/"
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


    trainer = latent_trainer_64.LatentTrainer(constants.DEHAZER_VERSION, constants.ITERATION, device, opts.g_lr, opts.d_lr)
    trainer.update_penalties(opts.adv_weight, opts.likeness_weight)

    start_epoch = 0
    iteration = 0

    if(opts.load_previous):
        latent_checkpoint = torch.load(constants.LATENT_CHECKPATH_64)
        start_epoch = latent_checkpoint['epoch'] + 1
        iteration = latent_checkpoint['iteration'] + 1
        trainer.load_saved_state(iteration, latent_checkpoint, constants.GENERATOR_KEY, constants.DISCRIMINATOR_KEY, constants.OPTIMIZER_KEY)

        print("Loaded checkpt: %s Current epoch: %d" % (constants.LATENT_CHECKPATH_64, start_epoch))
        print("===================================================")

    # Create the dataloader
    train_loader = dataset_loader.load_latent_dataset(constants.DATASET_VEMON_PATH_PATCH_64, constants.batch_size, opts.img_to_load)
    test_loader = dataset_loader.load_dehaze_dataset_test(constants.DATASET_VEMON_PATH_COMPLETE, constants.batch_size, opts.img_to_load)

    index = 0

    # Plot some training images
    if(constants.is_coare == 0):
        _, train_batch = next(iter(train_loader))
        _, test_batch = next(iter(test_loader))

        show_images(train_batch, "Training Images")
        show_images(test_batch, "Test Images")

    print("Starting Training Loop...")
    if(constants.is_coare == 0):
        for epoch in range(start_epoch, constants.num_epochs):
            # For each batch in the dataloader
            for i, (_, train_batch) in enumerate(iter(train_loader)):
                train_tensor = train_batch.to(device)

                #train dehazing
                trainer.train(train_tensor)

                if((i) % 2000 == 0):
                    _, test_batch = next(iter(test_loader))

                    test_batch = test_batch.to(device)
                    trainer.visdom_report(iteration, train_tensor, test_batch)

                    iteration = iteration + 1
                    index = (index + 1) % len(test_loader)

                    if(index == 0):
                        test_loader = dataset_loader.load_dehaze_dataset_test(constants.DATASET_VEMON_PATH_COMPLETE, constants.batch_size, opts.img_to_load)

                    trainer.save_states(epoch, iteration, constants.LATENT_CHECKPATH_64, constants.GENERATOR_KEY, constants.DISCRIMINATOR_KEY, constants.OPTIMIZER_KEY)
    else:
        for i, (_, train_batch) in enumerate(iter(train_loader)):
                train_tensor = train_batch.to(device)

                #train dehazing
                trainer.train(train_tensor)

        #save every X epoch
        trainer.save_states(start_epoch, iteration, constants.LATENT_CHECKPATH_64, constants.GENERATOR_KEY, constants.DISCRIMINATOR_KEY, constants.OPTIMIZER_KEY)

#FIX for broken pipe num_workers issue.
if __name__=="__main__":
    main(sys.argv)

