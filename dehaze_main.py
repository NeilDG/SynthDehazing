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
parser.add_option('--psnr_loss_weight', type=float, help="Weight", default="0.0")
parser.add_option('--num_blocks', type=int, help="Num Blocks", default = 6)
parser.add_option('--batch_size', type=int, help="batch_size", default="4")
parser.add_option('--g_lr', type=float, help="LR", default="0.0002")
parser.add_option('--d_lr', type=float, help="LR", default="0.0002")
parser.add_option('--comments', type=str, help="comments for bookmarking", default = "Cycle Dehazer GAN.")

#--img_to_load=-1 --load_previous=0
#Update config if on COARE
def update_config(opts):
    constants.is_coare = opts.coare

    if(constants.is_coare == 1):
        print("Using COARE configuration.")

        constants.ITERATION = str(opts.iteration)

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

    dehazer = dehaze_trainer.DehazeTrainer(device, opts.g_lr, opts.d_lr, opts.num_blocks, opts.batch_size)
    dehazer.update_penalties(opts.adv_weight, opts.likeness_weight, opts.psnr_loss_weight, opts.comments)

    start_epoch = 0
    iteration = 0

    #load transmission network
    checkpt = torch.load("checkpoint/transmission_albedo_estimator_v1.04_2.pt")
    transmission_G = cycle_gan.Generator(input_nc=3, output_nc=1, n_residual_blocks=8).to(device)
    transmission_G.load_state_dict(checkpt[constants.GENERATOR_KEY + "A"])
    transmission_G.eval()
    print("Transmission network loaded.")

    #load albedo
    checkpt = torch.load("checkpoint/albedo_transfer_v1.04_1.pt")
    albedo_G = ffa_gan.FFA(gps = 3, blocks = 18).to(device)
    albedo_G.load_state_dict(checkpt[constants.GENERATOR_KEY + "A"])
    albedo_G.eval()
    print("Albedo network loaded.")

    # load atmosphere estimator
    checkpt = torch.load("checkpoint/airlight_estimator_v1.04_1.pt")
    atmosphere_D = dh.AirlightEstimator_V2(num_channels = 3, disc_feature_size = 64).to(device)
    atmosphere_D.load_state_dict(checkpt[constants.DISCRIMINATOR_KEY + "A"])
    atmosphere_D.eval()
    print("Albedo network loaded.")

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
        _, a, b, c = next(iter(train_loader))
        show_images(a, "Training - Hazy Images")
        show_images(b, "Training - Transmission Images")
        show_images(c, "Training - Clear Images")

    print("Starting Training Loop...")
    for epoch in range(start_epoch, constants.num_epochs):
        # For each batch in the dataloader
        for i, train_data in enumerate(train_loader, 0):
            _, hazy_batch, transmission_batch, clear_batch = train_data
            hazy_tensor = hazy_batch.to(device)
            clear_tensor = clear_batch.to(device)

            with torch.no_grad():
                albedo_like = albedo_G(hazy_tensor)
                transmission_like = transmission_G(albedo_like)
                atmosphere_like = atmosphere_D(hazy_tensor)

            dehazer.train(albedo_like, transmission_like, atmosphere_like, clear_tensor)

            if (i % 100 == 0):
                dehazer.save_states(epoch, iteration)
                dehazer.visdom_report(iteration)
                dehazer.visdom_infer_train(albedo_like, transmission_like, atmosphere_like, clear_tensor)

                iteration = iteration + 1
                for i in range(len(test_loaders)):
                    _, hazy_batch = next(iter(test_loaders[i]))
                    hazy_tensor = hazy_batch.to(device)

                    with torch.no_grad():
                        albedo_like = albedo_G(hazy_tensor)
                        transmission_like = transmission_G(albedo_like)
                        atmosphere_like = atmosphere_D(hazy_tensor)

                    dehazer.visdom_infer_test(albedo_like, transmission_like, atmosphere_like)

                    index = (index + 1) % len(test_loaders[0])

                    if (index == 0):
                        test_loaders = [dataset_loader.load_dehaze_dataset_test(constants.DATASET_OHAZE_HAZY_PATH_COMPLETE, opts.batch_size, 500),
                                        dataset_loader.load_dehaze_dataset_test(constants.DATASET_RESIDE_TEST_PATH_COMPLETE, opts.batch_size, 500)]

#FIX for broken pipe num_workers issue.
if __name__=="__main__":
    main(sys.argv)

