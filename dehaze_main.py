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
from trainers import dehaze_trainer
from trainers import early_stopper
import constants
import itertools

parser = OptionParser()
parser.add_option('--server_config', type=int, help="Is running on COARE?", default=0)
parser.add_option('--cuda_device', type=str, help="CUDA Device?", default="cuda:0")
parser.add_option('--albedo_checkpt', type=str, help="Albedo checkpt?", default="checkpoint/albedo_transfer_v1.04_1.pt")
parser.add_option('--img_to_load', type=int, help="Image to load?", default=-1)
parser.add_option('--load_previous', type=int, help="Load previous?", default=0)
parser.add_option('--iteration', type=int, help="Style version?", default="1")
parser.add_option('--adv_weight', type=float, help="Weight", default="1.0")
parser.add_option('--likeness_weight', type=float, help="Weight", default="10.0")
parser.add_option('--edge_weight', type=float, help="Weight", default="5.0")
parser.add_option('--clear_like_weight', type=float, help="Weight", default="0.0")
parser.add_option('--is_t_unet',type=int, help="Is Unet?", default="0")
parser.add_option('--t_num_blocks', type=int, help="Num Blocks", default = 10)
parser.add_option('--a_num_blocks', type=int, help="Num Blocks", default = 4)
parser.add_option('--batch_size', type=int, help="batch_size", default="32")
parser.add_option('--g_lr', type=float, help="LR", default="0.0002")
parser.add_option('--d_lr', type=float, help="LR", default="0.0002")
parser.add_option('--num_workers', type=int, help="Workers", default="12")
parser.add_option('--comments', type=str, help="comments for bookmarking", default = "Joint training for transmission and atmospheric map. Size 32 x 32\n"
                                                                                     "0.3 - 0.95 = A range")
#--img_to_load=-1 --load_previous=0
# --server_config=2 --cuda_device=cuda:1
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

    print("Server config? %d Has GPU available? %d Count: %d" % (constants.server_config, torch.cuda.is_available(), torch.cuda.device_count()))
    print("Torch CUDA version: %s" % torch.version.cuda)

    manualSeed = random.randint(1, 10000) # use if you want new results
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    device = torch.device(opts.cuda_device if (torch.cuda.is_available()) else "cpu")
    print("Device: %s" % device)

    dehazer = dehaze_trainer.DehazeTrainer(device, opts.g_lr, opts.d_lr, opts.batch_size)
    dehazer.declare_models(opts.t_num_blocks, opts.is_t_unet, opts.a_num_blocks)
    dehazer.update_penalties(opts.adv_weight, opts.likeness_weight, opts.edge_weight, opts.clear_like_weight, opts.comments)

    early_stopper_l1 = early_stopper.EarlyStopper(20, early_stopper.EarlyStopperMethod.L1_TYPE)

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
    test_loaders = [dataset_loader.load_dehaze_dataset_test_paired(constants.DATASET_OHAZE_HAZY_PATH_COMPLETE, constants.DATASET_OHAZE_CLEAN_PATH_COMPLETE, opts.batch_size, opts.img_to_load)]
    # unseen_loaders = [dataset_loader.load_dehaze_dataset_test(constants.DATASET_CLEAN_PATH_COMPLETE_STYLED_3, opts.batch_size, 500),
    #                 dataset_loader.load_dehaze_dataset_test(constants.DATASET_STANDARD_PATH_COMPLETE, opts.batch_size, 500),
    #                 dataset_loader.load_dehaze_dataset_test(constants.DATASET_RESIDE_TEST_PATH_COMPLETE, opts.batch_size, 500)]

    index = 0

    # Plot some training images
    if(constants.server_config == 0):
        _, a, b, c, d = next(iter(train_loader))
        show_images(a, "Training - Hazy Images")
        show_images(b, "Training - Transmission Images")
        show_images(c, "Training - Clear Images")
        show_images(dehazer.provide_clean_like(a, b, d), "Training - Clear-Like Images")


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
            _, hazy_batch, transmission_batch, clear_batch, atmosphere_batch = train_data
            hazy_tensor = hazy_batch.to(device)
            clear_tensor = clear_batch.to(device)
            transmission_tensor = transmission_batch.to(device).float()
            atmosphere_tensor = atmosphere_batch.to(device).float()

            dehazer.train(iteration, hazy_tensor, transmission_tensor, atmosphere_tensor, clear_tensor)
            iteration = iteration + 1

            _, hazy_batch, clear_batch = test_data
            hazy_tensor = hazy_batch.to(device)
            clear_tensor = clear_batch.to(device)
            clear_like = dehazer.test(hazy_tensor, clear_tensor)

            if(early_stopper_l1.test(epoch, clear_like, clear_tensor)):
                break

            if (i % 300 == 0):
                dehazer.save_states(epoch, iteration)
                dehazer.visdom_report(iteration)
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

        if (early_stopper_l1.test(epoch, clear_like, clear_tensor)):
            break

#FIX for broken pipe num_workers issue.
if __name__=="__main__":
    main(sys.argv)

