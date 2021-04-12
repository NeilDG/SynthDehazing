# -*- coding: utf-8 -*-

import os
from model import dehaze_discriminator as dh
import constants
import torch
import itertools
import torch.nn as nn
from utils import plot_utils
import torch.cuda.amp as amp

class AirlightTrainer:
    def __init__(self, gpu_device, lr=0.0002):
        self.gpu_device = gpu_device
        self.lr = lr
        self.D_A = dh.AirlightEstimator_V1(input_nc=3, downsampling_layers = 3, residual_blocks = 3).to(self.gpu_device)
        self.D_B = dh.AirlightEstimator_V1(input_nc=6, downsampling_layers = 3, residual_blocks = 3).to(self.gpu_device)

        self.visdom_reporter = plot_utils.VisdomReporter()
        self.initialize_dict()

        self.optimizerDA = torch.optim.Adam(itertools.chain(self.D_A.parameters()), lr=self.lr)
        self.schedulerDA = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizerDA, patience=1000, threshold=0.00005)
        self.optimizerDB = torch.optim.Adam(itertools.chain(self.D_B.parameters()), lr=self.lr)
        self.schedulerDB = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizerDB, patience=1000, threshold=0.00005)

        self.fp16_scaler = amp.GradScaler()  # for automatic mixed precision

    def initialize_dict(self):
        self.AIRLOSS_A_KEY = "AIRLOSS_A_KEY"
        self.AIRLOSS_B_KEY = "AIRLOSS_B_KEY"
        self.AIRLOSS_A_KEY_TEST = "AIRLOSS_A_KEY_TEST"
        self.AIRLOSS_B_KEY_TEST = "AIRLOSS_b_KEY_TEST"

        # what to store in visdom?
        self.losses_dict = {}
        self.losses_dict[self.AIRLOSS_A_KEY] = []
        self.losses_dict[self.AIRLOSS_B_KEY] = []
        self.losses_dict[self.AIRLOSS_A_KEY_TEST] = []
        self.losses_dict[self.AIRLOSS_B_KEY_TEST] = []

        self.caption_dict = {}
        self.caption_dict[self.AIRLOSS_A_KEY] = "Airlight loss per iteration"
        self.caption_dict[self.AIRLOSS_B_KEY] = "Airlight (with light coords) loss per iteration"
        self.caption_dict[self.AIRLOSS_A_KEY_TEST] = "Airlight - testloss per iteration"
        self.caption_dict[self.AIRLOSS_B_KEY_TEST] = "Airlight (with light coords) - test loss per iteration"


    def network_loss(self, pred, target):
        loss = nn.L1Loss()
        return loss(pred, target)

    def update_penalties(self, loss_weight, comments):
        # what penalties to use for losses?
        self.loss_weight = loss_weight

        # save hyperparameters for bookkeeping
        HYPERPARAMS_PATH = "checkpoint/" + constants.AIRLIGHT_VERSION + "_" + constants.ITERATION + ".config"
        with open(HYPERPARAMS_PATH, "w") as f:
            print("Version: ", constants.AIRLIGHT_ESTIMATOR_CHECKPATH, file=f)
            print("Comment: ", comments, file=f)
            print("Learning rate for D: ", str(self.lr), file=f)
            print("====================================", file=f)
            print("Airlight weight: ", str(self.loss_weight), file=f)

    def train(self, albedo_tensor, rgb_tensor, airlight_tensor):
        with amp.autocast():
            self.D_A.train()
            self.optimizerDA.zero_grad()

            #train first model
            airlight_tensor = torch.unsqueeze(airlight_tensor, 1)
            D_A_loss = self.network_loss(self.D_A(rgb_tensor), airlight_tensor) * self.loss_weight

            self.losses_dict[self.AIRLOSS_A_KEY].append(D_A_loss.item())

            #train second model
            self.D_B.train()
            self.optimizerDB.zero_grad()
            D_B_loss = self.network_loss(self.D_B(torch.cat([rgb_tensor, albedo_tensor], 1)), airlight_tensor) * self.loss_weight

            self.losses_dict[self.AIRLOSS_B_KEY].append(D_B_loss.item())

            errD = D_A_loss + D_B_loss
            self.fp16_scaler.scale(errD).backward()
            self.fp16_scaler.step(self.optimizerDA)
            self.fp16_scaler.step(self.optimizerDB)
            self.schedulerDA.step(errD)
            self.schedulerDB.step(errD)

            self.fp16_scaler.update()


    def test(self, albedo_tensor, rgb_tensor, airlight_tensor):
        self.D_A.eval()
        self.D_B.eval()
        with torch.no_grad(), amp.autocast():
            airlight_tensor = torch.unsqueeze(airlight_tensor, 1)
            D_A_loss = self.network_loss(self.D_A(rgb_tensor), airlight_tensor) * self.loss_weight

            self.losses_dict[self.AIRLOSS_A_KEY_TEST].append(D_A_loss.item())

            D_B_loss = self.network_loss(self.D_B(torch.cat([rgb_tensor, albedo_tensor], 1)), airlight_tensor) * self.loss_weight

            self.losses_dict[self.AIRLOSS_B_KEY_TEST].append(D_B_loss.item())


    def visdom_report(self, iteration, albedo_tensor, rgb_tensor):
        # report to visdom
        self.visdom_reporter.plot_airlight_comparison(self.AIRLOSS_A_KEY_TEST, iteration,
                                                      [self.losses_dict[self.AIRLOSS_A_KEY], self.losses_dict[self.AIRLOSS_A_KEY_TEST]],
                                                      [self.losses_dict[self.AIRLOSS_B_KEY], self.losses_dict[self.AIRLOSS_B_KEY_TEST]],
                                                      [str(constants.AIRLIGHT_VERSION) + str(constants.ITERATION) + " Airlight (Standard) - Train Loss", "Airlight (Standard) - Test Loss"],
                                                      ["Airloss (Albedo) - Train loss", "Airloss (Albedo) - Test Loss"])

        self.visdom_reporter.plot_image((albedo_tensor), str(constants.AIRLIGHT_VERSION) + str(constants.ITERATION) + " Training Albedo images")
        self.visdom_reporter.plot_image((rgb_tensor), str(constants.AIRLIGHT_VERSION) + str(constants.ITERATION) + " Training Styled images")

    def load_saved_state(self, checkpoint):
        self.D_A.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + "A"])
        self.optimizerDA.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + constants.OPTIMIZER_KEY + "A"])
        self.schedulerDA.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + "scheduler" + "A"])

        self.D_B.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + "B"])
        self.optimizerDB.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + constants.OPTIMIZER_KEY + "B"])
        self.schedulerDB.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + "scheduler" + "B"])


    def save_states(self, epoch, iteration):
        save_dict = {'epoch': epoch, 'iteration': iteration}
        netDA_state_dict = self.D_A.state_dict()
        netDB_state_dict = self.D_B.state_dict()

        optimizerDA_state_dict = self.optimizerDA.state_dict()
        schedulerDA_state_dict = self.schedulerDA.state_dict()
        optimizerDB_state_dict = self.optimizerDB.state_dict()
        schedulerDB_state_dict = self.schedulerDB.state_dict()

        save_dict[constants.DISCRIMINATOR_KEY + "A"] = netDA_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + constants.OPTIMIZER_KEY + "A"] = optimizerDA_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + "scheduler" + "A"] = schedulerDA_state_dict

        save_dict[constants.DISCRIMINATOR_KEY + "B"] = netDB_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + constants.OPTIMIZER_KEY + "B"] = optimizerDB_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + "scheduler" + "B"] = schedulerDB_state_dict

        torch.save(save_dict, constants.AIRLIGHT_ESTIMATOR_CHECKPATH)
        print("Saved model state: %s Epoch: %d" % (len(save_dict), (epoch + 1)))

        # clear plots to avoid potential sudden jumps in visualization due to unstable gradients during early training
        if (epoch % 10 == 0):
            self.losses_dict[self.AIRLOSS_A_KEY].clear()
            self.losses_dict[self.AIRLOSS_B_KEY].clear()
            self.losses_dict[self.AIRLOSS_A_KEY_TEST].clear()
            self.losses_dict[self.AIRLOSS_B_KEY_TEST].clear()
