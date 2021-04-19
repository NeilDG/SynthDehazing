# -*- coding: utf-8 -*-

import os

import numpy as np

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
        self.A1 = dh.AirlightEstimator_V1(input_nc=3, downsampling_layers = 3, residual_blocks = 7, add_mean = False).to(self.gpu_device)
        self.A2 = dh.AirlightEstimator_V1(input_nc=6, downsampling_layers = 3, residual_blocks = 7, add_mean = False).to(self.gpu_device)
        #self.A3 = dh.AirlightEstimator_V1(input_nc=3, downsampling_layers=3, residual_blocks=5, add_mean=True).to(self.gpu_device)
        #self.A4 = dh.AirlightEstimator_V1(input_nc=6, downsampling_layers=3, residual_blocks=5, add_mean=True).to(self.gpu_device)

        self.visdom_reporter = plot_utils.VisdomReporter()
        self.initialize_dict()

        self.optimizerDA = torch.optim.Adam(self.A1.parameters(), lr=self.lr)
        self.schedulerDA = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizerDA, patience=100000 / constants.batch_size, threshold=0.00005)
        self.optimizerDB = torch.optim.Adam(self.A2.parameters(), lr=self.lr)
        self.schedulerDB = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizerDB, patience=100000 / constants.batch_size, threshold=0.00005)
        # self.optimizerDC = torch.optim.Adam(self.A3.parameters(), lr=self.lr)
        # self.schedulerDC = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizerDC, patience=100000 / constants.batch_size, threshold=0.00005)
        # self.optimizerDD = torch.optim.Adam(self.A4.parameters(), lr=self.lr)
        # self.schedulerDD = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizerDD, patience=100000 / constants.batch_size, threshold=0.00005)

        self.fp16_scalers = [amp.GradScaler(),
                             amp.GradScaler(),
                             amp.GradScaler(),
                             amp.GradScaler()]

    def initialize_dict(self):
        self.AIRLOSS_A_KEY = "AIRLOSS_A_KEY"
        self.AIRLOSS_B_KEY = "AIRLOSS_B_KEY"
        self.AIRLOSS_A_KEY_TEST = "AIRLOSS_A_KEY_TEST"
        self.AIRLOSS_B_KEY_TEST = "AIRLOSS_b_KEY_TEST"

        self.AIRLOSS_C_KEY = "AIRLOSS_C_KEY"
        self.AIRLOSS_C_KEY_TEST = "AIRLOSS_C_KEY_TEST"
        self.AIRLOSS_D_KEY = "AIRLOSS_D_KEY"
        self.AIRLOSS_D_KEY_TEST = "AIRLOSS_D_KEY_TEST"

        # what to store in visdom?
        self.losses_dict = {}
        self.losses_dict[self.AIRLOSS_A_KEY] = []
        self.losses_dict[self.AIRLOSS_B_KEY] = []
        self.losses_dict[self.AIRLOSS_A_KEY_TEST] = []
        self.losses_dict[self.AIRLOSS_B_KEY_TEST] = []
        self.losses_dict[self.AIRLOSS_C_KEY] = []
        self.losses_dict[self.AIRLOSS_D_KEY] = []
        self.losses_dict[self.AIRLOSS_C_KEY_TEST] = []
        self.losses_dict[self.AIRLOSS_D_KEY_TEST] = []

        self.caption_dict = {}
        self.caption_dict[self.AIRLOSS_A_KEY] = "Airlight loss per iteration"
        self.caption_dict[self.AIRLOSS_B_KEY] = "Airlight (Albedo) loss per iteration"
        self.caption_dict[self.AIRLOSS_A_KEY_TEST] = "Airlight - testloss per iteration"
        self.caption_dict[self.AIRLOSS_B_KEY_TEST] = "Airlight (Albedo) - test loss per iteration"
        self.caption_dict[self.AIRLOSS_C_KEY] = "Airlight + Add Mean loss per iteration"
        self.caption_dict[self.AIRLOSS_D_KEY] = "Airlight (Albedo) + Add Mean loss per iteration"
        self.caption_dict[self.AIRLOSS_C_KEY_TEST] = "Airlight + Add Mean - testloss per iteration"
        self.caption_dict[self.AIRLOSS_D_KEY_TEST] = "Airlight (Albedo) + Add Mean - test loss per iteration"


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

    def get_current_lr(self, optimizer, group_idx, parameter_idx):
        # Adam has different learning rates for each paramter. So we need to pick the
        # group and paramter first.
        group = optimizer.param_groups[group_idx]
        p = group['params'][parameter_idx]

        beta1, _ = group['betas']
        state = optimizer.state[p]

        bias_correction1 = 1 - beta1 ** state['step']
        current_lr = group['lr'] / bias_correction1
        return current_lr

    def train_a1(self, rgb_tensor, airlight_tensor):
        with amp.autocast():
            airlight_tensor = torch.unsqueeze(airlight_tensor, 1)
            self.optimizerDA.zero_grad()
            self.A1.train()
            errD = self.network_loss(self.A1(rgb_tensor), airlight_tensor) * self.loss_weight
            self.losses_dict[self.AIRLOSS_A_KEY].append(errD.item() / self.loss_weight)

            self.fp16_scalers[0].scale(errD).backward()
            self.fp16_scalers[0].step(self.optimizerDA)
            self.schedulerDA.step(errD)

            self.fp16_scalers[0].update()

    def train_a2(self, albedo_tensor, rgb_tensor, airlight_tensor):
        with amp.autocast():
            airlight_tensor = torch.unsqueeze(airlight_tensor, 1)
            self.optimizerDB.zero_grad()
            self.A2.train()
            errD = self.network_loss(self.A2(torch.cat([rgb_tensor, albedo_tensor], 1)), airlight_tensor) * self.loss_weight
            self.losses_dict[self.AIRLOSS_B_KEY].append(errD.item() / self.loss_weight)

            self.fp16_scalers[1].scale(errD).backward()
            self.fp16_scalers[1].step(self.optimizerDB)
            self.schedulerDB.step(errD)

            self.fp16_scalers[1].update()

    # def train_a3(self, rgb_tensor, airlight_tensor):
    #     with amp.autocast():
    #         airlight_tensor = torch.unsqueeze(airlight_tensor, 1)
    #         self.optimizerDC.zero_grad()
    #         self.A3.train()
    #         errD = self.network_loss(self.A3(rgb_tensor), airlight_tensor) * self.loss_weight
    #         self.losses_dict[self.AIRLOSS_C_KEY].append(errD.item())
    #
    #         self.fp16_scalers[2].scale(errD).backward()
    #         self.fp16_scalers[2].step(self.optimizerDC)
    #         self.schedulerDC.step(errD)
    #
    #         self.fp16_scalers[2].update()
    #
    # def train_a4(self, albedo_tensor, rgb_tensor, airlight_tensor):
    #     with amp.autocast():
    #         airlight_tensor = torch.unsqueeze(airlight_tensor, 1)
    #         self.optimizerDD.zero_grad()
    #         self.A4.train()
    #         errD = self.network_loss(self.A4(torch.cat([rgb_tensor, albedo_tensor], 1)), airlight_tensor) * self.loss_weight
    #         self.losses_dict[self.AIRLOSS_D_KEY].append(errD.item())
    #
    #         self.fp16_scalers[3].scale(errD).backward()
    #         self.fp16_scalers[3].step(self.optimizerDD)
    #         self.schedulerDD.step(errD)
    #
    #         self.fp16_scalers[3].update()

    # def train(self, albedo_tensor, rgb_tensor, airlight_tensor):
    #     with amp.autocast():
    #         self.optimizer.zero_grad()
    #
    #         self.A1.train()
    #         #train first model
    #         airlight_tensor = torch.unsqueeze(airlight_tensor, 1)
    #         D_A1_loss = self.network_loss(self.A1(rgb_tensor), airlight_tensor) * self.loss_weight
    #
    #         self.losses_dict[self.AIRLOSS_A_KEY].append(D_A1_loss.item())
    #
    #         #train second model
    #         self.A2.train()
    #         D_A2_loss = self.network_loss(self.A2(torch.cat([rgb_tensor, albedo_tensor], 1)), airlight_tensor) * self.loss_weight
    #
    #         self.losses_dict[self.AIRLOSS_B_KEY].append(D_A2_loss.item())
    #
    #         # train third model
    #         self.A3.train()
    #         D_A3_loss = self.network_loss(self.A3(rgb_tensor), airlight_tensor) * self.loss_weight
    #         self.losses_dict[self.AIRLOSS_C_KEY].append(D_A3_loss.item())
    #
    #         # train fourth  model
    #         self.A4.train()
    #         D_A4_loss = self.network_loss(self.A4(torch.cat([rgb_tensor, albedo_tensor], 1)), airlight_tensor) * self.loss_weight
    #         self.losses_dict[self.AIRLOSS_D_KEY].append(D_A4_loss.item())
    #
    #         errD = D_A1_loss + D_A2_loss + D_A3_loss + D_A4_loss
    #
    #         self.fp16_scaler.scale(errD).backward()
    #         self.fp16_scaler.step(self.optimizer)
    #         self.scheduler.step(errD)
    #
    #         self.fp16_scaler.update()

    def test(self, albedo_tensor, rgb_tensor, airlight_tensor):
        self.A1.eval()
        self.A2.eval()
        # self.A3.eval()
        # self.A4.eval()

        with torch.no_grad(), amp.autocast():
            airlight_tensor = torch.unsqueeze(airlight_tensor, 1)
            D_A1_loss = self.network_loss(self.A1(rgb_tensor), airlight_tensor) * self.loss_weight
            self.losses_dict[self.AIRLOSS_A_KEY_TEST].append(D_A1_loss.item())

            D_A2_loss = self.network_loss(self.A2(torch.cat([rgb_tensor, albedo_tensor], 1)), airlight_tensor) * self.loss_weight
            self.losses_dict[self.AIRLOSS_B_KEY_TEST].append(D_A2_loss.item())

            # D_A3_loss = self.network_loss(self.A3(rgb_tensor), airlight_tensor) * self.loss_weight
            # self.losses_dict[self.AIRLOSS_C_KEY_TEST].append(D_A3_loss.item())
            #
            # D_A4_loss = self.network_loss(self.A4(torch.cat([rgb_tensor, albedo_tensor], 1)), airlight_tensor) * self.loss_weight
            # self.losses_dict[self.AIRLOSS_D_KEY_TEST].append(D_A4_loss.item())


    def visdom_report(self, iteration, albedo_tensor, rgb_tensor):
        # report to visdom
        self.visdom_reporter.plot_airlight_comparison(self.AIRLOSS_A_KEY, iteration,
                                                      [self.losses_dict[self.AIRLOSS_A_KEY], self.losses_dict[self.AIRLOSS_B_KEY]],
                                                      [str(constants.AIRLIGHT_VERSION) + str(constants.ITERATION) + " Airlight (Standard) - Train Loss", "Airloss (Albedo) - Train loss"])

        self.visdom_reporter.plot_airlight_comparison(self.AIRLOSS_B_KEY, iteration,
                                                      [self.losses_dict[self.AIRLOSS_A_KEY_TEST], self.losses_dict[self.AIRLOSS_B_KEY_TEST]],
                                                      [str(constants.AIRLIGHT_VERSION) + str(constants.ITERATION) + " Airlight (Standard) - Test Loss", "Airloss (Albedo) - Test loss"])

        # self.visdom_reporter.plot_airlight_comparison(self.AIRLOSS_C_KEY, iteration,
        #                                               [self.losses_dict[self.AIRLOSS_C_KEY], self.losses_dict[self.AIRLOSS_D_KEY]],
        #                                               [str(constants.AIRLIGHT_VERSION) + str(constants.ITERATION) + " Airlight (Standard) + Added Mean - Train Loss", "Airloss (Albedo) - Train loss"])
        #
        # self.visdom_reporter.plot_airlight_comparison(self.AIRLOSS_D_KEY, iteration,
        #                                               [self.losses_dict[self.AIRLOSS_C_KEY_TEST], self.losses_dict[self.AIRLOSS_D_KEY_TEST]],
        #                                               [str(constants.AIRLIGHT_VERSION) + str(constants.ITERATION) + " Airlight (Standard + Added Mean - Test Loss", "Airloss (Albedo) - Test loss"])

        self.visdom_reporter.plot_image((albedo_tensor), str(constants.AIRLIGHT_VERSION) + str(constants.ITERATION) + " Training Albedo images")
        self.visdom_reporter.plot_image((rgb_tensor), str(constants.AIRLIGHT_VERSION) + str(constants.ITERATION) + " Training Styled images")

    def load_saved_state(self, checkpoint):
        self.A1.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + "A"])
        self.A2.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + "B"])
        # self.A3.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + "C"])
        # self.A4.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + "D"])

        self.optimizerDA.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + constants.OPTIMIZER_KEY + "A"])
        self.optimizerDB.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + constants.OPTIMIZER_KEY + "B"])
        # self.optimizerDC.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + constants.OPTIMIZER_KEY + "C"])
        # self.optimizerDD.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + constants.OPTIMIZER_KEY + "D"])
        self.schedulerDA.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + "scheduler" + "A"])
        self.schedulerDB.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + "scheduler" + "B"])
        # self.schedulerDC.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + "scheduler" + "C"])
        # self.schedulerDD.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + "scheduler" + "D"])


    def save_states(self, epoch, iteration):
        save_dict = {'epoch': epoch, 'iteration': iteration}

        save_dict[constants.DISCRIMINATOR_KEY + "A"] = self.A1.state_dict()
        save_dict[constants.DISCRIMINATOR_KEY + "B"] = self.A2.state_dict()
        # save_dict[constants.DISCRIMINATOR_KEY + "C"] = self.A3.state_dict()
        # save_dict[constants.DISCRIMINATOR_KEY + "D"] = self.A4.state_dict()

        save_dict[constants.DISCRIMINATOR_KEY + constants.OPTIMIZER_KEY + "A"] = self.optimizerDA.state_dict()
        save_dict[constants.DISCRIMINATOR_KEY + constants.OPTIMIZER_KEY + "B"] = self.optimizerDB.state_dict()
        # save_dict[constants.DISCRIMINATOR_KEY + constants.OPTIMIZER_KEY + "C"] = self.optimizerDC.state_dict()
        # save_dict[constants.DISCRIMINATOR_KEY + constants.OPTIMIZER_KEY + "D"] = self.optimizerDD.state_dict()
        save_dict[constants.DISCRIMINATOR_KEY + "scheduler" + "A"] = self.schedulerDA.state_dict()
        save_dict[constants.DISCRIMINATOR_KEY + "scheduler" + "B"] = self.schedulerDB.state_dict()
        # save_dict[constants.DISCRIMINATOR_KEY + "scheduler" + "C"] = self.schedulerDC.state_dict()
        # save_dict[constants.DISCRIMINATOR_KEY + "scheduler" + "D"] = self.schedulerDD.state_dict()

        torch.save(save_dict, constants.AIRLIGHT_ESTIMATOR_CHECKPATH)
        print("Saved model state: %s Epoch: %d" % (len(save_dict), (epoch + 1)))

        # clear plots to avoid potential sudden jumps in visualization due to unstable gradients during early training
        if (epoch % 20 == 0):
            self.losses_dict[self.AIRLOSS_A_KEY].clear()
            self.losses_dict[self.AIRLOSS_B_KEY].clear()
            self.losses_dict[self.AIRLOSS_A_KEY_TEST].clear()
            self.losses_dict[self.AIRLOSS_B_KEY_TEST].clear()
            self.losses_dict[self.AIRLOSS_C_KEY].clear()
            self.losses_dict[self.AIRLOSS_D_KEY].clear()
            self.losses_dict[self.AIRLOSS_C_KEY_TEST].clear()
            self.losses_dict[self.AIRLOSS_D_KEY_TEST].clear()
