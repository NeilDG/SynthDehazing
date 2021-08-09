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
    def __init__(self, gpu_device, batch_size, num_layers, lr=0.0002):
        self.gpu_device = gpu_device
        self.lr = lr
        self.A1 = dh.AirlightEstimator_Residual(num_channels = 3, out_features = 3, num_layers = num_layers).to(self.gpu_device)

        self.visdom_reporter = plot_utils.VisdomReporter()
        self.initialize_dict()

        self.optimizerDA = torch.optim.Adam(self.A1.parameters(), lr=self.lr)
        self.schedulerDA = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizerDA, patience=100000 / batch_size, threshold=0.00005)

        self.fp16_scalers = [amp.GradScaler(),
                             amp.GradScaler()]

        self.early_stop_tolerance = 40
        self.stop_counter = 0
        self.last_metric = 10000.0
        self.stop_condition_met = False

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
            #airlight_tensor = torch.unsqueeze(airlight_tensor, 1)
            self.optimizerDA.zero_grad()
            self.A1.train()

            errD = self.network_loss(self.A1(rgb_tensor), airlight_tensor) * self.loss_weight
            self.losses_dict[self.AIRLOSS_A_KEY].append(errD.item() / self.loss_weight)

            self.fp16_scalers[0].scale(errD).backward()
            self.fp16_scalers[0].step(self.optimizerDA)
            self.schedulerDA.step(errD)

            self.fp16_scalers[0].update()

    def test(self, epoch, rgb_tensor, airlight_tensor):
        self.A1.eval()
        with torch.no_grad(), amp.autocast():
            D_A1_loss = self.network_loss(self.A1(rgb_tensor), airlight_tensor) * self.loss_weight
            self.losses_dict[self.AIRLOSS_A_KEY_TEST].append(D_A1_loss.item())

        #early stopping mechanism
        if(self.last_metric < D_A1_loss and epoch > 10):
            self.stop_counter += 1
        elif(self.last_metric >= D_A1_loss):
            self.last_metric = D_A1_loss
            self.stop_counter = 0
            print("Early stopping mechanism reset. Best metric is now ", self.last_metric)

        if (self.stop_counter == self.early_stop_tolerance):
            self.stop_condition_met = True
            print("Met stopping condition with best metric of: ", self.last_metric, ". Latest metric: ", D_A1_loss)

    def did_stop_condition_met(self):
        return self.stop_condition_met

    def visdom_report(self, iteration, rgb_tensor):
        # report to visdom
        self.visdom_reporter.plot_airlight_comparison(self.AIRLOSS_A_KEY, iteration,
                                                      [self.losses_dict[self.AIRLOSS_A_KEY], self.losses_dict[self.AIRLOSS_B_KEY]],
                                                      [str(constants.AIRLIGHT_VERSION) + str(constants.ITERATION) + " Airlight (Standard) - Train Loss", "Airloss (Albedo) - Train loss"])
        self.visdom_reporter.plot_image((rgb_tensor), str(constants.AIRLIGHT_VERSION) + str(constants.ITERATION) + " Training Styled images")

    def load_saved_state(self, checkpoint):
        self.A1.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + "A"])
        self.optimizerDA.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + constants.OPTIMIZER_KEY + "A"])
        self.schedulerDA.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + "scheduler" + "A"])


    def save_states(self, epoch, iteration):
        save_dict = {'epoch': epoch, 'iteration': iteration}

        save_dict[constants.DISCRIMINATOR_KEY + "A"] = self.A1.state_dict()

        save_dict[constants.DISCRIMINATOR_KEY + constants.OPTIMIZER_KEY + "A"] = self.optimizerDA.state_dict()
        save_dict[constants.DISCRIMINATOR_KEY + "scheduler" + "A"] = self.schedulerDA.state_dict()

        torch.save(save_dict, constants.AIRLIGHT_ESTIMATOR_CHECKPATH)
        print("Saved model state: %s Epoch: %d" % (len(save_dict), (epoch + 1)))

        #clear plots to avoid potential sudden jumps in visualization due to unstable gradients during early training
        if (epoch % 20 == 0):
            self.losses_dict[self.AIRLOSS_A_KEY].clear()
            self.losses_dict[self.AIRLOSS_B_KEY].clear()
            self.losses_dict[self.AIRLOSS_A_KEY_TEST].clear()
            self.losses_dict[self.AIRLOSS_B_KEY_TEST].clear()
            self.losses_dict[self.AIRLOSS_C_KEY].clear()
            self.losses_dict[self.AIRLOSS_D_KEY].clear()
            self.losses_dict[self.AIRLOSS_C_KEY_TEST].clear()
            self.losses_dict[self.AIRLOSS_D_KEY_TEST].clear()
