# -*- coding: utf-8 -*-

import os
from model import dehaze_discriminator as dh
import constants
import torch
import random
import itertools
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision.utils as vutils
from utils import logger
from utils import plot_utils
from utils import tensor_utils
import kornia
from custom_losses import rmse_log_loss

class AirlightTrainer:
    def __init__(self, gan_version, gan_iteration, gpu_device, lr=0.0002):
        self.gpu_device = gpu_device
        self.lr = lr
        self.gan_version = gan_version
        self.gan_iteration = gan_iteration
        self.D_A = dh.AirlightEstimator(input_nc=3, num_layers = 2).to(self.gpu_device)

        self.visdom_reporter = plot_utils.VisdomReporter()
        self.initialize_dict()

        self.optimizerD = torch.optim.Adam(itertools.chain(self.D_A.parameters()), lr=self.lr)
        self.schedulerD = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizerD, patience=1000, threshold=0.00005)

    def initialize_dict(self):
        # what to store in visdom?
        self.losses_dict = {}
        self.losses_dict[constants.D_OVERALL_LOSS_KEY] = []

        self.caption_dict = {}
        self.caption_dict[constants.D_OVERALL_LOSS_KEY] = "D loss per iteration"


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

    def train(self, rgb_tensor, airlight_tensor):
        self.D_A.train()
        self.optimizerD.zero_grad()

        airlight_tensor = torch.unsqueeze(airlight_tensor, 1)
        D_A_loss = self.network_loss(self.D_A(rgb_tensor), airlight_tensor) * self.loss_weight
        errD = D_A_loss
        errD.backward()
        self.optimizerD.step()
        self.schedulerD.step(errD)

        # what to put to losses dict for visdom reporting?
        self.losses_dict[constants.D_OVERALL_LOSS_KEY].append(errD.item())

    def visdom_report(self, iteration, train_tensor):
        # report to visdom
        self.visdom_reporter.plot_finegrain_loss("Train loss", iteration, self.losses_dict, self.caption_dict)
        self.visdom_reporter.plot_image((train_tensor), "Training RGB images")

    def load_saved_state(self, iteration, checkpoint, model_key, optimizer_key):
        self.gan_iteration = iteration
        self.D_A.load_state_dict(checkpoint[model_key])
        self.optimizerD.load_state_dict(checkpoint[model_key + optimizer_key])
        self.schedulerD.load_state_dict(checkpoint[model_key + "scheduler"])


    def save_states(self, epoch, iteration, path, model_key, optimizer_key):
        save_dict = {'epoch': epoch, 'iteration': iteration}
        netDA_state_dict = self.D_A.state_dict()
        optimizerD_state_dict = self.optimizerD.state_dict()
        schedulerD_state_dict = self.schedulerD.state_dict()
        save_dict[model_key] = netDA_state_dict
        save_dict[model_key + optimizer_key] = optimizerD_state_dict
        save_dict[model_key + "scheduler"] = schedulerD_state_dict

        torch.save(save_dict, path)
        print("Saved model state: %s Epoch: %d" % (len(save_dict), (epoch + 1)))
