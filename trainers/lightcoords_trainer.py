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
from loaders import dataset_loader

class LightCoordsTrainer:
    def __init__(self, gpu_device, lr=0.0002):
        self.gpu_device = gpu_device
        self.lr = lr
        self.D_A = dh.LightCoordsEstimator_V2(input_nc = 3, num_layers = 4).to(self.gpu_device)

        self.visdom_reporter = plot_utils.VisdomReporter()
        self.initialize_dict()

        self.optimizerDA = torch.optim.Adam(itertools.chain(self.D_A.parameters()), lr=self.lr)
        self.schedulerDA = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizerDA, patience=1000, threshold=0.00005)

    def initialize_dict(self):
        # what to store in visdom?
        self.losses_dict = {}
        self.losses_dict[constants.D_OVERALL_LOSS_KEY] = []

        self.caption_dict = {}
        self.caption_dict[constants.D_OVERALL_LOSS_KEY] = "D(A) loss per iteration"


    def network_loss(self, pred, target):
        loss = nn.MSELoss()
        return loss(pred, target)

    def update_penalties(self, loss_weight, comments):
        # what penalties to use for losses?
        self.loss_weight = loss_weight

        # save hyperparameters for bookkeeping
        HYPERPARAMS_PATH = "checkpoint/" + constants.LIGHTS_ESTIMATOR_VERSION + "_" + constants.ITERATION + ".config"
        with open(HYPERPARAMS_PATH, "w") as f:
            print("Version: ", constants.LIGHTCOORDS_ESTIMATOR_CHECKPATH, file=f)
            print("Comment: ", comments, file=f)
            print("Learning rate for D: ", str(self.lr), file=f)
            print("====================================", file=f)
            print("Airlight weight: ", str(self.loss_weight), file=f)

    def train(self, rgb_tensor, light_coords_tensor):
        self.D_A.train()
        self.optimizerDA.zero_grad()

        #train
        D_A_loss = self.network_loss(self.D_A(rgb_tensor), light_coords_tensor) * self.loss_weight
        self.losses_dict[constants.D_OVERALL_LOSS_KEY].append(D_A_loss.item())
        D_A_loss.backward()

        self.optimizerDA.step()
        self.schedulerDA.step(D_A_loss)

    def visdom_report(self, iteration, train_tensor):
        # report to visdom
        self.visdom_reporter.plot_finegrain_loss("Train loss", iteration, self.losses_dict, self.caption_dict)
        self.visdom_reporter.plot_image((train_tensor), "Training RGB images")

    def load_saved_state(self, checkpoint):
        self.D_A.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY])
        self.optimizerDA.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + constants.OPTIMIZER_KEY])
        self.schedulerDA.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + "scheduler"])


    def save_states(self, epoch, iteration):
        save_dict = {'epoch': epoch, 'iteration': iteration}
        netDA_state_dict = self.D_A.state_dict()

        optimizerDA_state_dict = self.optimizerDA.state_dict()
        schedulerDA_state_dict = self.schedulerDA.state_dict()

        save_dict[constants.DISCRIMINATOR_KEY] = netDA_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + constants.OPTIMIZER_KEY] = optimizerDA_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + "scheduler"] = schedulerDA_state_dict

        torch.save(save_dict, constants.LIGHTCOORDS_ESTIMATOR_CHECKPATH)
        print("Saved model state: %s Epoch: %d" % (len(save_dict), (epoch + 1)))
