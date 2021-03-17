# -*- coding: utf-8 -*-

import os
from model import dehaze_discriminator as dh
import constants
import torch
import itertools
import torch.nn as nn
from utils import plot_utils

class AirlightTrainer:
    def __init__(self, gan_version, gan_iteration, gpu_device, lr=0.0002):
        self.gpu_device = gpu_device
        self.lr = lr
        self.gan_version = gan_version
        self.gan_iteration = gan_iteration
        self.D_A = dh.AirlightEstimator(input_nc=3, num_layers = 2).to(self.gpu_device)
        self.D_B = dh.AirlightEstimator_V2(input_nc=3, num_layers = 2).to(self.gpu_device)

        self.visdom_reporter = plot_utils.VisdomReporter()
        self.initialize_dict()

        self.optimizerDA = torch.optim.Adam(itertools.chain(self.D_A.parameters()), lr=self.lr)
        self.schedulerDA = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizerDA, patience=1000, threshold=0.00005)

        self.optimizerDB = torch.optim.Adam(itertools.chain(self.D_B.parameters()), lr=self.lr)
        self.schedulerDB = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizerDB, patience=1000, threshold=0.00005)



    def initialize_dict(self):
        self.AIRLOSS_B_KEY = "AIRLOSS_B_KEY"

        # what to store in visdom?
        self.losses_dict = {}
        self.losses_dict[constants.D_OVERALL_LOSS_KEY] = []
        self.losses_dict[self.AIRLOSS_B_KEY] = []

        self.caption_dict = {}
        self.caption_dict[constants.D_OVERALL_LOSS_KEY] = "D(A) loss per iteration"
        self.caption_dict[self.AIRLOSS_B_KEY] = "D(B) loss per iteration"


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

    def train(self, rgb_tensor, light_coords_tensor, airlight_tensor):
        self.D_A.train()
        self.optimizerDA.zero_grad()

        #train first model
        airlight_tensor = torch.unsqueeze(airlight_tensor, 1)
        D_A_loss = self.network_loss(self.D_A(rgb_tensor), airlight_tensor) * self.loss_weight
        self.losses_dict[constants.D_OVERALL_LOSS_KEY].append(D_A_loss.item())

        #train second model
        self.D_B.train()
        self.optimizerDB.zero_grad()
        D_B_loss = self.network_loss(self.D_B(rgb_tensor, light_coords_tensor), airlight_tensor) * self.loss_weight
        self.losses_dict[self.AIRLOSS_B_KEY].append(D_B_loss.item())

        errD = D_A_loss + D_B_loss
        errD.backward()

        self.optimizerDA.step()
        self.schedulerDA.step(errD)
        self.optimizerDB.step()
        self.schedulerDB.step(errD)

    def visdom_report(self, iteration, train_tensor):
        # report to visdom
        self.visdom_reporter.plot_finegrain_loss("Train loss", iteration, self.losses_dict, self.caption_dict)
        self.visdom_reporter.plot_image((train_tensor), "Training RGB images")

    def load_saved_state(self, iteration, checkpoint, model_key, optimizer_key):
        self.gan_iteration = iteration
        self.D_A.load_state_dict(checkpoint[model_key + "A"])
        self.optimizerDA.load_state_dict(checkpoint[model_key + optimizer_key + "A"])
        self.schedulerDA.load_state_dict(checkpoint[model_key + "scheduler" + "A"])

        self.D_B.load_state_dict(checkpoint[model_key + "B"])
        self.optimizerDB.load_state_dict(checkpoint[model_key + optimizer_key + "B"])
        self.schedulerDB.load_state_dict(checkpoint[model_key + "scheduler" + "B"])


    def save_states(self, epoch, iteration, model_key, optimizer_key):
        save_dict = {'epoch': epoch, 'iteration': iteration}
        netDA_state_dict = self.D_A.state_dict()
        netDB_state_dict = self.D_B.state_dict()

        optimizerDA_state_dict = self.optimizerDA.state_dict()
        schedulerDA_state_dict = self.schedulerDA.state_dict()
        optimizerDB_state_dict = self.optimizerDB.state_dict()
        schedulerDB_state_dict = self.schedulerDB.state_dict()

        save_dict[model_key + "A"] = netDA_state_dict
        save_dict[model_key + optimizer_key + "A"] = optimizerDA_state_dict
        save_dict[model_key + "scheduler" + "A"] = schedulerDA_state_dict

        save_dict[model_key + "B"] = netDB_state_dict
        save_dict[model_key + optimizer_key + "B"] = optimizerDB_state_dict
        save_dict[model_key + "scheduler" + "B"] = schedulerDB_state_dict

        torch.save(save_dict, constants.AIRLIGHT_ESTIMATOR_CHECKPATH)
        print("Saved model state: %s Epoch: %d" % (len(save_dict), (epoch + 1)))
