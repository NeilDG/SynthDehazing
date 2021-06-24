# -*- coding: utf-8 -*-

import os

import numpy as np

from model import vanilla_cycle_gan as cg
from model import style_transfer_gan as sg
from model import unet_gan as un
from model import dehaze_discriminator as dh
import constants
import torch
import itertools
import torch.nn as nn
from utils import plot_utils
import torch.cuda.amp as amp
import kornia

class AirlightGenTrainer:
    def __init__(self, gpu_device, batch_size, is_unet, g_lr = 0.0002, d_lr = 0.0002):
        self.gpu_device = gpu_device
        self.g_lr = g_lr
        self.d_lr = d_lr

        if (is_unet == 1):
            self.G_A = un.UnetGenerator(input_nc=3, output_nc=3, num_downs=8).to(self.gpu_device)
        else:
            self.G_A = cg.Generator(input_nc=3, output_nc=3, n_residual_blocks=10).to(self.gpu_device)

        self.D_A = dh.Discriminator(input_nc=3).to(self.gpu_device)

        self.visdom_reporter = plot_utils.VisdomReporter()
        self.initialize_dict()

        self.optimizerG = torch.optim.Adam(itertools.chain(self.G_A.parameters()), lr=self.g_lr)
        self.optimizerD = torch.optim.Adam(itertools.chain(self.D_A.parameters()), lr=self.d_lr)
        self.schedulerG = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizerG, patience=100000 / batch_size, threshold=0.00005)
        self.schedulerD = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizerD, patience=100000 / batch_size, threshold=0.00005)

        self.fp16_scaler = amp.GradScaler()  # for automatic mixed precision

    def initialize_dict(self):
        # what to store in visdom?
        self.losses_dict = {}
        self.losses_dict[constants.G_LOSS_KEY] = []
        self.losses_dict[constants.D_OVERALL_LOSS_KEY] = []
        self.losses_dict[constants.LIKENESS_LOSS_KEY] = []
        self.losses_dict[constants.EDGE_LOSS_KEY] = []
        self.losses_dict[constants.G_ADV_LOSS_KEY] = []
        self.losses_dict[constants.D_A_FAKE_LOSS_KEY] = []
        self.losses_dict[constants.D_A_REAL_LOSS_KEY] = []

        self.caption_dict = {}
        self.caption_dict[constants.G_LOSS_KEY] = "G loss per iteration"
        self.caption_dict[constants.D_OVERALL_LOSS_KEY] = "D loss per iteration"
        self.caption_dict[constants.LIKENESS_LOSS_KEY] = "Likeness loss per iteration"
        self.caption_dict[constants.EDGE_LOSS_KEY] = "Edge loss per iteration"
        self.caption_dict[constants.G_ADV_LOSS_KEY] = "G adv loss per iteration"
        self.caption_dict[constants.D_A_FAKE_LOSS_KEY] = "D(A) fake loss per iteration"
        self.caption_dict[constants.D_A_REAL_LOSS_KEY] = "D(A) real loss per iteration"

    def update_penalties(self, adv_weight, likeness_weight, edge_weight, comments):
        # what penalties to use for losses?
        self.adv_weight = adv_weight
        self.likeness_weight = likeness_weight
        self.edge_weight = edge_weight

        # save hyperparameters for bookkeeping
        HYPERPARAMS_PATH = "checkpoint/" + constants.AIRLIGHT_GEN_VERSION + "_" + constants.ITERATION + ".config"
        with open(HYPERPARAMS_PATH, "w") as f:
            print("Version: ", constants.AIRLIGHT_GEN_CHECKPATH, file=f)
            print("Comment: ", comments, file=f)
            print("Learning rate for G: ", str(self.g_lr), file=f)
            print("Learning rate for D: ", str(self.d_lr), file=f)
            print("====================================", file=f)
            print("Adv weight: ", str(self.adv_weight), file=f)
            print("Likeness weight: ", str(self.likeness_weight), file=f)
            print("Edge weight: ", str(self.edge_weight), file=f)

    def adversarial_loss(self, pred, target):
        loss = nn.BCEWithLogitsLoss()
        return loss(pred, target)

    def likeness_loss(self, pred, target):
        loss = nn.L1Loss()
        return loss(pred, target)

    def edge_loss(self, pred, target):
        loss = nn.L1Loss()
        pred_grad = kornia.filters.spatial_gradient(pred)
        target_grad = kornia.filters.spatial_gradient(target)

        return loss(pred_grad, target_grad)

    def train(self, rgb_tensor, depth_tensor):
        with amp.autocast():
            depth_like = self.G_A(rgb_tensor)
            # rgb_like = self.G_B(depth_tensor)

            self.D_A.train()
            # self.D_B.train()
            self.optimizerD.zero_grad()

            prediction = self.D_A(depth_tensor)
            real_tensor = torch.ones_like(prediction)
            fake_tensor = torch.zeros_like(prediction)

            D_A_real_loss = self.adversarial_loss(self.D_A(depth_tensor), real_tensor) * self.adv_weight
            D_A_fake_loss = self.adversarial_loss(self.D_A(depth_like.detach()), fake_tensor) * self.adv_weight

            # prediction = self.D_B(rgb_tensor)
            # real_tensor = torch.ones_like(prediction)
            # fake_tensor = torch.zeros_like(prediction)

            # D_B_real_loss = self.adversarial_loss(self.D_B(rgb_tensor), real_tensor) * self.adv_weight
            # D_B_fake_loss = self.adversarial_loss(self.D_B(rgb_like.detach()), fake_tensor) * self.adv_weight

            errD = D_A_real_loss + D_A_fake_loss
            if (self.fp16_scaler.scale(errD).item() > 0.1):
                self.fp16_scaler.scale(errD).backward()
                self.fp16_scaler.step(self.optimizerD)
                self.schedulerD.step(errD)

            self.G_A.train()
            # self.G_B.train()
            self.optimizerG.zero_grad()

            # print("Shape: ", np.shape(rgb_tensor), np.shape(depth_tensor))
            A_likeness_loss = self.likeness_loss(self.G_A(rgb_tensor), depth_tensor) * self.likeness_weight
            A_edge_loss = self.edge_loss(self.G_A(rgb_tensor), depth_tensor) * self.edge_weight

            prediction = self.D_A(self.G_A(rgb_tensor))
            real_tensor = torch.ones_like(prediction)
            A_adv_loss = self.adversarial_loss(prediction, real_tensor) * self.adv_weight

            errG = A_likeness_loss + A_adv_loss + A_edge_loss
            self.fp16_scaler.scale(errG).backward()
            self.fp16_scaler.step(self.optimizerG)
            self.schedulerG.step(errG)

            self.fp16_scaler.update()

            # what to put to losses dict for visdom reporting?
            self.losses_dict[constants.G_LOSS_KEY].append(errG.item())
            self.losses_dict[constants.D_OVERALL_LOSS_KEY].append(errD.item())
            self.losses_dict[constants.LIKENESS_LOSS_KEY].append(A_likeness_loss.item())
            self.losses_dict[constants.EDGE_LOSS_KEY].append(A_edge_loss.item())
            self.losses_dict[constants.G_ADV_LOSS_KEY].append(A_adv_loss.item())
            self.losses_dict[constants.D_A_FAKE_LOSS_KEY].append(D_A_fake_loss.item())
            self.losses_dict[constants.D_A_REAL_LOSS_KEY].append(D_A_real_loss.item())
            # self.losses_dict[constants.D_B_FAKE_LOSS_KEY].append(D_B_fake_loss.item())
            # self.losses_dict[constants.D_B_REAL_LOSS_KEY].append(D_B_real_loss.item())

    def visdom_report(self, iteration):
        self.visdom_reporter.plot_finegrain_loss("Transmission loss - " + str(constants.AIRLIGHT_GEN_VERSION) + str(constants.ITERATION), iteration, self.losses_dict, self.caption_dict)

    def visdom_infer_train(self, train_gray_tensor, train_depth_tensor, id):
        with torch.no_grad():
            train_depth_like = self.G_A(train_gray_tensor)

        self.visdom_reporter.plot_image((train_gray_tensor), str(id) + " Training - RGB " + str(constants.AIRLIGHT_GEN_VERSION) + str(constants.ITERATION))
        self.visdom_reporter.plot_image((train_depth_tensor), str(id) + " Training - Airlight " + str(constants.AIRLIGHT_GEN_VERSION) + str(constants.ITERATION))
        self.visdom_reporter.plot_image((train_depth_like), str(id) + " Training -  Airlight-Like " + str(constants.AIRLIGHT_GEN_VERSION) + str(constants.ITERATION))

    def visdom_infer_test(self, test_rgb_tensor, id):
        with torch.no_grad():
            test_depth_like = self.G_A(test_rgb_tensor)

        self.visdom_reporter.plot_image(test_rgb_tensor, str(id) + " Test - RGB " + str(constants.AIRLIGHT_GEN_VERSION) + str(constants.ITERATION))
        self.visdom_reporter.plot_image(test_depth_like, str(id) + " Test - Airlight-Like" + str(constants.AIRLIGHT_GEN_VERSION) + str(constants.ITERATION))

    def load_saved_state(self, checkpoint):
        self.G_A.load_state_dict(checkpoint[constants.GENERATOR_KEY + "A"])
        # self.G_B.load_state_dict(checkpoint[generator_key + "B"])
        self.D_A.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + "A"])
        # self.D_B.load_state_dict(checkpoint[discriminator_key + "B"])
        self.optimizerG.load_state_dict(checkpoint[constants.GENERATOR_KEY + constants.OPTIMIZER_KEY])
        self.optimizerD.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + constants.OPTIMIZER_KEY])

        self.schedulerG.load_state_dict(checkpoint[constants.GENERATOR_KEY + "scheduler"])
        self.schedulerD.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + "scheduler"])

    def save_states(self, epoch, iteration):
        save_dict = {'epoch': epoch, 'iteration': iteration}
        netGA_state_dict = self.G_A.state_dict()
        # netGB_state_dict = self.G_B.state_dict()
        netDA_state_dict = self.D_A.state_dict()
        # netDB_state_dict = self.D_B.state_dict()

        optimizerG_state_dict = self.optimizerG.state_dict()
        optimizerD_state_dict = self.optimizerD.state_dict()

        schedulerG_state_dict = self.schedulerG.state_dict()
        schedulerD_state_dict = self.schedulerD.state_dict()

        save_dict[constants.GENERATOR_KEY + "A"] = netGA_state_dict
        # save_dict[generator_key + "B"] = netGB_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + "A"] = netDA_state_dict
        # save_dict[discriminator_key + "B"] = netDB_state_dict

        save_dict[constants.GENERATOR_KEY + constants.OPTIMIZER_KEY] = optimizerG_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + constants.OPTIMIZER_KEY] = optimizerD_state_dict

        save_dict[constants.GENERATOR_KEY + "scheduler"] = schedulerG_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + "scheduler"] = schedulerD_state_dict

        torch.save(save_dict, constants.AIRLIGHT_GEN_CHECKPATH)
        print("Saved model state: %s Epoch: %d" % (len(save_dict), (epoch + 1)))

        # clear plots to avoid potential sudden jumps in visualization due to unstable gradients during early training
        if (epoch % 5 == 0):
            self.losses_dict[constants.G_LOSS_KEY].clear()
            self.losses_dict[constants.D_OVERALL_LOSS_KEY].clear()
            self.losses_dict[constants.LIKENESS_LOSS_KEY].clear()
            self.losses_dict[constants.EDGE_LOSS_KEY].clear()
            self.losses_dict[constants.G_ADV_LOSS_KEY].clear()
            self.losses_dict[constants.D_A_FAKE_LOSS_KEY].clear()
            self.losses_dict[constants.D_A_REAL_LOSS_KEY].clear()
