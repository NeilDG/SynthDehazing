# -*- coding: utf-8 -*-
# Joint trainer for dehazing model

import os
from model import vanilla_cycle_gan as cg
from model import ffa_net as ffa_gan
from model import unet_gan as un
from model import dehaze_discriminator as dh
import constants
import torch
import kornia
import torch.cuda.amp as amp
import random
import itertools
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision.utils as vutils
from utils import logger
from utils import plot_utils
import torchvision.transforms as transforms
from loaders import image_dataset

class DehazeTrainer:

    def __init__(self, gpu_device, g_lr, d_lr, batch_size):
        self.gpu_device = gpu_device
        self.g_lr = g_lr
        self.d_lr = d_lr
        self.batch_size = batch_size

        self.visdom_reporter = plot_utils.VisdomReporter()
        self.initialize_dict()

    def declare_models(self, t_blocks, is_t_unet, a_blocks, is_a_unet):
        if (is_t_unet == 1):
            self.G_T = un.UnetGenerator(input_nc=3, output_nc=1, num_downs= t_blocks).to(self.gpu_device)
        else:
            self.G_T = cg.Generator(input_nc=3, output_nc=1, n_residual_blocks= t_blocks).to(self.gpu_device)

        self.D_T = dh.Discriminator(input_nc=1).to(self.gpu_device)

        if (is_a_unet == 1):
            self.G_A = un.UnetGenerator(input_nc=3, output_nc=3, num_downs= a_blocks).to(self.gpu_device)
        else:
            self.G_A = cg.Generator(input_nc=3, output_nc=3, n_residual_blocks= a_blocks).to(self.gpu_device)

        self.D_A = dh.Discriminator(input_nc=3).to(self.gpu_device)

        self.optimizerG = torch.optim.Adam(itertools.chain(self.G_T.parameters(), self.G_A.parameters()), lr=self.g_lr)
        self.optimizerD = torch.optim.Adam(itertools.chain(self.D_T.parameters(), self.D_A.parameters()), lr=self.d_lr)
        self.schedulerG = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizerG, patience=100000 / self.batch_size, threshold=0.00005)
        self.schedulerD = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizerD, patience=100000 / self.batch_size, threshold=0.00005)
        self.fp16_scaler = amp.GradScaler()  # for automatic mixed precision

        # load albedo
        checkpt = torch.load("checkpoint/albedo_transfer_v1.04_1.pt")
        self.albedo_G = ffa_gan.FFA(gps=3, blocks=18).to(self.gpu_device)
        self.albedo_G.load_state_dict(checkpt[constants.GENERATOR_KEY + "A"])
        self.albedo_G.eval()
        print("Albedo network loaded.")

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

        HYPERPARAMS_PATH = "checkpoint/" + constants.DEHAZER_VERSION + "_" + constants.ITERATION + ".config"
        with open(HYPERPARAMS_PATH, "w") as f:
            print("Version: ", constants.DEHAZER_CHECKPATH, file=f)
            print("Comment: ", comments, file=f)
            print("Learning rate for G: ", str(self.g_lr), file=f)
            print("Learning rate for D: ", str(self.d_lr), file=f)
            print("====================================", file=f)
            print("Adv weight: ", str(self.adv_weight), file=f)
            print("Likeness weight: ", str(self.likeness_weight), file=f)
            print("Edge loss weight: ", str(self.edge_weight), file=f)

    def adversarial_loss(self, pred, target):
        loss = nn.BCEWithLogitsLoss()
        return loss(pred, target)

    def identity_loss(self, pred, target):
        loss = nn.L1Loss()
        return loss(pred, target)

    def likeness_loss(self, pred, target):
        loss = nn.L1Loss()
        return loss(pred, target)

    def psnr_loss(self, pred, target):
        loss = kornia.losses.PSNRLoss(max_val=0.5)
        return loss(pred, target)

    def edge_loss(self, pred, target):
        loss = nn.L1Loss()
        pred_grad = kornia.filters.spatial_gradient(pred)
        target_grad = kornia.filters.spatial_gradient(target)

        return loss(pred_grad, target_grad)

    def provide_clean_like(self, hazy_tensor, transmission_tensor, atmosphere_tensor):
        #normalize to 0.0 to 1.0 first
        hazy_tensor = (hazy_tensor * 0.5) + 0.5
        atmosphere_tensor = (atmosphere_tensor * 0.5) + 0.5
        transmission_tensor = (transmission_tensor * 0.5) + 0.5

        # print("Range of hazy: ", torch.min(hazy_tensor), torch.max(hazy_tensor))
        # print("Range of T: ", torch.min(transmission_tensor), torch.max(transmission_tensor))
        # print("Range of A: ", torch.min(atmosphere_tensor), torch.max(atmosphere_tensor))

        clean_like = (hazy_tensor - atmosphere_tensor) / transmission_tensor
        clean_like = torch.clip(clean_like, 0.0, 1.0)

        return clean_like


    def train(self, hazy_tensor, transmission_tensor, atmosphere_tensor, clear_tensor):
        with amp.autocast():
            albedo_tensor = self.albedo_G(hazy_tensor)
            transmission_like = self.G_T(albedo_tensor)
            atmosphere_like = self.G_A(atmosphere_tensor)

            self.D_A.train()
            self.optimizerD.zero_grad()

            prediction = self.D_T(transmission_tensor)
            real_tensor = torch.ones_like(prediction)
            fake_tensor = torch.zeros_like(prediction)

            D_T_real_loss = self.adversarial_loss(self.D_T(transmission_tensor), real_tensor) * self.adv_weight
            D_T_fake_loss = self.adversarial_loss(self.D_T(transmission_like.detach()), fake_tensor) * self.adv_weight

            prediction = self.D_A(atmosphere_tensor)
            real_tensor = torch.ones_like(prediction)
            fake_tensor = torch.zeros_like(prediction)

            D_A_real_loss = self.adversarial_loss(self.D_A(atmosphere_tensor), real_tensor) * self.adv_weight
            D_A_fake_loss = self.adversarial_loss(self.D_A(atmosphere_like.detach()), fake_tensor) * self.adv_weight

            errD = D_T_real_loss + D_T_fake_loss + D_A_real_loss + D_A_fake_loss

            self.fp16_scaler.scale(errD).backward()
            self.fp16_scaler.step(self.optimizerD)
            self.schedulerD.step(errD)

            #train T
            self.G_T.train()
            self.G_A.train()
            self.optimizerG.zero_grad()

            T_likeness_loss = self.likeness_loss(self.G_T(albedo_tensor), transmission_tensor) * self.likeness_weight
            T_edge_loss = self.edge_loss(self.G_T(albedo_tensor), transmission_tensor) * self.edge_weight

            prediction = self.D_T(self.G_T(albedo_tensor))
            real_tensor = torch.ones_like(prediction)
            T_adv_loss = self.adversarial_loss(prediction, real_tensor) * self.adv_weight

            A_likeness_loss = self.likeness_loss(self.G_A(hazy_tensor), atmosphere_tensor) * self.likeness_weight
            A_edge_loss = self.edge_loss(self.G_A(hazy_tensor), atmosphere_tensor) * self.edge_weight

            prediction = self.D_A(self.G_A(hazy_tensor))
            real_tensor = torch.ones_like(prediction)
            A_adv_loss = self.adversarial_loss(prediction, real_tensor) * self.adv_weight

            clear_like = self.provide_clean_like(hazy_tensor, self.G_T(hazy_tensor), self.G_A(hazy_tensor))
            clear_like_loss = self.likeness_loss(clear_like, clear_tensor) * self.likeness_weight

            errG = T_likeness_loss + T_edge_loss + T_adv_loss + T_likeness_loss + T_edge_loss + T_adv_loss + \
                   A_likeness_loss + A_edge_loss + A_adv_loss + A_likeness_loss + A_edge_loss + A_adv_loss + clear_like_loss

            self.fp16_scaler.scale(errG).backward()
            self.fp16_scaler.step(self.optimizerG)
            self.schedulerG.step(errG)

            self.fp16_scaler.update()


        # what to put to losses dict for visdom reporting?
        self.losses_dict[constants.G_LOSS_KEY].append(errG.item())
        self.losses_dict[constants.D_OVERALL_LOSS_KEY].append(errD.item())
        self.losses_dict[constants.LIKENESS_LOSS_KEY].append(T_likeness_loss.item() + A_likeness_loss.item() + clear_like_loss.item())
        self.losses_dict[constants.EDGE_LOSS_KEY].append(T_edge_loss.item() + A_edge_loss.item())
        self.losses_dict[constants.G_ADV_LOSS_KEY].append(T_adv_loss.item() + A_adv_loss.item())
        self.losses_dict[constants.D_A_FAKE_LOSS_KEY].append(D_T_fake_loss.item() + D_A_fake_loss.item())
        self.losses_dict[constants.D_A_REAL_LOSS_KEY].append(D_T_real_loss.item() + D_A_real_loss.item())

    def visdom_report(self, iteration):
        with torch.no_grad():
            # report to visdom
            self.visdom_reporter.plot_finegrain_loss(str(constants.DEHAZER_VERSION) + str(constants.ITERATION), iteration, self.losses_dict, self.caption_dict)

    def visdom_infer_train(self, hazy_tensor, transmission_tensor, atmosphere_tensor, clean_tensor):
        with torch.no_grad():
            albedo_tensor = self.albedo_G(hazy_tensor)
            clear_like = self.provide_clean_like(hazy_tensor, self.G_T(hazy_tensor), self.G_A(hazy_tensor))
            #transmission_like = self.G_T(hazy_tensor)
            #atmosphere_like = self.G_A(atmosphere_tensor)

            self.visdom_reporter.plot_image(hazy_tensor, "Train Hazy images - " + str(constants.DEHAZER_VERSION) + str(constants.ITERATION))
            self.visdom_reporter.plot_image(albedo_tensor, "Train Albedo-Like images - " + str(constants.DEHAZER_VERSION) + str(constants.ITERATION))
            self.visdom_reporter.plot_image(transmission_tensor, "Train Transmission images - " + str(constants.DEHAZER_VERSION) + str(constants.ITERATION))
            #self.visdom_reporter.plot_image(transmission_like, "Train Transmission-Like images - " + str(constants.DEHAZER_VERSION) + str(constants.ITERATION))
            self.visdom_reporter.plot_image(atmosphere_tensor, "Train Atmosphere images - " + str(constants.DEHAZER_VERSION) + str(constants.ITERATION))
            #self.visdom_reporter.plot_image(atmosphere_like, "Train Atmosphere-Like images - " + str(constants.DEHAZER_VERSION) + str(constants.ITERATION))
            self.visdom_reporter.plot_image(clear_like, "Train Clean-Like Images " + str(constants.DEHAZER_VERSION) + str(constants.ITERATION))
            self.visdom_reporter.plot_image(clean_tensor, "Train Clean Images " + str(constants.DEHAZER_VERSION) + str(constants.ITERATION))

    def visdom_infer_test(self, hazy_tensor, number):
        with torch.no_grad():
            albedo_tensor = self.albedo_G(hazy_tensor)
            clear_like = self.provide_clean_like(hazy_tensor, self.G_T(albedo_tensor), self.G_A(hazy_tensor))
            transmission_like = self.G_T(albedo_tensor)
            atmosphere_like = self.G_A(hazy_tensor)

            self.visdom_reporter.plot_image(hazy_tensor, "Train Hazy images - " + str(constants.DEHAZER_VERSION) + str(constants.ITERATION) + "-" + str(number))
            self.visdom_reporter.plot_image(albedo_tensor, "Train Albedo-Like images - " + str(constants.DEHAZER_VERSION) + str(constants.ITERATION) + "-" + str(number))
            self.visdom_reporter.plot_image(transmission_like, "Train Transmission-Like images - " + str(constants.DEHAZER_VERSION) + str(constants.ITERATION)+ "-" + str(number))
            self.visdom_reporter.plot_image(atmosphere_like, "Train Atmosphere-Like images - " + str(constants.DEHAZER_VERSION) + str(constants.ITERATION)+ "-" + str(number))
            self.visdom_reporter.plot_image(clear_like, "Train Clean-Like Images " + str(constants.DEHAZER_VERSION) + str(constants.ITERATION) + "-" + str(number))

    def load_saved_state(self, checkpoint):
        self.G_A.load_state_dict(checkpoint[constants.GENERATOR_KEY + "A"])
        self.D_A.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + "A"])
        self.G_T.load_state_dict(checkpoint[constants.GENERATOR_KEY + "T"])
        self.D_T.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + "T"])

        self.optimizerG.load_state_dict(checkpoint[constants.GENERATOR_KEY + constants.OPTIMIZER_KEY])
        self.optimizerD.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + constants.OPTIMIZER_KEY])
        self.schedulerG.load_state_dict(checkpoint[constants.GENERATOR_KEY + "scheduler"])
        self.schedulerD.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + "scheduler"])

    def save_states(self, epoch, iteration):
        save_dict = {'epoch': epoch, 'iteration': iteration}
        netGA_state_dict = self.G_A.state_dict()
        netDA_state_dict = self.D_A.state_dict()
        netGT_state_dict = self.G_T.state_dict()
        netDT_state_dict = self.D_T.state_dict()

        optimizerG_state_dict = self.optimizerG.state_dict()
        optimizerD_state_dict = self.optimizerD.state_dict()

        schedulerG_state_dict = self.schedulerG.state_dict()
        schedulerD_state_dict = self.schedulerD.state_dict()

        save_dict[constants.GENERATOR_KEY + "A"] = netGA_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + "A"] = netDA_state_dict
        save_dict[constants.GENERATOR_KEY + "T"] = netGT_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + "T"] = netDT_state_dict

        save_dict[constants.GENERATOR_KEY + constants.OPTIMIZER_KEY] = optimizerG_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + constants.OPTIMIZER_KEY] = optimizerD_state_dict

        save_dict[constants.GENERATOR_KEY + "scheduler"] = schedulerG_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + "scheduler"] = schedulerD_state_dict

        torch.save(save_dict, constants.DEHAZER_CHECKPATH)
        print("Saved model state: %s Epoch: %d" % (len(save_dict), (epoch + 1)))

        # clear plots to avoid potential sudden jumps in visualization due to unstable gradients during early training
        # if (epoch % 20 == 0):
        #     self.losses_dict[constants.G_LOSS_KEY].clear()
        #     self.losses_dict[constants.D_OVERALL_LOSS_KEY].clear()
        #     self.losses_dict[constants.LIKENESS_LOSS_KEY].clear()
        #     self.losses_dict[constants.PSNR_LOSS_KEY].clear()
        #     self.losses_dict[constants.G_ADV_LOSS_KEY].clear()
        #     self.losses_dict[constants.D_A_FAKE_LOSS_KEY].clear()
        #     self.losses_dict[constants.D_A_REAL_LOSS_KEY].clear()