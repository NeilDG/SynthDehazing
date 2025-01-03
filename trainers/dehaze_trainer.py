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
from utils import logger, dehazing_proper
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

        self.dc_kernel = torch.ones(3, 3).to(self.gpu_device)

    def declare_models(self, t_blocks, is_t_unet, a_blocks):
        if (is_t_unet == 1):
            self.G_T = un.UnetGenerator(input_nc=3, output_nc=1, num_downs= t_blocks).to(self.gpu_device)
        else:
            self.G_T = cg.Generator(input_nc=3, output_nc=1, n_residual_blocks= t_blocks).to(self.gpu_device)

        self.D_T = dh.Discriminator(input_nc=1).to(self.gpu_device)
        self.A1 = dh.AirlightEstimator_Residual(num_channels=3, out_features=3, num_layers=a_blocks).to(self.gpu_device)

        self.optimizerG = torch.optim.Adam(itertools.chain(self.G_T.parameters(), self.A1.parameters()), lr=self.g_lr)
        self.optimizerD = torch.optim.Adam(itertools.chain(self.D_T.parameters()), lr=self.d_lr)
        self.schedulerG = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizerG, patience=100000 / self.batch_size, threshold=0.00005)
        self.schedulerD = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizerD, patience=100000 / self.batch_size, threshold=0.00005)
        self.fp16_scaler = amp.GradScaler()  # for automatic mixed precision

        # load albedo
        checkpt = torch.load(constants.ALBEDO_CHECKPT)
        # self.albedo_G = ffa_gan.FFA(gps=3, blocks=18).to(self.gpu_device)
        self.albedo_G = ffa_gan.FFA(gps=3, blocks=4).to(self.gpu_device)
        self.albedo_G.load_state_dict(checkpt[constants.GENERATOR_KEY + "A"])
        self.albedo_G.eval()
        print("Albedo network loaded: ", constants.ALBEDO_CHECKPT)

    def initialize_dict(self):
        # what to store in visdom?
        self.AIRLOSS_A_KEY = "AIRLOSS_A_KEY"
        self.SSIM_KEY = "SSIM_KEY"

        self.losses_dict = {}
        self.losses_dict[constants.G_LOSS_KEY] = []
        self.losses_dict[constants.D_OVERALL_LOSS_KEY] = []
        self.losses_dict[constants.LIKENESS_LOSS_KEY] = []
        self.losses_dict[constants.CYCLE_LOSS_KEY] = []
        self.losses_dict[constants.EDGE_LOSS_KEY] = []
        self.losses_dict[constants.G_ADV_LOSS_KEY] = []
        self.losses_dict[constants.D_A_FAKE_LOSS_KEY] = []
        self.losses_dict[constants.D_A_REAL_LOSS_KEY] = []
        self.losses_dict[self.AIRLOSS_A_KEY] = []
        self.losses_dict[self.SSIM_KEY] = []

        self.caption_dict = {}
        self.caption_dict[constants.G_LOSS_KEY] = "G loss per iteration"
        self.caption_dict[constants.D_OVERALL_LOSS_KEY] = "D loss per iteration"
        self.caption_dict[constants.LIKENESS_LOSS_KEY] = "Likeness loss per iteration"
        self.caption_dict[constants.CYCLE_LOSS_KEY] = "Clear like loss per iteration"
        self.caption_dict[constants.EDGE_LOSS_KEY] = "Edge loss per iteration"
        self.caption_dict[constants.G_ADV_LOSS_KEY] = "G adv loss per iteration"
        self.caption_dict[constants.D_A_FAKE_LOSS_KEY] = "D(T) fake loss per iteration"
        self.caption_dict[constants.D_A_REAL_LOSS_KEY] = "D(T) real loss per iteration"
        self.caption_dict[self.AIRLOSS_A_KEY] = "Airlight loss per iteration"
        self.caption_dict[self.SSIM_KEY] = "SSIM loss per iteration"

    def update_penalties(self, adv_weight, likeness_weight, edge_weight, clear_like_weight, comments):
        # what penalties to use for losses?
        self.adv_weight = adv_weight
        self.likeness_weight = likeness_weight
        self.edge_weight = edge_weight
        self.clear_like_weight = clear_like_weight

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
            print("Clear like loss weight: ", str(self.clear_like_weight), file=f)

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

    def measure_ssim(self, pred, target):
        ssim_metric = kornia.losses.SSIMLoss(window_size=5, max_val=0.5)
        return ssim_metric(pred, target)

    def edge_loss(self, pred, target):
        loss = nn.L1Loss()
        pred_grad = kornia.filters.spatial_gradient(pred)
        target_grad = kornia.filters.spatial_gradient(target)

        return loss(pred_grad, target_grad)

    def provide_clean_like(self, hazy_tensor, transmission_tensor, a_tensor):
        with torch.no_grad():
            #normalize to 0.0 to 1.0 first
            hazy_tensor = (hazy_tensor * 0.5) + 0.5
            transmission_tensor = (transmission_tensor * 0.5) + 0.5

            batch_clean_like = torch.zeros_like(hazy_tensor)

            for batch in range(0, np.shape(a_tensor)[0]):
                S = torch.full_like(transmission_tensor[batch], a_tensor[batch][0].item(), dtype=torch.float16, requires_grad=False)
                S = torch.mul(S, torch.sub(torch.full_like(transmission_tensor[batch], 1), transmission_tensor[batch]))  # A * (1 - T)

                for i in range(1, 3):
                    S_cat = torch.full_like(transmission_tensor[batch], a_tensor[batch][i].item(), dtype=torch.float16, requires_grad=False)
                    S_cat = torch.mul(S_cat, torch.sub(torch.full_like(transmission_tensor[batch], 1), transmission_tensor[batch]))  # A * (1 - T)
                    S = torch.cat([S, S_cat], 0)

                clean_like = ((hazy_tensor[batch] - S) / transmission_tensor[batch])
                clean_like = torch.clip(clean_like, 0.0, 1.0)

                batch_clean_like[batch] = clean_like

        #print("Final clean like shape: ", np.shape(batch_clean_like))
        return batch_clean_like

    def train(self, iteration, hazy_tensor, transmission_tensor, airlight_tensor, clear_tensor):
        with amp.autocast():
            albedo_tensor = self.albedo_G(hazy_tensor)
            transmission_like = self.G_T(albedo_tensor)

            self.D_T.train()
            self.optimizerD.zero_grad()

            prediction = self.D_T(transmission_tensor)
            real_tensor = torch.ones_like(prediction)
            fake_tensor = torch.zeros_like(prediction)

            D_T_real_loss = self.adversarial_loss(self.D_T(transmission_tensor), real_tensor) * self.adv_weight
            D_T_fake_loss = self.adversarial_loss(self.D_T(transmission_like.detach()), fake_tensor) * self.adv_weight

            errD = D_T_real_loss + D_T_fake_loss

            self.fp16_scaler.scale(errD).backward()
            if (iteration % (512 / self.batch_size) == 0):
                self.fp16_scaler.step(self.optimizerD)
                self.schedulerD.step(errD)

            #train T
            self.G_T.train()
            self.optimizerG.zero_grad()

            T_likeness_loss = self.likeness_loss(self.G_T(albedo_tensor), transmission_tensor) * self.likeness_weight
            T_edge_loss = self.edge_loss(self.G_T(albedo_tensor), transmission_tensor) * self.edge_weight

            prediction = self.D_T(self.G_T(albedo_tensor))
            real_tensor = torch.ones_like(prediction)
            T_adv_loss = self.adversarial_loss(prediction, real_tensor) * self.adv_weight

            #train A
            self.A1.train()
            atmosphere_like = self.A1(hazy_tensor)
            A1_loss = self.likeness_loss(self.A1(hazy_tensor), airlight_tensor) * self.likeness_weight

            clear_like = self.provide_clean_like(hazy_tensor, self.G_T(hazy_tensor), atmosphere_like)
            clear_like_loss = self.likeness_loss(clear_like, clear_tensor) #only use for profiling purposes

            errG = T_likeness_loss + T_edge_loss + T_adv_loss + T_likeness_loss + T_edge_loss + T_adv_loss + A1_loss

            self.fp16_scaler.scale(errG).backward()
            if (iteration % (512 / self.batch_size) == 0):
                self.fp16_scaler.step(self.optimizerG)
                self.schedulerG.step(errG)
                self.fp16_scaler.update()

        # what to put to losses dict for visdom reporting?
        self.losses_dict[constants.G_LOSS_KEY].append(errG.item())
        self.losses_dict[constants.D_OVERALL_LOSS_KEY].append(errD.item())
        self.losses_dict[constants.LIKENESS_LOSS_KEY].append(T_likeness_loss.item())
        self.losses_dict[constants.CYCLE_LOSS_KEY].append(clear_like_loss.item())
        self.losses_dict[constants.EDGE_LOSS_KEY].append(T_edge_loss.item())
        self.losses_dict[constants.G_ADV_LOSS_KEY].append(T_adv_loss.item())
        self.losses_dict[constants.D_A_FAKE_LOSS_KEY].append(D_T_fake_loss.item())
        self.losses_dict[constants.D_A_REAL_LOSS_KEY].append(D_T_real_loss.item())
        self.losses_dict[self.AIRLOSS_A_KEY].append(A1_loss.item())

    def test(self, hazy_tensor, clear_tensor):
        with torch.no_grad():
            albedo_tensor = self.albedo_G(hazy_tensor)
            transmission_like = self.G_T(albedo_tensor)
            atmosphere_like = self.A1(hazy_tensor)
            clear_like = self.provide_clean_like(hazy_tensor, transmission_like, atmosphere_like)
            ssim_metric = self.measure_ssim(clear_like, clear_tensor)
            self.losses_dict[self.SSIM_KEY].append(ssim_metric.item())

            return clear_like

    def visdom_report(self, iteration):
        with torch.no_grad():
            # report to visdom
            self.visdom_reporter.plot_finegrain_loss(str(constants.DEHAZER_VERSION) + str(constants.ITERATION), iteration, self.losses_dict, self.caption_dict)

    def visdom_infer_train(self, hazy_tensor, transmission_tensor, atmosphere_tensor, clear_tensor):
        with torch.no_grad():
            albedo_tensor = self.albedo_G(hazy_tensor)
            transmission_like = self.G_T(hazy_tensor)
            atmosphere_like = self.A1(hazy_tensor)
            clear_like = self.provide_clean_like(hazy_tensor, transmission_like, atmosphere_like)

            self.visdom_reporter.plot_image(hazy_tensor, "Train Hazy images - " + str(constants.DEHAZER_VERSION) + str(constants.ITERATION))
            self.visdom_reporter.plot_image(albedo_tensor, "Train Albedo-Like images - " + str(constants.DEHAZER_VERSION) + str(constants.ITERATION))
            self.visdom_reporter.plot_image(transmission_tensor, "Train Transmission images - " + str(constants.DEHAZER_VERSION) + str(constants.ITERATION))
            self.visdom_reporter.plot_image(transmission_like, "Train Transmission-Like images - " + str(constants.DEHAZER_VERSION) + str(constants.ITERATION))
            self.visdom_reporter.plot_image(clear_like, "Train Clean-Like Images " + str(constants.DEHAZER_VERSION) + str(constants.ITERATION))
            self.visdom_reporter.plot_image(clear_tensor, "Train Clean Images " + str(constants.DEHAZER_VERSION) + str(constants.ITERATION))

    def visdom_infer_test(self, hazy_tensor, number):
        with torch.no_grad():
            albedo_tensor = self.albedo_G(hazy_tensor)
            transmission_like = self.G_T(albedo_tensor)
            atmosphere_like = self.A1(hazy_tensor)
            clear_like = self.provide_clean_like(hazy_tensor, transmission_like, atmosphere_like)

            self.visdom_reporter.plot_image(hazy_tensor, "Unseen Hazy images - " + str(constants.DEHAZER_VERSION) + str(constants.ITERATION) + "-" + str(number))
            self.visdom_reporter.plot_image(albedo_tensor, "Test Albedo-Like images - " + str(constants.DEHAZER_VERSION) + str(constants.ITERATION) + "-" + str(number))
            self.visdom_reporter.plot_image(transmission_like, "Test Transmission-Like images - " + str(constants.DEHAZER_VERSION) + str(constants.ITERATION)+ "-" + str(number))
            self.visdom_reporter.plot_image(clear_like, "Unseen Clean-Like Images " + str(constants.DEHAZER_VERSION) + str(constants.ITERATION) + "-" + str(number))

    def visdom_infer_test_paired(self, hazy_tensor, clear_tensor, number):
        with torch.no_grad():
            albedo_tensor = self.albedo_G(hazy_tensor)
            transmission_like = self.G_T(albedo_tensor)
            atmosphere_like = self.A1(hazy_tensor)
            clear_like = self.provide_clean_like(hazy_tensor, transmission_like, atmosphere_like)

            self.visdom_reporter.plot_image(hazy_tensor, "Test Hazy images - " + str(constants.DEHAZER_VERSION) + str(constants.ITERATION) + "-" + str(number))
            self.visdom_reporter.plot_image(transmission_like, "Test Transmission-Like images - " + str(constants.DEHAZER_VERSION) + str(constants.ITERATION) + "-" + str(number))
            self.visdom_reporter.plot_image(clear_like, "Test Clean-Like Images " + str(constants.DEHAZER_VERSION) + str(constants.ITERATION) + "-" + str(number))
            self.visdom_reporter.plot_image(clear_tensor, "Test Clean Images " + str(constants.DEHAZER_VERSION) + str(constants.ITERATION) + "-" + str(number))
            self.visdom_reporter.plot_image(clear_tensor, "Test Clean Images " + str(constants.DEHAZER_VERSION) + str(constants.ITERATION) + "-" + str(number))

    def load_saved_state(self, checkpoint):
        self.A1.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + "A"])
        self.G_T.load_state_dict(checkpoint[constants.GENERATOR_KEY + "T"])
        self.D_T.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + "T"])

        self.optimizerG.load_state_dict(checkpoint[constants.GENERATOR_KEY + constants.OPTIMIZER_KEY])
        self.optimizerD.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + constants.OPTIMIZER_KEY])
        self.schedulerG.load_state_dict(checkpoint[constants.GENERATOR_KEY + "scheduler"])
        self.schedulerD.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + "scheduler"])

    def save_states(self, epoch, iteration):
        save_dict = {'epoch': epoch, 'iteration': iteration}
        netDA_state_dict = self.A1.state_dict()
        netGT_state_dict = self.G_T.state_dict()
        netDT_state_dict = self.D_T.state_dict()

        optimizerG_state_dict = self.optimizerG.state_dict()
        optimizerD_state_dict = self.optimizerD.state_dict()

        schedulerG_state_dict = self.schedulerG.state_dict()
        schedulerD_state_dict = self.schedulerD.state_dict()

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