# -*- coding: utf-8 -*-
# Template trainer. Do not use this for actual training.

import itertools
import os

import kornia
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.cuda.amp as amp
import torch.nn as nn
import torchvision.utils as vutils
import constants
from model import vanilla_cycle_gan as cycle_gan
from model import ffa_net as ffa_gan
from utils import plot_utils


class AlbedoTrainer:

    def __init__(self, gpu_device, g_lr, d_lr, num_blocks):
        self.gpu_device = gpu_device
        self.g_lr = g_lr
        self.d_lr = d_lr
        #self.G_A = cycle_gan.Generator(downsampling_blocks = 2, n_residual_blocks=16).to(self.gpu_device)
        self.G_A = ffa_gan.FFA(gps = 3, blocks = num_blocks).to(self.gpu_device)
        self.D_A = cycle_gan.Discriminator().to(self.gpu_device)  # use CycleGAN's discriminator

        self.visdom_reporter = plot_utils.VisdomReporter()
        self.optimizerG = torch.optim.Adam(itertools.chain(self.G_A.parameters()), lr=self.g_lr)
        self.optimizerD = torch.optim.Adam(itertools.chain(self.D_A.parameters()), lr=self.d_lr)
        self.schedulerG = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizerG, patience=100000 / constants.batch_size, threshold=0.00005)
        self.schedulerD = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizerD, patience=100000 / constants.batch_size, threshold=0.00005)
        self.initialize_dict()

        self.fp16_scaler = amp.GradScaler()  # for automatic mixed precision

    def initialize_dict(self):
        # what to store in visdom?
        self.losses_dict = {}
        self.losses_dict[constants.G_LOSS_KEY] = []
        self.losses_dict[constants.D_OVERALL_LOSS_KEY] = []
        self.losses_dict[constants.LIKENESS_LOSS_KEY] = []
        self.losses_dict[constants.PSNR_LOSS_KEY] = []
        self.losses_dict[constants.G_ADV_LOSS_KEY] = []
        self.losses_dict[constants.D_A_FAKE_LOSS_KEY] = []
        self.losses_dict[constants.D_A_REAL_LOSS_KEY] = []

        self.caption_dict = {}
        self.caption_dict[constants.G_LOSS_KEY] = "G loss per iteration"
        self.caption_dict[constants.D_OVERALL_LOSS_KEY] = "D loss per iteration"
        self.caption_dict[constants.LIKENESS_LOSS_KEY] = "Likeness loss per iteration"
        self.caption_dict[constants.PSNR_LOSS_KEY] = "PSNR/SSIM color loss per iteration"
        self.caption_dict[constants.G_ADV_LOSS_KEY] = "G adv loss per iteration"
        self.caption_dict[constants.D_A_FAKE_LOSS_KEY] = "D(A) fake loss per iteration"
        self.caption_dict[constants.D_A_REAL_LOSS_KEY] = "D(A) real loss per iteration"

    def update_penalties(self, adv_weight, likeness_weight, psnr_loss_weight, use_psnr, comments):
        # what penalties to use for losses?
        self.adv_weight = adv_weight
        self.likeness_weight = likeness_weight
        self.psnr_loss_weight = psnr_loss_weight
        self.use_psnr = use_psnr

        # save hyperparameters for bookeeping
        HYPERPARAMS_PATH = "checkpoint/" + constants.COLOR_TRANSFER_VERSION + "_" + constants.ITERATION + ".config"
        with open(HYPERPARAMS_PATH, "w") as f:
            print("Version: ", constants.COLOR_TRANSFER_CHECKPATH, file=f)
            print("Comment: ", comments, file=f)
            print("Learning rate for G: ", str(self.g_lr), file=f)
            print("Learning rate for D: ", str(self.d_lr), file=f)
            print("====================================", file=f)
            print("Adv weight: ", str(self.adv_weight), file=f)
            print("Likeness weight: ", str(self.likeness_weight), file=f)
            print("Use PSNR: ", str(self.use_psnr), file = f)
            print("PSNR/SSIM loss weight: ", str(self.psnr_loss_weight), file=f)

    def adversarial_loss(self, pred, target):
        # loss = nn.L1Loss()
        # return loss(pred, target)

        loss = nn.BCEWithLogitsLoss()
        return loss(pred, target)

    def identity_loss(self, pred, target):
        loss = nn.L1Loss()
        return loss(pred, target)

    def cycle_loss(self, pred, target):
        loss = nn.L1Loss()
        return loss(pred, target)

    def likeness_loss(self, pred, target):
        loss = nn.L1Loss()
        return loss(pred, target)

    def smoothness_loss(self, pred, target):
        loss = nn.L1Loss()
        pred_blur = kornia.gaussian_blur2d(pred, (7, 7), (5.5, 5.5))
        target_blur = kornia.gaussian_blur2d(target, (7, 7), (5.5, 5.5))

        return loss(pred_blur, target_blur)

    def psnr_loss(self, pred, target):
        if(self.use_psnr):
            loss = kornia.losses.PSNRLoss(max_val=0.5)
        else:
            loss = kornia.losses.SSIMLoss(window_size=3, max_val=0.5)

        return loss(pred, target)

    def train(self, styled_tensor, albedo_tensor):
        with amp.autocast():
            albedo_like = self.G_A(styled_tensor)

            self.D_A.train()
            self.optimizerD.zero_grad()

            prediction = self.D_A(albedo_tensor)
            real_tensor = torch.ones_like(prediction)
            fake_tensor = torch.zeros_like(prediction)

            D_A_real_loss = self.adversarial_loss(self.D_A(albedo_tensor), real_tensor) * self.adv_weight
            D_A_fake_loss = self.adversarial_loss(self.D_A(albedo_like.detach()), fake_tensor) * self.adv_weight

            errD = D_A_real_loss + D_A_fake_loss
            self.fp16_scaler.scale(errD).backward()
            self.fp16_scaler.step(self.optimizerD)
            self.schedulerD.step(errD)

            self.G_A.train()
            self.optimizerG.zero_grad()

            albedo_like = self.G_A(styled_tensor)

            A_likeness_loss = self.likeness_loss(albedo_like, albedo_tensor) * self.likeness_weight
            A_color_loss = self.psnr_loss(albedo_like, albedo_tensor) * self.psnr_loss_weight

            prediction = self.D_A(albedo_like)
            real_tensor = torch.ones_like(prediction)
            A_adv_loss = self.adversarial_loss(prediction, real_tensor) * self.adv_weight

            errG = A_likeness_loss + A_color_loss + A_adv_loss

            self.fp16_scaler.scale(errG).backward()
            self.fp16_scaler.step(self.optimizerG)
            self.schedulerG.step(errG)

            self.fp16_scaler.update()

        # what to put to losses dict for visdom reporting?
        self.losses_dict[constants.G_LOSS_KEY].append(errG.item())
        self.losses_dict[constants.D_OVERALL_LOSS_KEY].append(errD.item())
        self.losses_dict[constants.LIKENESS_LOSS_KEY].append(A_likeness_loss.item())
        self.losses_dict[constants.PSNR_LOSS_KEY].append(A_color_loss.item())
        self.losses_dict[constants.G_ADV_LOSS_KEY].append(A_adv_loss.item())
        self.losses_dict[constants.D_A_FAKE_LOSS_KEY].append(D_A_fake_loss.item())
        self.losses_dict[constants.D_A_REAL_LOSS_KEY].append(D_A_real_loss.item())

    def visdom_report(self, iteration, styled_tensor, albedo_tensor, test_styled_tensor, test_albedo_tensor):
        with torch.no_grad():
            albedo_like = self.G_A(styled_tensor)
            test_albedo_like = self.G_A(test_styled_tensor)


        # report to visdom
        self.visdom_reporter.plot_finegrain_loss(str(constants.COLOR_TRANSFER_VERSION) +str(constants.ITERATION), iteration, self.losses_dict, self.caption_dict)
        self.visdom_reporter.plot_image(styled_tensor, "Training Dirty images - " + str(constants.COLOR_TRANSFER_VERSION) + str(constants.ITERATION))
        self.visdom_reporter.plot_image(albedo_tensor, "Training Clean images - " + str(constants.COLOR_TRANSFER_VERSION) + str(constants.ITERATION))
        self.visdom_reporter.plot_image(albedo_like, "Training Clean-like images - "+str(constants.COLOR_TRANSFER_VERSION) +str(constants.ITERATION))
        self.visdom_reporter.plot_image(test_styled_tensor, "Test Dirty images - " + str(constants.COLOR_TRANSFER_VERSION) + str(constants.ITERATION))
        self.visdom_reporter.plot_image(test_albedo_tensor, "Test Clean images - " + str(constants.COLOR_TRANSFER_VERSION) + str(constants.ITERATION))
        self.visdom_reporter.plot_image(test_albedo_like, "Test Clean-like images - "+str(constants.COLOR_TRANSFER_VERSION) +str(constants.ITERATION))

    def visdom_infer(self, test_styled_tensor, caption_dirty, caption_clean):
        with torch.no_grad():
            test_clean_like = self.G_A(test_styled_tensor)

        self.visdom_reporter.plot_image(test_styled_tensor, caption_dirty + " - " + str(constants.COLOR_TRANSFER_VERSION) + str(constants.ITERATION))
        self.visdom_reporter.plot_image(test_clean_like, caption_clean + " - "+str(constants.COLOR_TRANSFER_VERSION) +str(constants.ITERATION))

    def produce_image(self, dirty_tensor):
        with torch.no_grad():
            clean_like = self.G_A(dirty_tensor, self.denoise_model(dirty_tensor))
            resized_fake = nn.functional.interpolate(clean_like, scale_factor=1.0, mode="bilinear", recompute_scale_factor=True)

        return resized_fake

    def infer(self, dirty_tensor, file_number):
        LOCATION = os.getcwd() + "/figures/"
        with torch.no_grad():
            clean_like = self.G_A(dirty_tensor).detach()

        # resize tensors for better viewing
        resized_normal = nn.functional.interpolate(dirty_tensor, scale_factor=4.0, mode="bilinear", recompute_scale_factor=True)
        resized_fake = nn.functional.interpolate(clean_like, scale_factor=4.0, mode="bilinear", recompute_scale_factor=True)

        print("New shapes: %s %s" % (np.shape(resized_normal), np.shape(resized_fake)))

        fig, ax = plt.subplots(2, 1)
        fig.set_size_inches((32, 16))
        fig.tight_layout()

        ims = np.transpose(vutils.make_grid(resized_normal, nrow=8, padding=2, normalize=True).cpu(), (1, 2, 0))
        ax[0].set_axis_off()
        ax[0].imshow(ims)

        ims = np.transpose(vutils.make_grid(resized_fake, nrow=8, padding=2, normalize=True).cpu(), (1, 2, 0))
        ax[1].set_axis_off()
        ax[1].imshow(ims)

        plt.subplots_adjust(left=0.06, wspace=0.0, hspace=0.15)
        plt.savefig(LOCATION + "result_" + str(file_number) + ".png")
        plt.show()

    def load_saved_state(self, checkpoint):
        self.G_A.load_state_dict(checkpoint[constants.GENERATOR_KEY + "A"])
        self.D_A.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + "A"])
        self.optimizerG.load_state_dict(checkpoint[constants.GENERATOR_KEY + constants.OPTIMIZER_KEY])
        self.optimizerD.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + constants.OPTIMIZER_KEY])
        self.schedulerG.load_state_dict(checkpoint[constants.GENERATOR_KEY + "scheduler"])
        self.schedulerD.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + "scheduler"])

    def save_states(self, epoch, iteration):
        save_dict = {'epoch': epoch, 'iteration': iteration}
        netGA_state_dict = self.G_A.state_dict()
        netDA_state_dict = self.D_A.state_dict()

        optimizerG_state_dict = self.optimizerG.state_dict()
        optimizerD_state_dict = self.optimizerD.state_dict()

        schedulerG_state_dict = self.schedulerG.state_dict()
        schedulerD_state_dict = self.schedulerD.state_dict()

        save_dict[constants.GENERATOR_KEY + "A"] = netGA_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + "A"] = netDA_state_dict

        save_dict[constants.GENERATOR_KEY + constants.OPTIMIZER_KEY] = optimizerG_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + constants.OPTIMIZER_KEY] = optimizerD_state_dict

        save_dict[constants.GENERATOR_KEY + "scheduler"] = schedulerG_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + "scheduler"] = schedulerD_state_dict

        torch.save(save_dict, constants.COLOR_TRANSFER_CHECKPATH)
        print("Saved model state: %s Epoch: %d" % (len(save_dict), (epoch + 1)))

        # clear plots to avoid potential sudden jumps in visualization due to unstable gradients during early training
        if (epoch % 20 == 0):
            self.losses_dict[constants.G_LOSS_KEY].clear()
            self.losses_dict[constants.D_OVERALL_LOSS_KEY].clear()
            self.losses_dict[constants.LIKENESS_LOSS_KEY].clear()
            self.losses_dict[constants.PSNR_LOSS_KEY].clear()
            self.losses_dict[constants.G_ADV_LOSS_KEY].clear()
            self.losses_dict[constants.D_A_FAKE_LOSS_KEY].clear()
            self.losses_dict[constants.D_A_REAL_LOSS_KEY].clear()
