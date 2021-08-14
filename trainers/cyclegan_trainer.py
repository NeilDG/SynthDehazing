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
from model import vanilla_cycle_gan as discrim_gan
from utils import plot_utils
from utils import pytorch_colors


class CycleGANTrainer:

    def __init__(self, gpu_device, g_lr, d_lr):
        self.gpu_device = gpu_device
        self.g_lr = g_lr
        self.d_lr = d_lr
        self.G_A = discrim_gan.Generator(n_residual_blocks=8).to(self.gpu_device)
        # self.G_A = transfer_gan.Generator(n_residual_blocks=8).to(self.gpu_device)
        self.G_B = discrim_gan.Generator(n_residual_blocks=8).to(self.gpu_device)
        # self.G_B = transfer_gan.Generator(n_residual_blocks=8).to(self.gpu_device)

        self.D_A = discrim_gan.Discriminator().to(self.gpu_device)  # use CycleGAN's discriminator
        self.D_B = discrim_gan.Discriminator().to(self.gpu_device)

        self.visdom_reporter = plot_utils.VisdomReporter()
        self.optimizerG = torch.optim.Adam(itertools.chain(self.G_A.parameters(), self.G_B.parameters()), lr=self.g_lr)
        self.optimizerD = torch.optim.Adam(itertools.chain(self.D_A.parameters(), self.D_B.parameters()), lr=self.d_lr)
        self.schedulerG = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizerG, patience=1000, threshold=0.00005)
        self.schedulerD = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizerD, patience=1000, threshold=0.00005)
        self.initialize_dict()

        self.fp16_scaler = amp.GradScaler()  # for automatic mixed precision

    def initialize_dict(self):
        # what to store in visdom?
        self.losses_dict = {}
        self.losses_dict[constants.G_LOSS_KEY] = []
        self.losses_dict[constants.D_OVERALL_LOSS_KEY] = []
        self.losses_dict[constants.IDENTITY_LOSS_KEY] = []
        self.losses_dict[constants.LIKENESS_LOSS_KEY] = []
        self.losses_dict[constants.SMOOTHNESS_LOSS_KEY] = []
        self.losses_dict[constants.G_ADV_LOSS_KEY] = []
        self.losses_dict[constants.D_A_FAKE_LOSS_KEY] = []
        self.losses_dict[constants.D_A_REAL_LOSS_KEY] = []
        self.losses_dict[constants.D_B_FAKE_LOSS_KEY] = []
        self.losses_dict[constants.D_B_REAL_LOSS_KEY] = []
        self.losses_dict[constants.CYCLE_LOSS_KEY] = []

        self.caption_dict = {}
        self.caption_dict[constants.G_LOSS_KEY] = "G loss per iteration"
        self.caption_dict[constants.D_OVERALL_LOSS_KEY] = "D loss per iteration"
        self.caption_dict[constants.IDENTITY_LOSS_KEY] = "Identity loss per iteration"
        self.caption_dict[constants.LIKENESS_LOSS_KEY] = "Likeness loss per iteration"
        self.caption_dict[constants.SMOOTHNESS_LOSS_KEY] = "Smoothness loss per iteration"
        self.caption_dict[constants.G_ADV_LOSS_KEY] = "G adv loss per iteration"
        self.caption_dict[constants.D_A_FAKE_LOSS_KEY] = "D(A) fake loss per iteration"
        self.caption_dict[constants.D_A_REAL_LOSS_KEY] = "D(A) real loss per iteration"
        self.caption_dict[constants.D_B_FAKE_LOSS_KEY] = "D(B) fake loss per iteration"
        self.caption_dict[constants.D_B_REAL_LOSS_KEY] = "D(B) real loss per iteration"
        self.caption_dict[constants.CYCLE_LOSS_KEY] = "Cycle loss per iteration"

    def update_penalties(self, adv_weight, id_weight, likeness_weight, cycle_weight, smoothness_weight, comments):
        # what penalties to use for losses?
        self.adv_weight = adv_weight
        self.id_weight = id_weight
        self.likeness_weight = likeness_weight
        self.cycle_weight = cycle_weight
        self.smoothness_weight = smoothness_weight

        # save hyperparameters for bookeeping
        HYPERPARAMS_PATH = "checkpoint/" + constants.COLOR_TRANSFER_VERSION + "_" + constants.ITERATION + ".config"
        with open(HYPERPARAMS_PATH, "w") as f:
            print("Version: ", constants.COLOR_TRANSFER_CHECKPATH, file=f)
            print("Comment: ", comments, file=f)
            print("Learning rate for G: ", str(self.g_lr), file=f)
            print("Learning rate for D: ", str(self.d_lr), file=f)
            print("====================================", file=f)
            print("Adv weight: ", str(self.adv_weight), file=f)
            print("Identity weight: ", str(self.id_weight), file=f)
            print("Likeness weight: ", str(self.likeness_weight), file=f)
            print("Smoothness weight: ", str(self.smoothness_weight), file=f)
            print("Cycle weight: ", str(self.cycle_weight), file=f)
            print("====================================", file=f)
            print("Brightness enhance: ", str(constants.brightness_enhance), file=f)
            print("Contrast enhance: ", str(constants.contrast_enhance), file=f)

    def adversarial_loss(self, pred, target):
        loss = nn.BCEWithLogitsLoss()
        return loss(pred, target)

    def identity_loss(self, pred, target):
        loss = nn.L1Loss()
        return loss(pred, target)

    def cycle_loss(self, pred, target):
        loss = nn.L1Loss()
        return loss(pred, target)
        # loss = ssim_loss.SSIM()
        # return 1 - loss(pred, target)

    def likeness_loss(self, pred, target):
        loss = nn.L1Loss()
        return loss(pred, target)

        # loss = vgg.VGGPerceptualLoss().to(self.gpu_device)
        # return loss(pred, target)

    def smoothness_loss(self, pred, target):
        loss = nn.L1Loss()
        pred_blur = kornia.gaussian_blur2d(pred, (7, 7), (5.5, 5.5))
        target_blur = kornia.gaussian_blur2d(target, (7, 7), (5.5, 5.5))

        return loss(pred_blur, target_blur)

    def color_shift_loss(self, pred, target):
        pred_lab = pytorch_colors.rgb_to_lab(pred.detach())
        target_lab = pytorch_colors.rgb_to_lab(target.detach())

        (y, u, v) = torch.chunk(pred_lab.transpose(0, 1), 3)
        pred_ab = torch.cat((u, v))

        (y, u, v) = torch.chunk(target_lab.transpose(0, 1), 3)
        target_ab = torch.cat((u, v))

        pred_ab = torch.cat((torch.zeros_like(y), pred_ab))
        target_ab = torch.cat((torch.zeros_like(y), target_ab))
        pred_ab = pred_ab.transpose(0, 1)
        target_ab = target_ab.transpose(0, 1)

        # impose color penalty to tensor for autograd by canceling out original pred/target pair
        pred = pred + pred_ab
        target = target + target_ab

        loss = nn.L1Loss()
        return loss(pred, target)

    def train(self, dirty_tensor, clean_tensor):
        with amp.autocast():
            clean_like = self.G_A(dirty_tensor)
            dirty_like = self.G_B(clean_tensor)

            self.D_A.train()
            self.D_B.train()
            self.optimizerD.zero_grad()

            prediction = self.D_A(clean_tensor)
            real_tensor = torch.ones_like(prediction)
            fake_tensor = torch.zeros_like(prediction)

            D_A_real_loss = self.adversarial_loss(self.D_A(clean_tensor), real_tensor) * self.adv_weight
            D_A_fake_loss = self.adversarial_loss(self.D_A(clean_like.detach()), fake_tensor) * self.adv_weight

            prediction = self.D_B(dirty_tensor)
            real_tensor = torch.ones_like(prediction)
            fake_tensor = torch.zeros_like(prediction)

            D_B_real_loss = self.adversarial_loss(self.D_B(dirty_tensor), real_tensor) * self.adv_weight
            D_B_fake_loss = self.adversarial_loss(self.D_B(dirty_like.detach()), fake_tensor) * self.adv_weight

            errD = D_A_real_loss + D_A_fake_loss + D_B_real_loss + D_B_fake_loss
            # errD.backward()
            # self.optimizerD.step()
            self.fp16_scaler.scale(errD).backward()
            self.fp16_scaler.step(self.optimizerD)
            self.schedulerD.step(errD)

            self.G_A.train()
            self.G_B.train()
            self.optimizerG.zero_grad()

            identity_like = self.G_A(clean_tensor)
            clean_like = self.G_A(dirty_tensor)
            dirty_like = self.G_B(clean_like)

            identity_loss = self.identity_loss(identity_like, clean_tensor) * self.id_weight
            A_likeness_loss = self.likeness_loss(clean_like, clean_tensor) * self.likeness_weight
            A_smoothness_loss = self.smoothness_loss(clean_like, clean_tensor) * self.smoothness_weight
            A_cycle_loss = self.cycle_loss(dirty_like, dirty_tensor) * self.cycle_weight

            dirty_like = self.G_B(clean_tensor)
            B_likeness_loss = self.likeness_loss(dirty_like, dirty_tensor) * self.likeness_weight
            B_smoothness_loss = self.smoothness_loss(dirty_like, dirty_tensor) * self.smoothness_weight
            B_cycle_loss = self.cycle_loss(self.G_A(dirty_like), clean_tensor) * self.cycle_weight

            prediction = self.D_A(clean_like)
            real_tensor = torch.ones_like(prediction)
            A_adv_loss = self.adversarial_loss(prediction, real_tensor) * self.adv_weight

            prediction = self.D_B(dirty_like)
            real_tensor = torch.ones_like(prediction)
            B_adv_loss = self.adversarial_loss(prediction, real_tensor) * self.adv_weight

            errG = identity_loss + A_likeness_loss + B_likeness_loss + A_smoothness_loss + B_smoothness_loss + A_adv_loss + B_adv_loss + A_cycle_loss + B_cycle_loss
            # errG.backward()
            # self.optimizerG.step()
            self.fp16_scaler.scale(errG).backward()
            self.fp16_scaler.step(self.optimizerG)
            self.schedulerG.step(errG)

            self.fp16_scaler.update()

        # what to put to losses dict for visdom reporting?
        self.losses_dict[constants.G_LOSS_KEY].append(errG.item())
        self.losses_dict[constants.D_OVERALL_LOSS_KEY].append(errD.item())
        self.losses_dict[constants.IDENTITY_LOSS_KEY].append(identity_loss.item())
        self.losses_dict[constants.LIKENESS_LOSS_KEY].append(B_likeness_loss.item())
        self.losses_dict[constants.SMOOTHNESS_LOSS_KEY].append(A_smoothness_loss.item() + B_smoothness_loss.item())
        self.losses_dict[constants.G_ADV_LOSS_KEY].append(A_adv_loss.item() + B_adv_loss.item())
        self.losses_dict[constants.D_A_FAKE_LOSS_KEY].append(D_A_fake_loss.item())
        self.losses_dict[constants.D_A_REAL_LOSS_KEY].append(D_A_real_loss.item())
        self.losses_dict[constants.D_B_FAKE_LOSS_KEY].append(D_B_fake_loss.item())
        self.losses_dict[constants.D_B_REAL_LOSS_KEY].append(D_B_real_loss.item())
        self.losses_dict[constants.CYCLE_LOSS_KEY].append(A_cycle_loss.item() + B_cycle_loss.item())

    def visdom_report(self, iteration, dirty_tensor, clean_tensor, test_dirty_tensor, test_clean_tensor):
        with torch.no_grad():
            clean_like = self.G_A(dirty_tensor)
            test_clean_like = self.G_A(test_dirty_tensor)
            test_dirty_like = self.G_B(test_clean_like)

            # inferred albedo image appears darker --> adjust brightness and contrast of albedo image
            test_clean_like = kornia.adjust_brightness(test_clean_like, 0.6)

        # report to visdom
        self.visdom_reporter.plot_finegrain_loss(str(constants.COLOR_TRANSFER_VERSION) +str(constants.ITERATION), iteration, self.losses_dict, self.caption_dict)
        self.visdom_reporter.plot_image(dirty_tensor, "Training Dirty images - "+str(constants.COLOR_TRANSFER_VERSION) +str(constants.ITERATION))
        self.visdom_reporter.plot_image(clean_tensor, "Training Clean images - " +str(constants.COLOR_TRANSFER_VERSION) +str(constants.ITERATION))
        self.visdom_reporter.plot_image(clean_like, "Training Clean-like images - "+str(constants.COLOR_TRANSFER_VERSION) +str(constants.ITERATION))
        self.visdom_reporter.plot_image(test_dirty_tensor, "Test Dirty images - " +str(constants.COLOR_TRANSFER_VERSION) +str(constants.ITERATION))
        self.visdom_reporter.plot_image(test_dirty_like, "Test Dirty-like images - "+str(constants.COLOR_TRANSFER_VERSION) +str(constants.ITERATION))
        self.visdom_reporter.plot_image(test_clean_tensor, "Test Clean images - "+str(constants.COLOR_TRANSFER_VERSION) +str(constants.ITERATION))
        self.visdom_reporter.plot_image(test_clean_like, "Test Clean-like images - "+str(constants.COLOR_TRANSFER_VERSION) +str(constants.ITERATION))

    def visdom_infer(self, test_dirty_tensor, caption_dirty, caption_clean):
        with torch.no_grad():
            test_clean_like = self.G_A(test_dirty_tensor)
            # inferred albedo image appears darker --> adjust brightness and contrast of albedo image
            test_clean_like = kornia.adjust_brightness(test_clean_like, 0.6)

        self.visdom_reporter.plot_image(test_dirty_tensor, caption_dirty + " - " +str(constants.COLOR_TRANSFER_VERSION) +str(constants.ITERATION))
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
        self.G_B.load_state_dict(checkpoint[constants.GENERATOR_KEY + "B"])
        self.D_A.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + "A"])
        self.D_B.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + "B"])
        self.optimizerG.load_state_dict(checkpoint[constants.GENERATOR_KEY + constants.OPTIMIZER_KEY])
        self.optimizerD.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + constants.OPTIMIZER_KEY])
        self.schedulerG.load_state_dict(checkpoint[constants.GENERATOR_KEY + "scheduler"])
        self.schedulerD.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + "scheduler"])

    def save_states(self, epoch, iteration):
        save_dict = {'epoch': epoch, 'iteration': iteration}
        netGA_state_dict = self.G_A.state_dict()
        netGB_state_dict = self.G_B.state_dict()
        netDA_state_dict = self.D_A.state_dict()
        netDB_state_dict = self.D_B.state_dict()

        optimizerG_state_dict = self.optimizerG.state_dict()
        optimizerD_state_dict = self.optimizerD.state_dict()

        schedulerG_state_dict = self.schedulerG.state_dict()
        schedulerD_state_dict = self.schedulerD.state_dict()

        save_dict[constants.GENERATOR_KEY + "A"] = netGA_state_dict
        save_dict[constants.GENERATOR_KEY + "B"] = netGB_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + "A"] = netDA_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + "B"] = netDB_state_dict

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
            self.losses_dict[constants.IDENTITY_LOSS_KEY].clear()
            self.losses_dict[constants.LIKENESS_LOSS_KEY].clear()
            self.losses_dict[constants.SMOOTHNESS_LOSS_KEY].clear()
            self.losses_dict[constants.G_ADV_LOSS_KEY].clear()
            self.losses_dict[constants.D_A_FAKE_LOSS_KEY].clear()
            self.losses_dict[constants.D_A_REAL_LOSS_KEY].clear()
            self.losses_dict[constants.D_B_FAKE_LOSS_KEY].clear()
            self.losses_dict[constants.D_B_REAL_LOSS_KEY].clear()
            self.losses_dict[constants.CYCLE_LOSS_KEY].clear()
