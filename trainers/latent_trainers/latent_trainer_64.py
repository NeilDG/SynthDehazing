# -*- coding: utf-8 -*-
# Used for training a GAN using a Z latent vector.

import os
from model import latent_network
import constants
import torch
import random
import itertools
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision.utils as vutils
from utils import logger, tensor_utils
from utils import plot_utils


class LatentTrainer:
    
    def __init__(self, gan_version, gan_iteration, gpu_device, g_lr, d_lr):
        self.gpu_device = gpu_device
        self.g_lr = g_lr
        self.d_lr = d_lr
        self.gan_version = gan_version
        self.gan_iteration = gan_iteration
        self.LN = latent_network.LatentNetworkScaler().to(self.gpu_device)
        self.D_A = latent_network.Discriminator().to(self.gpu_device)  # use CycleGAN's discriminator
        self.optimizerG = torch.optim.Adam(itertools.chain(self.LN.parameters()), lr = self.g_lr, weight_decay=1e-5)
        self.optimizerD = torch.optim.Adam(itertools.chain(self.D_A.parameters()), lr = self.d_lr, weight_decay =1e-5)
        self.visdom_reporter = plot_utils.VisdomReporter()
        self.initialize_dict()

        latent_checkpoint = torch.load(constants.LATENT_CHECKPATH)
        self.LN_32 = latent_network.LatentNetwork().to(self.gpu_device)
        self.LN_32.load_state_dict(latent_checkpoint[constants.GENERATOR_KEY])
    
    def initialize_dict(self):
        #what to store in visdom?
        self.losses_dict = {}
        self.losses_dict[constants.G_LOSS_KEY] = []
        self.losses_dict[constants.D_OVERALL_LOSS_KEY] = []
        self.losses_dict[constants.G_ADV_LOSS_KEY] = []
        self.losses_dict[constants.LIKENESS_LOSS_KEY] = []

        self.caption_dict = {}
        self.caption_dict[constants.G_LOSS_KEY] = "G loss per iteration"
        self.caption_dict[constants.D_OVERALL_LOSS_KEY] = "D loss per iteration"
        self.caption_dict[constants.G_ADV_LOSS_KEY] = "G adv loss per iteration"
        self.caption_dict[constants.LIKENESS_LOSS_KEY] = "Likeness loss per iteration"

    def adversarial_loss(self, pred, target):
        loss = nn.BCELoss()
        return loss(pred, target)

    def likeness_loss(self, pred, target):
        loss = nn.L1Loss()
        return loss(pred, target)

    def update_penalties(self, adv_weight, likeness_weight):
        #what penalties to use for losses?
        self.adv_weight = adv_weight
        self.likeness_weight = likeness_weight

        # save hyperparameters for bookeeping
        HYPERPARAMS_PATH = "checkpoint/" + constants.LATENT_VERSION + "_" + constants.ITERATION + ".config"
        with open(HYPERPARAMS_PATH, "w") as f:
            print("Version: ", constants.LATENT_CHECKPATH, file=f)
            print("Learning rate for G: ", str(self.g_lr), file=f)
            print("Learning rate for D: ", str(self.d_lr), file=f)
            print("====================================", file=f)
            print("Adv weight: ", str(self.adv_weight), file=f)
            print("Likeness weight: ", str(self.likeness_weight), file=f)
    
    def train(self, img_tensor):
        #produce image from previous model
        z_signal = tensor_utils.compute_z_signal(np.random.uniform(-5000.0, 5000.0), np.shape(img_tensor)[0], constants.PATCH_IMAGE_SIZE).to(self.gpu_device)
        img_32 = self.LN_32(z_signal)

        #create new signal and concat with previous generated image
        z_signal = tensor_utils.compute_z_signal_concat(np.random.uniform(-5000.0, 5000.0), np.shape(img_tensor)[0], (32, 32)).to(self.gpu_device)
        img_like = self.LN(img_32, z_signal)

        self.D_A.train()
        self.optimizerD.zero_grad()

        prediction = self.D_A(img_tensor)
        real_tensor = torch.ones_like(prediction)
        fake_tensor = torch.zeros_like(prediction)

        #print("Shape: ", np.shape(img_like), np.shape(img_tensor))
        D_A_real_loss = self.adversarial_loss(self.D_A(img_tensor), real_tensor) *  self.adv_weight
        D_A_fake_loss = self.adversarial_loss(self.D_A(img_like.detach()), fake_tensor) * self.adv_weight

        errD = D_A_real_loss + D_A_fake_loss
        errD.backward()
        self.optimizerD.step()

        self.LN.train()
        self.optimizerG.zero_grad()
        img_like = self.LN(img_32, z_signal)

        prediction = self.D_A(img_like)
        real_tensor = torch.ones_like(prediction)
        A_adv_loss = self.adversarial_loss(prediction, real_tensor) * self.adv_weight
        A_likeness_loss = self.likeness_loss(img_like, img_tensor) * self.likeness_weight
        errG = A_adv_loss + A_likeness_loss
        errG.backward()
        self.optimizerG.step()

        #what to put to losses dict for visdom reporting?
        self.losses_dict[constants.G_LOSS_KEY].append(errG.item())
        self.losses_dict[constants.D_OVERALL_LOSS_KEY].append(errD.item())
        self.losses_dict[constants.G_ADV_LOSS_KEY].append(A_adv_loss.item())
        self.losses_dict[constants.LIKENESS_LOSS_KEY].append(A_likeness_loss.item())
    
    def visdom_report(self, iteration, train_img_tensor, test_img_tensor):
        with torch.no_grad():
            # produce image from previous model
            z_signal_32 = tensor_utils.compute_z_signal(np.random.uniform(-5000.0, 5000.0), np.shape(train_img_tensor)[0],constants.PATCH_IMAGE_SIZE).to(self.gpu_device)
            img_32 = self.LN_32(z_signal_32)

            #infer
            z_signal_64 = tensor_utils.compute_z_signal_concat(np.random.uniform(-5000.0, 5000.0), np.shape(train_img_tensor)[0], (32, 32)).to(self.gpu_device)
            train_img_like = self.LN(img_32, z_signal_64)

            #z_signal_test = tensor_utils.compute_z_signal_concat(np.random.uniform(-5000.0, 5000.0), np.shape(test_img_tensor)[0], (np.shape(test_img_tensor)[2], np.shape(test_img_tensor)[3])).to(self.gpu_device)
            #test_img_like = self.LN(test_img_tensor, z_signal_test)
        
        #report to visdom
        self.visdom_reporter.plot_finegrain_loss("Creation loss", iteration, self.losses_dict, self.caption_dict)
        self.visdom_reporter.plot_image((train_img_like), "Generated Training Images")
        self.visdom_reporter.plot_image((train_img_tensor), "Training images")
        #self.visdom_reporter.plot_image((test_img_like), "Generated Test Images")
        #self.visdom_reporter.plot_image((test_img_tensor), "Test Images")
    
    def load_saved_state(self, iteration, checkpoint, generator_key, discriminator_key, optimizer_key):
        self.iteration = iteration
        self.LN.load_state_dict(checkpoint[generator_key])
        self.D_A.load_state_dict(checkpoint[discriminator_key])
        self.optimizerG.load_state_dict(checkpoint[generator_key + optimizer_key])
        self.optimizerD.load_state_dict(checkpoint[discriminator_key + optimizer_key])
    
    def save_states(self, epoch, iteration, path, generator_key, discriminator_key, optimizer_key):
        save_dict = {'epoch': epoch, 'iteration': iteration}
        netLN_state_dict = self.LN.state_dict()
        netD_state_dict = self.D_A.state_dict()

        optimizerG_state_dict = self.optimizerG.state_dict()
        optimizerD_state_dict = self.optimizerD.state_dict()

        save_dict[generator_key] = netLN_state_dict
        save_dict[discriminator_key] = netD_state_dict

        save_dict[generator_key + optimizer_key] = optimizerG_state_dict
        save_dict[discriminator_key + optimizer_key] = optimizerD_state_dict

        torch.save(save_dict, path)
        print("Saved model state: %s Epoch: %d" % (len(save_dict), (epoch + 1)))