# -*- coding: utf-8 -*-
# Template trainer. Do not use this for actual training.

import os
from model import ffa_net as ffa
from model import latent_network
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

class FFATrainer:
    
    def __init__(self, gan_version, gan_iteration, gpu_device, blocks = 19, lr = 0.0002):
        self.gpu_device = gpu_device
        self.lr = lr
        self.gan_version = gan_version
        self.gan_iteration = gan_iteration
        self.G = ffa.FFA(gps = 3, blocks = blocks).to(self.gpu_device)
        self.optimizerG = torch.optim.Adam(itertools.chain(self.G.parameters()), lr=self.lr)

        self.LN = latent_network.LatentNetwork().to(self.gpu_device)
        self.visdom_reporter = plot_utils.VisdomReporter()
        self.initialize_dict()
        
    
    def initialize_dict(self):
        # what to store in visdom?
        self.losses_dict = {}
        self.losses_dict[constants.G_LOSS_KEY] = []

        self.caption_dict = {}
        self.caption_dict[constants.G_LOSS_KEY] = "Clarity loss per iteration"
        
    def clarity_loss(self, pred, target):
        loss = nn.L1Loss()
        return loss(pred, target)

    def update_penalties(self, l1_weight):
        #what penalties to use for losses?
        self.l1_weight = l1_weight

    def compute_z_signal(self, value, batch_size, image_size):
        z_size = (int(image_size[0] / 16), int(image_size[1] / 16))
        z_signal = torch.randn((batch_size, 1, z_size[0], z_size[1])).to(self.gpu_device)
        z_signal = z_signal.new_full((batch_size, 1, z_size[0], z_size[1]), value)

        return z_signal

    def train(self, hazy_tensor, clean_tensor):
        self.G.train()
        self.optimizerG.zero_grad()

        z_signal = self.compute_z_signal(np.random.uniform(-1.0, 1.0), np.shape(hazy_tensor)[0], constants.PATCH_IMAGE_SIZE)
        #print("Z signal shape: ", np.shape(z_signasl))
        latent_vector = self.LN(z_signal)
        #print("Latent vector shape: ", np.shape(latent_vector))

        clean_like = self.G(hazy_tensor, latent_vector)
        clarity_loss = self.clarity_loss(clean_like, clean_tensor) * self.l1_weight
        clarity_loss.backward()

        self.optimizerG.step()
        #what to put to losses dict for visdom reporting?
        self.losses_dict[constants.G_LOSS_KEY].append(clarity_loss.item())
    
    def visdom_report(self, iteration, hazy_tensor, clean_tensor, hazy_test, clean_test, vemon_tensor):
        with torch.no_grad():
            # report to visdom
            clean_like = self.G(hazy_tensor, self.LN(self.compute_z_signal(np.random.uniform(-1.0, 1.0), np.shape(hazy_tensor)[0], constants.PATCH_IMAGE_SIZE)))
            test_clean_like = self.G(hazy_test, self.LN(self.compute_z_signal(np.random.uniform(-1.0, 1.0), np.shape(hazy_test)[0], constants.TEST_IMAGE_SIZE)))
            vemon_clean = self.G(vemon_tensor, self.LN(self.compute_z_signal(np.random.uniform(-1.0, 1.0), np.shape(vemon_tensor)[0], constants.TEST_IMAGE_SIZE)))

            self.visdom_reporter.plot_finegrain_loss("dehazing_loss", iteration, self.losses_dict, self.caption_dict)
            self.visdom_reporter.plot_image(hazy_tensor, "Training Hazy images")
            self.visdom_reporter.plot_image(clean_like, "Training Clean-like images")
            self.visdom_reporter.plot_image(clean_tensor, "Training Clean images")

            self.visdom_reporter.plot_image(hazy_test, "Test Hazy images")
            self.visdom_reporter.plot_image(test_clean_like, "Test Clean-like images")
            self.visdom_reporter.plot_image(clean_test, "Test Clean images")

            self.visdom_reporter.plot_image(vemon_tensor, "Vemon Hazy Images")
            self.visdom_reporter.plot_image(vemon_clean, "Vemon Clean Images")
    
    def load_saved_state(self, iteration, checkpoint, model_key, optimizer_key):
        self.iteration = iteration
        self.G.load_state_dict(checkpoint[model_key])
        self.optimizerG.load_state_dict(checkpoint[optimizer_key])
    
    def save_states(self, epoch, iteration, path, model_key, optimizer_key):
        save_dict = {'epoch': epoch, 'iteration': iteration}
        netGA_state_dict = self.G.state_dict()
        optimizerG_state_dict = self.optimizerG.state_dict()
        save_dict[model_key] = netGA_state_dict
        save_dict[optimizer_key] = optimizerG_state_dict

        torch.save(save_dict, path)
        print("Saved model state: %s Epoch: %d" % (len(save_dict), (epoch + 1)))