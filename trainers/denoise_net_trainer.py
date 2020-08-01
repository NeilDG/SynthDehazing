# -*- coding: utf-8 -*-
# Template trainer. Do not use this for actual training.

import os
from model import vanilla_cycle_gan as cg
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
from custom_losses import ssim_loss

class DenoiseTrainer:
    
    def __init__(self, gan_version, gan_iteration, gpu_device, gen_blocks, lr = 0.0005):
        self.gpu_device = gpu_device
        self.lr = lr
        self.gan_version = gan_version
        self.gan_iteration = gan_iteration
        self.visdom_reporter = plot_utils.VisdomReporter()
        
        self.G = cg.Generator(n_residual_blocks=gen_blocks).to(self.gpu_device)
        self.D = cg.Discriminator().to(self.gpu_device)
        
        self.optimizerG = torch.optim.Adam(self.G.parameters(), lr = lr)
        self.optimizerD = torch.optim.Adam(self.D.parameters(), lr = lr)
        self.initialize_dict()
        
    
    def initialize_dict(self):
        #what to store in visdom?
        self.losses_dict = {}
        self.losses_dict[constants.G_LOSS_KEY] = []
        self.losses_dict[constants.D_OVERALL_LOSS_KEY] = []
        self.losses_dict[constants.IDENTITY_LOSS_KEY] = []
        self.losses_dict[constants.LIKENESS_LOSS_KEY] = []
        self.losses_dict[constants.G_ADV_LOSS_KEY] = []
        self.losses_dict[constants.D_A_FAKE_LOSS_KEY] = []
        self.losses_dict[constants.D_A_REAL_LOSS_KEY] = []
    
    def update_penalties(self, adv_weight, id_weight, likeness_weight):
        #what penalties to use for losses?
        self.adv_weight = adv_weight
        self.id_weight = id_weight
        self.likeness_weight = likeness_weight
        
    
    def adversarial_loss(self, pred, target):
        loss = nn.MSELoss()
        return loss(pred, target)
    
    def identity_loss(self, pred, target):
        loss = nn.L1Loss()
        return loss(pred, target)
    
    def likeness_loss(self, pred, target):
        #loss = ssim_loss.SSIM()
        #return 1 - loss(pred, target)
        loss = nn.MSELoss()
        return loss(pred, target)
    
    def train(self, dirty_tensor, clean_tensor):
        clean_like = self.G(dirty_tensor)
        
        self.D.train()
        self.optimizerD.zero_grad()
        
        prediction = self.D(clean_tensor)
        noise_value = random.uniform(0.8, 1.0)
        real_tensor = torch.ones_like(prediction) * noise_value #add noise value to avoid perfect predictions for real
        fake_tensor = torch.zeros_like(prediction)
        
        D_A_real_loss = self.adversarial_loss(self.D(clean_tensor), real_tensor) * self.adv_weight
        D_A_fake_loss = self.adversarial_loss(self.D(clean_like.detach()), fake_tensor) * self.adv_weight
        errD = D_A_real_loss + D_A_fake_loss
        if(errD.item() > 0.1):
            errD.backward()
            self.optimizerD.step()
        
        self.G.train()
        self.optimizerG.zero_grad()
        
        identity_like = self.G(clean_tensor)
        clean_like = self.G(dirty_tensor)
        
        identity_loss = self.identity_loss(identity_like, clean_tensor) * self.id_weight
        likeness_loss = self.likeness_loss(clean_like, clean_tensor) * self.likeness_weight
        
        prediction = self.D(self.G(dirty_tensor))
        real_tensor = torch.ones_like(prediction)
        adv_loss = self.adversarial_loss(prediction, real_tensor) * self.adv_weight
        
        errG = identity_loss + likeness_loss + adv_loss
        errG.backward()
        self.optimizerG.step()
        
        #what to put to losses dict for visdom reporting?
        self.losses_dict[constants.G_LOSS_KEY].append(errG.item())
        self.losses_dict[constants.D_OVERALL_LOSS_KEY].append(errD.item())
        self.losses_dict[constants.IDENTITY_LOSS_KEY].append(identity_loss.item())
        self.losses_dict[constants.LIKENESS_LOSS_KEY].append(likeness_loss.item())
        self.losses_dict[constants.G_ADV_LOSS_KEY].append(adv_loss.item())
        self.losses_dict[constants.D_A_FAKE_LOSS_KEY].append(D_A_fake_loss.item())
        self.losses_dict[constants.D_A_REAL_LOSS_KEY].append(D_A_real_loss.item())
    
    def visdom_report(self, iteration, dirty_tensor, clean_tensor, test_dirty_tensor, test_clean_tensor):
        with torch.no_grad():
            clean_like = self.G(dirty_tensor)
            test_clean_like = self.G(test_dirty_tensor)
        
        #report to visdom
        self.visdom_reporter.plot_finegrain_loss(iteration, self.losses_dict)
        self.visdom_reporter.plot_image(dirty_tensor, clean_tensor, clean_like)
        self.visdom_reporter.plot_denoise_test_image(test_dirty_tensor, test_clean_tensor, test_clean_like)
    
    def infer(self, dirty_tensor, file_number):
        LOCATION = os.getcwd() + "/figures/"
        with torch.no_grad():
            clean_like = self.G(dirty_tensor).detach()
        
        #resize tensors for better viewing
        resized_normal = nn.functional.interpolate(dirty_tensor, scale_factor = 4.0, mode = "bilinear", recompute_scale_factor = True)
        resized_fake = nn.functional.interpolate(clean_like, scale_factor = 4.0, mode = "bilinear", recompute_scale_factor = True)
        
        print("New shapes: %s %s" % (np.shape(resized_normal), np.shape(resized_fake)))
        
        fig, ax = plt.subplots(2, 1)
        fig.set_size_inches(constants.FIG_SIZE)
        fig.tight_layout()
        
        ims = np.transpose(vutils.make_grid(resized_normal, nrow = 8, padding=2, normalize=True).cpu(),(1,2,0))
        ax[0].set_axis_off()
        ax[0].imshow(ims)
        
        ims = np.transpose(vutils.make_grid(resized_fake, nrow = 8, padding=2, normalize=True).cpu(),(1,2,0))
        ax[1].set_axis_off()
        ax[1].imshow(ims)
        
        plt.subplots_adjust(left = 0.06, wspace=0.0, hspace=0.15) 
        plt.savefig(LOCATION + "result_" + str(file_number) + ".png")
        plt.show()
        
    def load_saved_state(self, iteration, checkpoint, generator_key, disriminator_key, optimizer_key):
        self.G.load_state_dict(checkpoint[generator_key + "A"])
        self.D.load_state_dict(checkpoint[disriminator_key + "A"])
        self.optimizerG.load_state_dict(checkpoint[generator_key + optimizer_key])
        self.optimizerD.load_state_dict(checkpoint[disriminator_key + optimizer_key])
    
    def save_states(self, epoch, iteration, path, generator_key, disriminator_key, optimizer_key):
        save_dict = {'epoch': epoch, 'iteration': iteration}
        netGA_state_dict = self.G.state_dict()
        netDA_state_dict = self.D.state_dict()
        
        optimizerG_state_dict = self.optimizerG.state_dict()
        optimizerD_state_dict = self.optimizerD.state_dict()
        
        save_dict[generator_key + "A"] = netGA_state_dict
        save_dict[disriminator_key + "A"] = netDA_state_dict
        
        save_dict[generator_key + optimizer_key] = optimizerG_state_dict
        save_dict[disriminator_key + optimizer_key] = optimizerD_state_dict
    
        torch.save(save_dict, path)
        print("Saved model state: %s Epoch: %d" % (len(save_dict), (epoch + 1)))