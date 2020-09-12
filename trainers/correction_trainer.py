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

class CorrectionTrainer:
    
    def __init__(self, gan_version, gan_iteration, gpu_device, gen_blocks, lr = 0.0002):
        self.gpu_device = gpu_device
        self.lr = lr
        self.gan_version = gan_version
        self.gan_iteration = gan_iteration
        self.visdom_reporter = plot_utils.VisdomReporter()
        self.initialize_dict()
        
        self.G_A = cg.Generator(input_nc = 1, output_nc = 3, n_residual_blocks=gen_blocks).to(self.gpu_device)
        self.D_A = cg.Discriminator(input_nc = 3).to(self.gpu_device)
        self.optimizerG = torch.optim.Adam(itertools.chain(self.G_A.parameters()), lr = lr)
        self.optimizerD = torch.optim.Adam(itertools.chain(self.D_A.parameters()), lr = lr)
    
    def initialize_dict(self):
        #what to store in visdom?
        self.losses_dict = {}
        self.losses_dict[constants.G_LOSS_KEY] = []
        self.losses_dict[constants.D_OVERALL_LOSS_KEY] = []
        self.losses_dict[constants.LIKENESS_LOSS_KEY] = []
        self.losses_dict[constants.D_A_FAKE_LOSS_KEY] = []
        self.losses_dict[constants.D_A_REAL_LOSS_KEY] = []
        
        self.caption_dict = {}
        self.caption_dict[constants.G_LOSS_KEY] = "G loss per iteration"
        self.caption_dict[constants.D_OVERALL_LOSS_KEY] = "D loss per iteration"
        self.caption_dict[constants.LIKENESS_LOSS_KEY] = "Likeness loss per iteration"
        self.caption_dict[constants.D_A_FAKE_LOSS_KEY] = "D(A) fake loss per iteration"
        self.caption_dict[constants.D_A_REAL_LOSS_KEY] = "D(B) fake loss per iteration"
    
    def update_penalties(self, color_weight, adv_weight):
        #what penalties to use for losses?
        self.color_weight = color_weight
        self.adv_weight = adv_weight
        
        #save hyperparameters for bookeeping
        HYPERPARAMS_PATH = "checkpoint/" + constants.COLORIZER_VERSION + "_" + constants.ITERATION + ".config"
        with open(HYPERPARAMS_PATH, "w") as f:
            print("Version: ", constants.COLORIZER_CHECKPATH, file = f)
            print("Learning rate: ", str(self.lr), file = f)
            print("====================================", file = f)
            print("Adv weight: ", str(self.adv_weight), file = f)
            print("Color weight: ", str(self.color_weight), file = f)
    
    def adversarial_loss(self, pred, target):
        loss = nn.MSELoss()
        return loss(pred, target)
    
    def color_loss(self, pred, target):
        loss = nn.L1Loss()
        return loss(pred, target)
    
    def train(self, gray_tensor, colored_tensor):        
        #what to put to losses dict for visdom reporting?
        colored_like = self.G_A(gray_tensor)
        self.D_A.train()
        self.optimizerD.zero_grad()
        
        prediction = self.D_A(colored_tensor)
        noise_value = random.uniform(0.8, 1.0)
        real_tensor = torch.ones_like(prediction) * noise_value #add noise value to avoid perfect predictions for real
        fake_tensor = torch.zeros_like(prediction)
        
        D_A_real_loss = self.adversarial_loss(self.D_A(colored_tensor), real_tensor) * self.adv_weight
        D_A_fake_loss = self.adversarial_loss(self.D_A(colored_like.detach()), fake_tensor) * self.adv_weight
        errD = D_A_real_loss + D_A_fake_loss
        if(errD.item() > 0.1):
            errD.backward()
            self.optimizerD.step()
        
        self.G_A.train()
        self.optimizerG.zero_grad()
        
        color_loss = self.color_loss(self.G_A(gray_tensor), colored_tensor) * self.color_weight
        prediction = self.D_A(self.G_A(gray_tensor))
        real_tensor = torch.ones_like(prediction)
        adv_loss = self.adversarial_loss(prediction, real_tensor) * self.adv_weight
        
        errG = color_loss + adv_loss
        errG.backward()
        self.optimizerG.step()
        
        #what to put to losses dict for visdom reporting?
        self.losses_dict[constants.G_LOSS_KEY].append(errG.item())
        self.losses_dict[constants.D_OVERALL_LOSS_KEY].append(errD.item())
        self.losses_dict[constants.LIKENESS_LOSS_KEY].append(color_loss.item())
        self.losses_dict[constants.D_A_FAKE_LOSS_KEY].append(D_A_fake_loss.item())
        self.losses_dict[constants.D_A_REAL_LOSS_KEY].append(D_A_real_loss.item())
    
    def visdom_report(self, iteration, test_gray, test_color):
        with torch.no_grad():
            test_color_like = self.G_A(test_gray)
        
        #report to visdom
        self.visdom_reporter.plot_finegrain_loss("colorization_loss", iteration, self.losses_dict, self.caption_dict)
        self.visdom_reporter.plot_image(test_gray, "Test Gray images")
        self.visdom_reporter.plot_image(test_color, "Test Color images")
        self.visdom_reporter.plot_image(test_color_like, "Test Color-like images")
    
    def load_saved_state(self, iteration, checkpoint, generator_key, disriminator_key, optimizer_key):
        self.iteration = iteration
        self.G_A.load_state_dict(checkpoint[generator_key + "A"])
        self.D_A.load_state_dict(checkpoint[disriminator_key + "A"])
        self.optimizerG.load_state_dict(checkpoint[generator_key + optimizer_key])
        self.optimizerD.load_state_dict(checkpoint[disriminator_key + optimizer_key])
    
    def save_states(self, epoch, iteration, path, generator_key, disriminator_key, optimizer_key):
        save_dict = {'epoch': epoch, 'iteration': iteration}
        netGA_state_dict = self.G_A.state_dict()
        netDA_state_dict = self.D_A.state_dict()
        
        optimizerG_state_dict = self.optimizerG.state_dict()
        optimizerD_state_dict = self.optimizerD.state_dict()
        
        save_dict[generator_key + "A"] = netGA_state_dict
        save_dict[disriminator_key + "A"] = netDA_state_dict
        
        save_dict[generator_key + optimizer_key] = optimizerG_state_dict
        save_dict[disriminator_key + optimizer_key] = optimizerD_state_dict
    
        torch.save(save_dict, path)
        print("Saved model state: %s Epoch: %d" % (len(save_dict), (epoch + 1)))