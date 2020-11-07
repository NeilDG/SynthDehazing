# -*- coding: utf-8 -*-
# Template trainer. Do not use this for actual training.

import os
import torchvision.transforms as transforms
from model import vanilla_cycle_gan as cg
from model import style_transfer_gan as sg
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
from utils import tensor_utils
from custom_losses import ssim_loss

class CorrectionTrainer:
    
    def __init__(self, gan_version, gan_iteration, gpu_device, g_lr, d_lr):
        self.gpu_device = gpu_device
        self.g_lr = g_lr
        self.d_lr = d_lr
        self.gan_version = gan_version
        self.gan_iteration = gan_iteration
        self.visdom_reporter = plot_utils.VisdomReporter()
        self.initialize_dict()
        
        self.G_A = sg.Generator(input_nc = 3, output_nc = 3).to(self.gpu_device)
        self.G_B = sg.Generator(input_nc = 3, output_nc = 1).to(self.gpu_device)
        self.D_A = cg.Discriminator(input_nc = 3).to(self.gpu_device) #use CycleGAN's discriminator
        self.D_B = cg.Discriminator(input_nc = 1).to(self.gpu_device)
        self.optimizerG = torch.optim.Adam(itertools.chain(self.G_A.parameters(), self.G_B.parameters()), lr = self.g_lr)
        self.optimizerD = torch.optim.Adam(itertools.chain(self.D_A.parameters(), self.D_B.parameters()), lr = self.d_lr)
    
    def initialize_dict(self):
        #what to store in visdom?
        self.losses_dict = {}
        self.losses_dict[constants.G_LOSS_KEY] = []
        self.losses_dict[constants.D_OVERALL_LOSS_KEY] = []
        self.losses_dict[constants.LIKENESS_LOSS_KEY] = []
        self.losses_dict[constants.D_A_FAKE_LOSS_KEY] = []
        self.losses_dict[constants.D_A_REAL_LOSS_KEY] = []
        self.losses_dict[constants.D_B_FAKE_LOSS_KEY] = []
        self.losses_dict[constants.D_B_REAL_LOSS_KEY] = []
        self.losses_dict[constants.CYCLE_LOSS_KEY] = []
        
        self.caption_dict = {}
        self.caption_dict[constants.G_LOSS_KEY] = "G loss per iteration"
        self.caption_dict[constants.D_OVERALL_LOSS_KEY] = "D loss per iteration"
        self.caption_dict[constants.CYCLE_LOSS_KEY] = "Cycle loss per iteration"
        self.caption_dict[constants.LIKENESS_LOSS_KEY] = "Color loss per iteration"
        self.caption_dict[constants.D_A_FAKE_LOSS_KEY] = "D(A) fake loss per iteration"
        self.caption_dict[constants.D_A_REAL_LOSS_KEY] = "D(A) real loss per iteration"
        self.caption_dict[constants.D_B_FAKE_LOSS_KEY] = "D(B) fake loss per iteration"
        self.caption_dict[constants.D_B_REAL_LOSS_KEY] = "D(B) real loss per iteration"
    
    def update_penalties(self, color_weight, cycle_weight, adv_weight):
        #what penalties to use for losses?
        self.color_weight = color_weight
        self.adv_weight = adv_weight
        self.cycle_weight = cycle_weight
        
        #save hyperparameters for bookeeping
        HYPERPARAMS_PATH = "checkpoint/" + constants.COLORIZER_VERSION + "_" + constants.ITERATION + ".config"
        with open(HYPERPARAMS_PATH, "w") as f:
            print("Version: ", constants.COLORIZER_CHECKPATH, file = f)
            print("Learning rate G: ", str(self.g_lr), file = f)
            print("Learning rate D: ", str(self.d_lr), file = f)
            print("====================================", file = f)
            print("Adv weight: ", str(self.adv_weight), file = f)
            print("Color weight: ", str(self.color_weight), file = f)
            print("Cycle weight: ", str(self.cycle_weight), file = f)
    
    def adversarial_loss(self, pred, target):
        loss = nn.MSELoss()
        return loss(pred, target)
    
    def color_loss(self, pred, target):
        # loss = ssim_loss.SSIM()
        # return 1 - loss(pred, target)
        loss = nn.MSELoss()
        return loss(pred, target)
    
    def cycle_loss(self, pred, target):
        loss = nn.L1Loss()
        return loss(pred, target)
    
    def train(self, gray_tensor, yuv_tensor):
        #replace
        (y, u, v) = torch.chunk(yuv_tensor.transpose(0,1), 3)
        input_tensor = torch.cat((gray_tensor.transpose(0,1), u, v)).transpose(0, 1)
        yuv_tensor_like = self.G_A(input_tensor) #refined color
        gray_like = self.G_B(yuv_tensor) #reverse color
        
        self.D_A.train()
        self.D_B.train()
        self.optimizerD.zero_grad()
        
        prediction = self.D_A(yuv_tensor)
        noise_value = random.uniform(0.8, 1.0)
        real_tensor = torch.ones_like(prediction) * noise_value #add noise value to avoid perfect predictions for real
        fake_tensor = torch.zeros_like(prediction)
        
        D_A_real_loss = self.adversarial_loss(self.D_A(yuv_tensor), real_tensor) * self.adv_weight
        D_A_fake_loss = self.adversarial_loss(self.D_A(yuv_tensor_like.detach()), fake_tensor) * self.adv_weight
        
        prediction = self.D_B(gray_tensor)
        real_tensor = torch.ones_like(prediction) * noise_value #add noise value to avoid perfect predictions for real
        fake_tensor = torch.zeros_like(prediction)
        
        D_B_real_loss = self.adversarial_loss(self.D_B(gray_tensor), real_tensor) * self.adv_weight
        D_B_fake_loss = self.adversarial_loss(self.D_B(gray_like.detach()), fake_tensor) * self.adv_weight
        
        errD = D_A_real_loss + D_A_fake_loss + D_B_real_loss + D_B_fake_loss
        if(errD.item() > 0.1):
            errD.backward()
            self.optimizerD.step()
        
        self.G_A.train()
        self.G_B.train()
        self.optimizerG.zero_grad()

        (y, u, v) = torch.chunk(yuv_tensor.transpose(0,1), 3)
        input_tensor = torch.cat((gray_tensor.transpose(0, 1), u, v)).transpose(0, 1)
        yuv_tensor_like = self.G_A(input_tensor) #refined color
        gray_like = self.G_B(yuv_tensor) #reverse color
        
        color_loss = self.color_loss(yuv_tensor_like, yuv_tensor) * self.color_weight
        
        A_cycle_loss = self.cycle_loss(self.G_B(self.G_A(input_tensor)), gray_tensor) * self.cycle_weight
        input_tensor = torch.cat((self.G_B(yuv_tensor).transpose(0, 1), u, v)).transpose(0, 1)
        B_cycle_loss = self.cycle_loss(self.G_A(input_tensor), yuv_tensor) * self.cycle_weight
        
        prediction = self.D_B(gray_like)
        real_tensor = torch.ones_like(prediction)
        B_adv_loss = self.adversarial_loss(prediction, real_tensor) * self.adv_weight
        
        prediction = self.D_A(yuv_tensor_like)
        real_tensor = torch.ones_like(prediction)
        A_adv_loss = self.adversarial_loss(prediction, real_tensor) * self.adv_weight
        
        errG = color_loss + A_adv_loss + B_adv_loss + A_cycle_loss + B_cycle_loss
        errG.backward()
        self.optimizerG.step()
        
        #what to put to losses dict for visdom reporting?
        self.losses_dict[constants.G_LOSS_KEY].append(errG.item())
        self.losses_dict[constants.D_OVERALL_LOSS_KEY].append(errD.item())
        self.losses_dict[constants.LIKENESS_LOSS_KEY].append(color_loss.item())
        self.losses_dict[constants.D_A_FAKE_LOSS_KEY].append(D_A_fake_loss.item())
        self.losses_dict[constants.D_A_REAL_LOSS_KEY].append(D_A_real_loss.item())
        self.losses_dict[constants.D_B_FAKE_LOSS_KEY].append(D_B_fake_loss.item())
        self.losses_dict[constants.D_B_REAL_LOSS_KEY].append(D_B_real_loss.item())
        self.losses_dict[constants.CYCLE_LOSS_KEY].append(A_cycle_loss.item() + B_cycle_loss.item())
    
    def visdom_report(self, iteration, gray_tensor, yuv_tensor):
        with torch.no_grad():
            (y, u, v) = torch.chunk(yuv_tensor.transpose(0, 1), 3)
            input_tensor = torch.cat((gray_tensor.transpose(0, 1), u, v)).transpose(0, 1)
            refined_tensor = tensor_utils.yuv_to_rgb(self.G_A(input_tensor)) #refined color
            #refined_tensor = tensor_utils.yuv_to_rgb(input_tensor)
        
        #report to visdom
        self.visdom_reporter.plot_finegrain_loss("colorization_loss", iteration, self.losses_dict, self.caption_dict)
        self.visdom_reporter.plot_image(refined_tensor, "Test Refined images")
        self.visdom_reporter.plot_image(tensor_utils.yuv_to_rgb(yuv_tensor), "Test Original images")
    
    def load_saved_state(self, iteration, checkpoint, generator_key, disriminator_key, optimizer_key):
        self.iteration = iteration
        self.G_A.load_state_dict(checkpoint[generator_key + "A"])
        self.D_A.load_state_dict(checkpoint[disriminator_key + "A"])
        self.G_B.load_state_dict(checkpoint[generator_key + "B"])
        self.D_B.load_state_dict(checkpoint[disriminator_key + "B"])
        self.optimizerG.load_state_dict(checkpoint[generator_key + optimizer_key])
        self.optimizerD.load_state_dict(checkpoint[disriminator_key + optimizer_key])
    
    def save_states(self, epoch, iteration, path, generator_key, disriminator_key, optimizer_key):
        save_dict = {'epoch': epoch, 'iteration': iteration}
        netGA_state_dict = self.G_A.state_dict()
        netDA_state_dict = self.D_A.state_dict()
        netGB_state_dict = self.G_B.state_dict()
        netDB_state_dict = self.D_B.state_dict()
        
        optimizerG_state_dict = self.optimizerG.state_dict()
        optimizerD_state_dict = self.optimizerD.state_dict()
        
        save_dict[generator_key + "A"] = netGA_state_dict
        save_dict[disriminator_key + "A"] = netDA_state_dict
        save_dict[generator_key + "B"] = netGB_state_dict
        save_dict[disriminator_key + "B"] = netDB_state_dict
        
        save_dict[generator_key + optimizer_key] = optimizerG_state_dict
        save_dict[disriminator_key + optimizer_key] = optimizerD_state_dict
    
        torch.save(save_dict, path)
        print("Saved model state: %s Epoch: %d" % (len(save_dict), (epoch + 1)))