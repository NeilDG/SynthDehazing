# -*- coding: utf-8 -*-
"""
Experiment on denoising training
Created on Wed Nov  6 19:41:58 2019

@author: delgallegon
"""

import os
#from model import new_style_transfer_gan as st
#from model import style_transfer_gan as st
from model import vanilla_cycle_gan as cg
from model import denoise_discriminator
from model import vgg_loss_model
from loaders import dataset_loader
import constants
import torch
import random
from torch import optim
import itertools
import numpy as np
import torchvision.models as models
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision.utils as vutils
import torchvision.transforms as transforms
from torch import autograd
from utils import tensor_utils
from utils import logger
from utils import plot_utils

#print = logger.log

class GANTrainer:
    def __init__(self, gan_version, gan_iteration, gpu_device, gen_blocks, disc_blocks, lr = 0.0002, weight_decay = 0.0, betas = (0.5, 0.999)):
        self.gpu_device = gpu_device
        self.lr = lr
        self.gan_version = gan_version
        self.gan_iteration = gan_iteration
        self.visdom_reporter = plot_utils.VisdomReporter()

        # self.G_A = st.Generator(gen_blocks).to(gpu_device) #use multistyle net as architecture
        # self.G_B = st.Generator(gen_blocks).to(gpu_device)
        # self.D_A = st.Discriminator(disc_blocks).to(gpu_device)
        self.G_A = cg.Generator().to(gpu_device) #use multistyle net as architecture
        self.G_B = cg.Generator().to(gpu_device)
        self.D_A = cg.Discriminator().to(gpu_device)
        #self.D_Features = [cg.FeatureDiscriminator(64).to(gpu_device), cg.FeatureDiscriminator(256).to(gpu_device)]
        
        #use VGG for extracting features
        self.vgg_loss = vgg_loss_model.VGGPerceptualLoss().to(gpu_device)
        
        print("Gen blocks set to %d. Disc blocks set to %d." %(gen_blocks, disc_blocks))
        print(self.G_B)
        print(self.D_A)
        #print(self.D_Features[0])
        #print(self.D_Features[1])
        
        self.optimizerG = torch.optim.Adam(itertools.chain(self.G_A.parameters(), self.G_B.parameters()), lr=lr, betas=betas, weight_decay=weight_decay)
        self.optimizerD = torch.optim.Adam(itertools.chain(self.D_A.parameters()), lr=lr, betas=betas, weight_decay=weight_decay)
        
        self.G_losses = []
        self.D_losses = []
        self.initialize_dict()
        
        self.iteration = 0
        self.identity_weight = 1.0; self.cycle_weight = 10.0; self.adv_weight = 1.0; self.tv_weight = 10.0; self.like_weight = 1.0;
    
    def initialize_dict(self):
        self.losses_dict = {}
        self.losses_dict[constants.G_LOSS_KEY] = []
        self.losses_dict[constants.IDENTITY_LOSS_KEY] = []
        self.losses_dict[constants.CYCLE_LOSS_KEY] = []
        self.losses_dict[constants.TV_LOSS_KEY] = []
        self.losses_dict[constants.ADV_LOSS_KEY] = []
        self.losses_dict[constants.PERCEP_LOSS_KEY] = []
        self.losses_dict[constants.D_FAKE_LOSS_KEY] = []
        self.losses_dict[constants.D_REAL_LOSS_KEY]  = []
        self.losses_dict[constants.D_OVERALL_LOSS_KEY]  = []
        
    def update_penalties(self, identity, cycle, adv, tv, gen_skips, disc_skips):
        self.identity_weight = identity
        self.cycle_weight = cycle
        self.adv_weight = adv
        self.tv_weight = tv
        self.gen_skips = gen_skips
        self.disc_skips = disc_skips
        print("Weights updated to the following: %f %f %f %f" % (self.identity_weight, self.cycle_weight, self.adv_weight, self.tv_weight))
        print("Skips updated to the following: %d %d" %(self.gen_skips, self.disc_skips))
        
    def adversarial_loss(self, pred, target):
        loss = nn.MSELoss()
        return loss(pred, target)
    
    def identity_loss(self, pred, target):
        loss = nn.L1Loss()
        return loss(pred, target)
    
    def cycle_loss(self, pred, target):
        loss = nn.MSELoss()
        return loss(pred, target)
    
    def likeness_loss(self, pred, target):
        loss = nn.L1Loss()
        return loss(pred, target)
    
    def tv_loss(self, yhat, y):
        bsize, chan, height, width = y.size()
        error = []
        dy = torch.abs(y[:,:,1:,:] - y[:,:,:-1,:])
        dyhat = torch.abs(yhat[:,:,1:,:] - yhat[:,:,:-1,:])
        error = torch.norm(dy - dyhat, 1)
        return error / height
    
    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]
    
    def tv_impose(self, x):
        batch_size = x.size()[0]
        count_h = x[:,:,1:,:].size()[1]*x[:,:,1:,:].size()[2]*x[:,:,1:,:].size()[3]
        count_w = x[:,:,:,1:].size()[1]*x[:,:,:,1:].size()[2]*x[:,:,:,1:].size()[3]
        h_x = x.size()[2]
        w_x = x.size()[3]
        
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        
        return 2 * (h_tv/count_h + w_tv/count_w) / batch_size
        
    # Input = image
    # Performs a discriminator forward-backward pass, then a generator forward-backward pass
    # a = vemon image
    # b = gta image
    def train(self, dirty_tensor, clean_tensor):
       
        #update D first
        clean_like = self.G_A(dirty_tensor)
        
        self.D_A.train()
        #self.D_Features[0].train(); self.D_Features[1].train()
        
        self.optimizerD.zero_grad()
        
        prediction = self.D_A(clean_like)
        noise_value = random.uniform(0.8, 1.0)
        real_tensor = torch.ones_like(prediction) * noise_value #add noise value to avoid perfect predictions for real
        fake_tensor = torch.zeros_like(prediction)
        
        D_A_real_loss = (self.adversarial_loss(self.D_A(clean_tensor), real_tensor)) * self.adv_weight
        D_A_fake_loss = self.adversarial_loss(self.D_A(clean_like.detach()), fake_tensor) * self.adv_weight
        
        #check discriminator features
        # prediction = self.D_Features[0](real_image_features[0])
        # real_tensor = torch.ones_like(prediction) * noise_value #add noise value to avoid perfect predictions for real
        # fake_tensor = torch.zeros_like(prediction)
        
        # D_Features_loss_real_1 = self.adversarial_loss(prediction, real_tensor) * 5.0
        # D_Features_loss_fake_1 = self.adversarial_loss(self.D_Features[0](fake_image_features[0]), fake_tensor)
        
        # prediction = self.D_Features[1](real_image_features[2])
        # real_tensor = torch.ones_like(prediction) * noise_value #add noise value to avoid perfect predictions for real
        # fake_tensor = torch.zeros_like(prediction)
        
        # D_Features_loss_real_2 = self.adversarial_loss(prediction, real_tensor) * 10.0
        # D_Features_loss_fake_2 = self.adversarial_loss(self.D_Features[1](fake_image_features[2]), fake_tensor)
           
        errD = D_A_real_loss + D_A_fake_loss
        if(self.iteration % self.disc_skips == 0 and errD.item() > 0.1): #only update discriminator every N iterations and if discriminator becomes worse
            errD.backward()
            self.optimizerD.step()
        
        #update G next
        self.G_A.train()
        self.G_B.train()
        self.optimizerG.zero_grad()
        
        identity_like = self.G_A(clean_tensor)
        dirty_like = self.G_B(clean_like)
        
        A_identity_loss = self.identity_loss(identity_like, clean_tensor) * self.identity_weight
        A_cycle_loss = self.cycle_loss(dirty_like, dirty_tensor) * self.cycle_weight
        A_tv_loss = self.tv_impose(clean_like) * self.tv_weight
        
        prediction = self.D_A(clean_like)
        real_tensor = torch.ones_like(prediction)
        adv_loss = self.adversarial_loss(prediction, real_tensor) * self.adv_weight
        
        clean_like = tensor_utils.preprocess_batch(clean_like)
        clean_tensor = tensor_utils.preprocess_batch(clean_tensor)
        percep_loss = self.vgg_loss(clean_like, clean_tensor)
        
        if(self.iteration % self.gen_skips == 0): #only update generator for cycle loss, every N iterations
            errG = A_identity_loss + A_cycle_loss + A_tv_loss + adv_loss + percep_loss
            errG.backward()
            self.optimizerG.step()
        else:
            errG = A_identity_loss + A_tv_loss + adv_loss
            errG.backward()
            self.optimizerG.step()
        
        
        # Save Losses for plotting later
        self.losses_dict[constants.G_LOSS_KEY].append(errG.item())
        self.losses_dict[constants.D_OVERALL_LOSS_KEY].append(errD.item())
        self.losses_dict[constants.IDENTITY_LOSS_KEY].append(A_identity_loss.item())
        self.losses_dict[constants.CYCLE_LOSS_KEY].append(A_cycle_loss.item())
        self.losses_dict[constants.TV_LOSS_KEY].append(A_tv_loss.item())
        self.losses_dict[constants.ADV_LOSS_KEY].append(adv_loss.item())
        self.losses_dict[constants.PERCEP_LOSS_KEY].append(percep_loss.item())
        self.losses_dict[constants.D_FAKE_LOSS_KEY].append(D_A_fake_loss.item())
        self.losses_dict[constants.D_REAL_LOSS_KEY].append(D_A_real_loss.item())
        
        if(self.iteration % 10 == 0):
            #print("Iteration: %d G loss: %f  G Adv loss: %f D loss: %f" % (self.iteration, errG.item(), adv_loss, errD.item()))
            #print("G High level feature loss: %f" %(percep_loss.item()))
            self.visdom_reporter.plot_finegrain_loss(self.iteration, self.losses_dict)
        
        self.iteration += 1
    
    def visdom_report(self, dirty_tensor, clean_tensor):
        with torch.no_grad():
            clean_like = self.G_A(dirty_tensor)
            dirty_like = self.G_B(clean_like)
            
        self.visdom_reporter.plot_image(dirty_tensor, dirty_like, clean_tensor, clean_like)
    
    def vemon_verify(self, dirty_tensor, file_number):
        LOCATION = os.getcwd() + "/figures/"
        with torch.no_grad():
            clean_like = self.G_A(dirty_tensor).detach()
        
        #resize tensors for better viewing
        resized_normal = nn.functional.interpolate(dirty_tensor, scale_factor = 2.0, mode = "bilinear", recompute_scale_factor = True)
        resized_fake = nn.functional.interpolate(clean_like, scale_factor = 2.0, mode = "bilinear", recompute_scale_factor = True)
        
        print("New shapes: %s %s" % (np.shape(resized_normal), np.shape(resized_fake)))
        
        fig, ax = plt.subplots(2, 1)
        fig.set_size_inches(34, 34)
        fig.tight_layout()
        
        ims = np.transpose(vutils.make_grid(resized_normal, nrow = 16, padding=2, normalize=True).cpu(),(1,2,0))
        ax[0].set_axis_off()
        ax[0].imshow(ims)
        
        ims = np.transpose(vutils.make_grid(resized_fake, nrow = 16, padding=2, normalize=True).cpu(),(1,2,0))
        ax[1].set_axis_off()
        ax[1].imshow(ims)
        
        plt.subplots_adjust(left = 0.06, wspace=0.0, hspace=0.15) 
        plt.savefig(LOCATION + "result_" + str(file_number) + ".png")
        plt.show()
    
    def load_saved_state(self, iteration, checkpoint, generator_key, disriminator_key, optimizer_key):
        self.iteration = iteration
        self.G_A.load_state_dict(checkpoint[generator_key + "A"])
        self.G_B.load_state_dict(checkpoint[generator_key + "B"])
        self.D_A.load_state_dict(checkpoint[disriminator_key + "A"])
        self.optimizerG.load_state_dict(checkpoint[generator_key + optimizer_key])
        self.optimizerD.load_state_dict(checkpoint[disriminator_key + optimizer_key])
    
    def save_states(self, epoch, path, generator_key, disriminator_key, optimizer_key):
        self.G_losses = []
        self.D_losses = []
        self.initialize_dict()
        
        save_dict = {'epoch': epoch, 'iteration': self.iteration}
        netGA_state_dict = self.G_A.state_dict()
        netGB_state_dict = self.G_B.state_dict()
        netDA_state_dict = self.D_A.state_dict()
        
        optimizerG_state_dict = self.optimizerG.state_dict()
        optimizerD_state_dict = self.optimizerD.state_dict()
        
        save_dict[generator_key + "A"] = netGA_state_dict
        save_dict[generator_key + "B"] = netGB_state_dict
        save_dict[disriminator_key + "A"] = netDA_state_dict
        
        save_dict[generator_key + optimizer_key] = optimizerG_state_dict
        save_dict[disriminator_key + optimizer_key] = optimizerD_state_dict
    
        torch.save(save_dict, path)
        print("Saved model state: %s Epoch: %d" % (len(save_dict), (epoch + 1)))
    
    def get_gen_state_dicts(self):
        return self.netG.state_dict(), self.optimizerG.state_dict()
    
    def get_disc_state_dicts(self):
        return self.netD.state_dict(), self.optimizerD.state_dict()