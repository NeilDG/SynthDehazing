# -*- coding: utf-8 -*-
"""
Experiment on denoising training
Created on Wed Nov  6 19:41:58 2019

@author: delgallegon
"""

import os
from model import style_transfer_gan as st
from model import denoise_discriminator
from loaders import dataset_loader
import constants
import torch
from torch import optim
import itertools
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch.nn as nn
import torchvision.utils as vutils
from utils import logger

print = logger.log

class GANTrainer:
    def __init__(self, gan_version, gan_iteration, gpu_device, writer, lr = 0.0002, weight_decay = 0.0, betas = (0.5, 0.999)):
        self.gpu_device = gpu_device
        self.lr = lr
        self.gan_version = gan_version
        self.gan_iteration = gan_iteration
        self.writer = writer
        self.visualized = False
    
        self.G_A = st.Generator().to(gpu_device) #use multistyle net as architecture
        self.G_B = st.Generator().to(gpu_device)
        
        self.D_A = denoise_discriminator.Discriminator().to(gpu_device)
        
        self.optimizerG = torch.optim.Adam(itertools.chain(self.G_A.parameters(), self.G_B.parameters()), lr=lr, betas=betas, weight_decay=weight_decay)
        self.optimizerD = torch.optim.Adam(self.D_A.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        
        self.G_losses = []
        self.D_losses = []
        
        self.identity_weight = 1.0; self.cycle_weight = 10.0; self.adv_weight = 1.0; self.tv_weight = 10.0
    
    def update_penalties(self, identity, cycle, adv, tv):
        self.identity_weight = identity
        self.cycle_weight = cycle
        self.adv_weight = adv
        self.tv_weight = tv
        print("Weights updated to the following: %f %f %f %f" % (self.identity_weight, self.cycle_weight, self.adv_weight, self.tv_weight))
        
    def adversarial_loss(self, pred, target):
        loss = nn.MSELoss()
        return loss(pred, target)
    
    def identity_loss(self, pred, target):
        loss = nn.L1Loss()
        return loss(pred, target)
    
    def cycle_loss(self, pred, target):
        loss = nn.MSELoss()
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
    def train(self, dirty_tensor, clean_tensor, iteration):
        self.G_A.train()
        self.G_B.train()
        self.optimizerG.zero_grad()
        
        identity_like = self.G_A(dirty_tensor)
        clean_like = self.G_A(dirty_tensor)
        
        A_identity_loss = self.identity_loss(identity_like, dirty_tensor) * self.identity_weight
        A_cycle_loss = self.cycle_loss(self.G_B(clean_like), dirty_tensor) * self.cycle_weight
        A_tv_loss = self.tv_impose(clean_like) * self.tv_weight
        
        prediction = self.D_A(clean_like, clean_tensor)
        real_tensor = torch.ones_like(prediction)
        fake_tensor = torch.zeros_like(prediction)
        
        adv_loss = self.adversarial_loss(prediction, real_tensor) * self.adv_weight
        
        errG = A_identity_loss +  A_cycle_loss + A_tv_loss + adv_loss
        errG.backward()
        self.optimizerG.step()
        
        self.D_A.train()
        self.optimizerD.zero_grad()
        
        
        D_A_real_loss = self.adversarial_loss(self.D_A(clean_tensor, clean_tensor), real_tensor) * self.adv_weight
        D_A_fake_loss = self.adversarial_loss(self.D_A(clean_like.detach(), clean_tensor), fake_tensor) * self.adv_weight
        errD = D_A_real_loss + D_A_fake_loss
        errD.backward()
        
        self.optimizerD.step()
        
        # Save Losses for plotting later
        self.G_losses.append(errG.item())
        self.D_losses.append(errD.item())
        
        #print("Output size: %s", fake_A.size())
        if(iteration % 500 == 0):
            print("Iteration: %d G loss: %f D loss: %f" % (iteration, errG.item(), errD.item()))

    def verify(self, dirty_tensor, clean_tensor):        
        # Check how the generator is doing by saving G's output on fixed_noise
        with torch.no_grad():
            clean_like = self.G_A(dirty_tensor).detach()
            dirty_like = self.G_B(clean_like).detach()
        
        fig, ax = plt.subplots(4, 1)
        fig.set_size_inches(40, 15)
        
        ims = np.transpose(vutils.make_grid(dirty_tensor, nrow = 16, padding=2, normalize=True).cpu(),(1,2,0))
        ax[0].set_axis_off()
        ax[0].imshow(ims)
        
        ims = np.transpose(vutils.make_grid(dirty_like, nrow = 16, padding=2, normalize=True).cpu(),(1,2,0))
        ax[1].set_axis_off()
        ax[1].imshow(ims)
        
        ims = np.transpose(vutils.make_grid(clean_like, nrow = 16, padding=2, normalize=True).cpu(),(1,2,0))
        ax[2].set_axis_off()
        ax[2].imshow(ims)
        
        ims = np.transpose(vutils.make_grid(clean_tensor, nrow = 16, padding=2, normalize=True).cpu(),(1,2,0))
        ax[3].set_axis_off()
        ax[3].imshow(ims)
        
        plt.subplots_adjust(left = 0.06, wspace=0.0, hspace=0.15) 
        plt.show()
        
        #verify reconstruction loss with MSE. for reporting purposes
        mse_loss = nn.MSELoss()
        self.current_mse_loss = mse_loss(clean_like, clean_tensor)
    
    def verify_and_save(self, vemon_tensor, gta_tensor, file_number):
        LOCATION = os.getcwd() + "/figures/"
        with torch.no_grad():
            fake = self.G_A(vemon_tensor).detach().cpu()
        
        fig, ax = plt.subplots(3, 1)
        fig.set_size_inches(15, 15)
        fig.tight_layout()
        
        ims = np.transpose(vutils.make_grid(vemon_tensor, nrow = 16, padding=2, normalize=True).cpu(),(1,2,0))
        ax[0].set_axis_off()
        ax[0].imshow(ims)
        
        ims = np.transpose(vutils.make_grid(fake, nrow = 16, padding=2, normalize=True).cpu(),(1,2,0))
        ax[1].set_axis_off()
        ax[1].imshow(ims)
        
        ims = np.transpose(vutils.make_grid(gta_tensor, nrow = 16, padding=2, normalize=True).cpu(),(1,2,0))
        ax[2].set_axis_off()
        ax[2].imshow(ims)
        
        plt.subplots_adjust(left = 0.06, wspace=0.0, hspace=0.15) 
        plt.savefig(LOCATION + "result_" + str(file_number) + ".png")
        plt.show()
    
    def vemon_verify(self, dirty_tensor, clean_tensor, file_number):
        LOCATION = os.getcwd() + "/figures/"
        with torch.no_grad():
            clean_like = self.G_A(dirty_tensor).detach()
            dirty_like = self.G_B(clean_like).detach()
        
        #resize tensors for better viewing
        resized_clean = nn.functional.interpolate(clean_tensor, scale_factor = 2.0, mode = "bilinear", recompute_scale_factor = True)
        resized_dirty = nn.functional.interpolate(dirty_tensor, scale_factor = 2.0, mode = "bilinear", recompute_scale_factor = True)
        
        resized_clean_like = nn.functional.interpolate(clean_like, scale_factor = 2.0, mode = "bilinear", recompute_scale_factor = True)
        resized_dirty_like = nn.functional.interpolate(dirty_like, scale_factor = 2.0, mode = "bilinear", recompute_scale_factor = True)
        
        print("New shapes: %s %s" % (np.shape(resized_clean_like), np.shape(resized_dirty_like)))
        
        fig, ax = plt.subplots(4, 1)
        fig.set_size_inches(40, 80)
        
        ims = np.transpose(vutils.make_grid(resized_dirty, nrow = 16, padding=2, normalize=True).cpu(),(1,2,0))
        ax[0].set_axis_off()
        ax[0].imshow(ims)
        
        ims = np.transpose(vutils.make_grid(resized_dirty_like, nrow = 16, padding=2, normalize=True).cpu(),(1,2,0))
        ax[1].set_axis_off()
        ax[1].imshow(ims)
        
        ims = np.transpose(vutils.make_grid(resized_clean_like, nrow = 16, padding=2, normalize=True).cpu(),(1,2,0))
        ax[2].set_axis_off()
        ax[2].imshow(ims)
        
        ims = np.transpose(vutils.make_grid(resized_clean, nrow = 16, padding=2, normalize=True).cpu(),(1,2,0))
        ax[3].set_axis_off()
        ax[3].imshow(ims)
        
        plt.subplots_adjust(left = 0.06, wspace=0.0, hspace=0.15) 
        plt.savefig(LOCATION + "result_" + str(file_number) + ".png")
        plt.show()
        
    #reports metrics to necessary tools such as tensorboard
    def report(self, epoch):
        self.log_weights("gen_A", self.G_A, self.writer, epoch)
        self.log_weights("gen_B", self.G_B, self.writer, epoch)
        self.log_weights("disc_A", self.D_A, self.writer, epoch)
        
        self.tensorboard_plot(epoch)

    def log_weights(self, model_name, model, writer, epoch):
        #log update in weights
        for module_name,module in model.named_modules():
            for name, param in module.named_parameters():
                if(module_name != ""):
                    #print("Layer added to tensorboard: ", module_name + '/weights/' +name)
                    writer.add_histogram(model_name + "/" + module_name + '/' +name, param.data, global_step = epoch)
    
    def tensorboard_plot(self, epoch):
        ave_G_loss = sum(self.G_losses) / (len(self.G_losses) * 1.0)
        ave_D_loss = sum(self.D_losses) / (len(self.D_losses) * 1.0)
        
        self.writer.add_scalars(self.gan_version +'/loss' + "/" + self.gan_iteration, {'g_train_loss' :ave_G_loss, 'd_train_loss' : ave_D_loss},
                           global_step = epoch + 1)
        self.writer.add_scalars(self.gan_version +'/mse_loss' + "/" + self.gan_iteration, {'mse_loss' :self.current_mse_loss},
                           global_step = epoch + 1)
        self.writer.close()
        
        print("Epoch: %d G loss: %f D loss: %f" % (epoch, ave_G_loss, ave_D_loss))
    
    def load_saved_state(self, checkpoint, generator_key, disriminator_key, optimizer_key):
        self.G_A.load_state_dict(checkpoint[generator_key + "A"])
        self.G_B.load_state_dict(checkpoint[generator_key + "B"])
        self.D_A.load_state_dict(checkpoint[disriminator_key + "A"])
        self.optimizerG.load_state_dict(checkpoint[generator_key + optimizer_key])
        self.optimizerD.load_state_dict(checkpoint[disriminator_key + optimizer_key])
    
    def save_states(self, epoch, path, generator_key, disriminator_key, optimizer_key):
        save_dict = {'epoch': epoch}
        netGA_state_dict = self.G_A.state_dict()
        netGB_state_dict = self.G_B.state_dict()
        netDA_state_dict = self.D_A.state_dict()
        
        optimizerG_state_dict = self.optimizerG.state_dict()
        optimizerD_state_dict = self.optimizerD.state_dict()
        
        save_dict[generator_key + "A"] = netGA_state_dict
        save_dict[generator_key + "B"] = netGA_state_dict
        save_dict[disriminator_key + "A"] = netDA_state_dict
        
        save_dict[generator_key + optimizer_key] = optimizerG_state_dict
        save_dict[disriminator_key + optimizer_key] = optimizerD_state_dict
    
        torch.save(save_dict, path)
        print("Saved model state: %s Epoch: %d" % (len(save_dict), (epoch + 1)))
    
    def get_gen_state_dicts(self):
        return self.netG.state_dict(), self.optimizerG.state_dict()
    
    def get_disc_state_dicts(self):
        return self.netD.state_dict(), self.optimizerD.state_dict()