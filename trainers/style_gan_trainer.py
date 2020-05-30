# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 19:41:58 2019

@author: delgallegon
"""

import os
from model import style_transfer_gan
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
    
        self.G_A = style_transfer_gan.Generator()
        self.G_B = style_transfer_gan.Generator()
        
        self.D_A = style_transfer_gan.Discriminator()
        self.D_B = style_transfer_gan.Discriminator()
        
        if(torch.cuda.device_count() > 1):
            print("Let's use %d GPUs!", torch.cuda.device_count())
            self.G_A = nn.DataParallel(self.G_A)
            self.G_B = nn.DataParallel(self.G_B)
            self.D_A = nn.DataParallel(self.D_A)
            self.D_B = nn.DataParallel(self.D_B)
        
        self.G_A.to(gpu_device)
        self.G_B.to(gpu_device)
        self.D_A.to(gpu_device)
        self.D_B.to(gpu_device)
        
        self.optimizerG = torch.optim.Adam(itertools.chain(self.G_A.parameters(), self.G_B.parameters()), lr=lr, betas=betas, weight_decay=weight_decay)
        self.optimizerD = torch.optim.Adam(itertools.chain(self.D_A.parameters(), self.D_B.parameters()), lr=lr, betas=betas, weight_decay=weight_decay)
        
        self.G_losses = []
        self.D_losses = []
    
    def adversarial_loss(self, pred, target):
        loss = nn.BCELoss() #binary cross-entropy loss
        #loss = nn.MSELoss()
        return loss(pred, target)
    
    def identity_loss(self, pred, target):
        loss = nn.L1Loss()
        return loss(pred, target)
    
    def cycle_loss(self, pred, target):
        loss = nn.L1Loss()
        return loss(pred, target)
        
    # Input = image
    # Performs a discriminator forward-backward pass, then a generator forward-backward pass
    # a = vemon image
    # b = gta image
    def train(self, vemon_tensor, gta_tensor, iteration):
        real_label = 1
        fake_label = 0
        
        self.lambda_identity = 1.0; self.lambda_cycle = 1.0; self.lambda_adv = 1.0
        
        self.G_A.train()
        self.G_B.train()
        self.optimizerG.zero_grad()
        
        #A = generated gta, B = generated vemon
        fake_A = self.G_A(vemon_tensor)
        fake_B = self.G_B(gta_tensor)
        
        A_identity_loss = self.identity_loss(self.G_A(vemon_tensor), vemon_tensor) * self.lambda_identity
        B_identity_loss = self.identity_loss(self.G_B(gta_tensor), gta_tensor) * self.lambda_identity
        
        A_cycle_loss = self.cycle_loss(self.G_A(fake_B), vemon_tensor) * self.lambda_cycle
        B_cycle_loss = self.cycle_loss(self.G_B(fake_A), gta_tensor) * self.lambda_cycle
        
        output_A = self.D_A(fake_A)
        output_B = self.D_B(fake_B)
        
        real_tensor = torch.ones_like(output_A)
        fake_tensor = torch.zeros_like(output_A)
        
        A_adv_loss = self.adversarial_loss(output_A, real_tensor) * self.lambda_adv
        B_adv_loss = self.adversarial_loss(output_B, real_tensor) * self.lambda_adv
        
        errG = A_identity_loss + B_identity_loss + A_cycle_loss + B_cycle_loss + A_adv_loss + B_adv_loss
        #errG = A_cycle_loss + B_cycle_loss + A_adv_loss + B_adv_loss
        errG.backward()
        self.optimizerG.step()
        
        self.D_A.train()
        self.D_B.train()
        self.optimizerD.zero_grad()
        D_A_real_loss = self.adversarial_loss(self.D_A(vemon_tensor), real_tensor)
        D_A_fake_loss = self.adversarial_loss(self.D_A(fake_A.detach()), fake_tensor)
        D_B_real_loss = self.adversarial_loss(self.D_B(gta_tensor), real_tensor)
        D_B_fake_loss = self.adversarial_loss(self.D_B(fake_B.detach()), fake_tensor)
        
        errD = D_A_real_loss + D_A_fake_loss + D_B_real_loss + D_B_fake_loss
        errD.backward()
        self.optimizerD.step()
        
        # Save Losses for plotting later
        self.G_losses.append(errG.item())
        self.D_losses.append(errD.item())
        
        #print("Output size: %s", fake_A.size())
        if(iteration % 500 == 0):
            print("Iteration: %d G loss: %f D loss: %f" % (iteration, errG.item(), errD.item()))

    def verify(self, vemon_tensor, gta_tensor):        
        # Check how the generator is doing by saving G's output on fixed_noise
        with torch.no_grad():
            fake_ab = self.G_A(vemon_tensor).detach()
            fake_ba = self.G_B(gta_tensor).detach()
        
        fig, ax = plt.subplots(4, 1)
        fig.set_size_inches(40, 15)
        
        ims = np.transpose(vutils.make_grid(vemon_tensor, nrow = 16, padding=2, normalize=True).cpu(),(1,2,0))
        ax[0].set_axis_off()
        ax[0].imshow(ims)
        
        ims = np.transpose(vutils.make_grid(fake_ab, nrow = 16, padding=2, normalize=True).cpu(),(1,2,0))
        ax[1].set_axis_off()
        ax[1].imshow(ims)
        
        ims = np.transpose(vutils.make_grid(fake_ba, nrow = 16, padding=2, normalize=True).cpu(),(1,2,0))
        ax[2].set_axis_off()
        ax[2].imshow(ims)
        
        ims = np.transpose(vutils.make_grid(gta_tensor, nrow = 16, padding=2, normalize=True).cpu(),(1,2,0))
        ax[3].set_axis_off()
        ax[3].imshow(ims)
        
        plt.subplots_adjust(left = 0.06, wspace=0.0, hspace=0.15) 
        plt.show()
        
        #verify reconstruction loss with MSE. for reporting purposes
        mse_loss = nn.MSELoss()
        self.current_mse_loss = mse_loss(fake_ab, gta_tensor)
    
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
    
    def vemon_verify(self, vemon_tensor, file_number):
        LOCATION = os.getcwd() + "/figures/"
        with torch.no_grad():
            fake = self.G_A(vemon_tensor).detach().cpu()
        
        fig, ax = plt.subplots(2, 1)
        fig.set_size_inches(18, 7)
        fig.tight_layout()
        
        ims = np.transpose(vutils.make_grid(vemon_tensor, nrow = 16, padding=2, normalize=True).cpu(),(1,2,0))
        ax[0].set_axis_off()
        ax[0].imshow(ims)
        
        ims = np.transpose(vutils.make_grid(fake, nrow = 16, padding=2, normalize=True).cpu(),(1,2,0))
        ax[1].set_axis_off()
        ax[1].imshow(ims)
        
        plt.subplots_adjust(left = 0.06, wspace=0.0, hspace=0.15) 
        plt.savefig(LOCATION + "result_" + str(file_number) + ".png")
        plt.show()
        
    #reports metrics to necessary tools such as tensorboard
    def report(self, epoch):
        self.log_weights("gen_A", self.G_A, self.writer, epoch)
        self.log_weights("gen_B", self.G_B, self.writer, epoch)
        self.log_weights("disc_A", self.D_A, self.writer, epoch)
        self.log_weights("disc_B", self.D_B, self.writer, epoch)
        
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
        self.D_B.load_state_dict(checkpoint[disriminator_key + "B"])
        self.optimizerG.load_state_dict(checkpoint[generator_key + optimizer_key])
        self.optimizerD.load_state_dict(checkpoint[disriminator_key + optimizer_key])
    
    def save_states(self, epoch, path, generator_key, disriminator_key, optimizer_key):
        save_dict = {'epoch': epoch}
        netGA_state_dict = self.G_A.state_dict()
        netGB_state_dict = self.G_B.state_dict()
        netDA_state_dict = self.D_A.state_dict()
        netDB_state_dict = self.D_B.state_dict()
        
        optimizerG_state_dict = self.optimizerG.state_dict()
        optimizerD_state_dict = self.optimizerD.state_dict()
        
        save_dict[generator_key + "A"] = netGA_state_dict
        save_dict[generator_key + "B"] = netGA_state_dict
        save_dict[disriminator_key + "A"] = netDA_state_dict
        save_dict[disriminator_key + "B"] = netDB_state_dict
        
        save_dict[generator_key + optimizer_key] = optimizerG_state_dict
        save_dict[disriminator_key + optimizer_key] = optimizerD_state_dict
    
        torch.save(save_dict, path)
        print("Saved model state: %s Epoch: %d" % (len(save_dict), (epoch + 1)))
    
    def get_gen_state_dicts(self):
        return self.netG.state_dict(), self.optimizerG.state_dict()
    
    def get_disc_state_dicts(self):
        return self.netD.state_dict(), self.optimizerD.state_dict()