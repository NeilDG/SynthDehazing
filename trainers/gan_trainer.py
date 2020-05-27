# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 19:41:58 2019

@author: delgallegon
"""

import os
from model import topdown_gan
from loaders import dataset_loader
import constants
import torch
from torch import optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch.nn as nn
import torchvision.utils as vutils

print = logging.info

class GANTrainer:
    def __init__(self, gan_version, gan_iteration, gpu_device, writer, lr = 0.0002, weight_decay = 0.0, betas = (0.5, 0.999)):
        self.gpu_device = gpu_device
        self.lr = lr
        self.gan_version = gan_version
        self.gan_iteration = gan_iteration
        self.writer = writer
        self.visualized = False
    
        self.netG = topdown_gan.Generator().to(self.gpu_device)
        print(self.netG)
        
        self.netD = topdown_gan.Discriminator().to(self.gpu_device)
        print(self.netD)
        
        self.optimizerD = optim.Adam(self.netD.parameters(), lr, betas)
        self.optimizerG = optim.Adam(self.netG.parameters(), lr, betas)
        
        self.G_losses = []
        self.D_losses = []
    
    def bce_loss(self, pred, target):
        loss = nn.BCELoss() #binary cross-entropy loss
        return loss(pred, target)
    
    #computes the L1 reconstruction loss of GAN
    def compute_gan_loss(self, pred, target):
        loss = nn.L1Loss()
        return loss(pred, target)
        
    # Input = image
    # Performs a discriminator forward-backward pass, then a generator forward-backward pass
    # a = normal image
    # b = topdown image
    def train(self, normal_tensor, homog_tensor, topdown_tensor, iteration):
        real_label = 1
        fake_label = 0
        
        self.netG.train()
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        self.netD.zero_grad()
        
        b_size = normal_tensor.size(0)
        label = torch.full((b_size,), real_label, device = self.gpu_device)
        output_ab = self.netD(normal_tensor, topdown_tensor)
        
        # Calculate loss on all-real batch
        #print("[REAL-D] Label shape: ", np.shape(label))
        #print("[REAL-D] Output shape: ", np.shape(output))
        
        errD_real_ab = self.bce_loss(output_ab, label)
        errD_real_ab.backward()
    
        ## Generate fake topdown image
        fake_b = self.netG(normal_tensor, homog_tensor)
        label.fill_(fake_label)
        
        #print("Generated img shape: ", np.shape(fake))
        
        output_ab = self.netD(normal_tensor, fake_b.detach())
        
        #print("[FAKE-D] Label shape: ", np.shape(label))
        #print("[FAKE-D] Output shape: ", np.shape(output))
        
        # Calculate D's loss on the all-fake batch
        errD_fake_ab = self.bce_loss(output_ab, label)
        errD_fake_ab.backward()
        
        #perform B to A
        a_size = fake_b.size(0)
        label = torch.full((a_size,), real_label, device = self.gpu_device)
        output_ba = self.netD(topdown_tensor, normal_tensor)
        errD_real_ba = self.bce_loss(output_ba, label)
        errD_real_ba.backward()
        
        fake_a = self.netG(fake_b, homog_tensor)
        label.fill_(fake_label)
        output_ba = self.netD(topdown_tensor, fake_a.detach())
        errD_fake_ba = self.bce_loss(output_ba, label)
        errD_fake_ba.backward()
        
        # Add the gradients
        errD = errD_real_ab + errD_fake_ab + errD_real_ba + errD_fake_ba
        # Update D
        self.optimizerD.step()
    
        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        self.netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output_b = self.netD(normal_tensor, fake_b).view(-1)
        output_a = self.netD(topdown_tensor, fake_a).view(-1)
        
        bce_ab = self.bce_loss(output_b, label)
        bce_ba = self.bce_loss(output_a, label)
        l1_ab = self.compute_gan_loss(fake_b, topdown_tensor)
        l1_ba = self.compute_gan_loss(fake_a, normal_tensor)
        
        lambda_ab = 10.0
        lambda_ba = 10.0
        # Calculate G's loss based on this output
        errG = bce_ab + bce_ba + (lambda_ab * l1_ab) + (lambda_ba * l1_ba)
        # Calculate gradients for G
        errG.backward()
        # Update G
        self.optimizerG.step()
        
        # Save Losses for plotting later
        self.G_losses.append(errG.item())
        self.D_losses.append(errD.item())
        
        #print("Iteration: ", iteration, " G loss: ", errG.item(), " D loss: ", errD.item())

    def verify(self, normal_tensor, homog_tensor, topdown_tensor):        
        # Check how the generator is doing by saving G's output on fixed_noise
        with torch.no_grad():
            fake_ab = self.netG(normal_tensor, homog_tensor).detach()
            fake_ba = self.netG(topdown_tensor, homog_tensor).detach()
        
        fig, ax = plt.subplots(4, 1)
        fig.set_size_inches(40, 30)
        
        ims = np.transpose(vutils.make_grid(normal_tensor, nrow = 16, padding=2, normalize=True).cpu(),(1,2,0))
        ax[0].set_axis_off()
        ax[0].imshow(ims)
        
        ims = np.transpose(vutils.make_grid(fake_ba, nrow = 16, padding=2, normalize=True).cpu(),(1,2,0))
        ax[1].set_axis_off()
        ax[1].imshow(ims)
        
        ims = np.transpose(vutils.make_grid(fake_ab, nrow = 16, padding=2, normalize=True).cpu(),(1,2,0))
        ax[2].set_axis_off()
        ax[2].imshow(ims)
        
        ims = np.transpose(vutils.make_grid(topdown_tensor, nrow = 16, padding=2, normalize=True).cpu(),(1,2,0))
        ax[3].set_axis_off()
        ax[3].imshow(ims)
        
        plt.subplots_adjust(left = 0.06, wspace=0.0, hspace=0.15) 
        plt.show()
        
        #verify reconstruction loss with MSE. for reporting purposes
        mse_loss = nn.MSELoss()
        self.current_mse_loss = mse_loss(fake_ab, topdown_tensor)
    
    def verify_and_save(self, normal_tensor, homog_tensor, topdown_tensor, file_number):
        LOCATION = os.getcwd() +"/figures/"
        with torch.no_grad():
            fake = self.netG(normal_tensor, homog_tensor).detach().cpu()
        
        fig, ax = plt.subplots(3, 1)
        fig.set_size_inches(15, 25)
        fig.tight_layout()
        
        ims = np.transpose(vutils.make_grid(normal_tensor, nrow = 16, padding=2, normalize=True).cpu(),(1,2,0))
        ax[0].set_axis_off()
        ax[0].imshow(ims)
        
        ims = np.transpose(vutils.make_grid(fake, nrow = 16, padding=2, normalize=True).cpu(),(1,2,0))
        ax[1].set_axis_off()
        ax[1].imshow(ims)
        
        ims = np.transpose(vutils.make_grid(topdown_tensor, nrow = 16, padding=2, normalize=True).cpu(),(1,2,0))
        ax[2].set_axis_off()
        ax[2].imshow(ims)
        
        plt.subplots_adjust(left = 0.06, wspace=0.0, hspace=0.15) 
        plt.savefig(LOCATION + "result_" + str(file_number) + ".png")
        plt.show()
    
    def vemon_verify(self, normal_tensor, homog_tensor, file_number):
        LOCATION = os.getcwd() + "/figures/"
        with torch.no_grad():
            fake = self.netG(normal_tensor, homog_tensor).detach().cpu()
        
        fig, ax = plt.subplots(2, 1)
        fig.set_size_inches(15, 7)
        fig.tight_layout()
        
        ims = np.transpose(vutils.make_grid(normal_tensor, nrow = 16, padding=2, normalize=True).cpu(),(1,2,0))
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
        self.log_weights("gen", self.netG, self.writer, epoch)
        self.log_weights("disc", self.netD, self.writer, epoch)
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
        
        print("Epoch: ", epoch, " G loss: ", ave_G_loss, " D loss: ", ave_D_loss)
    
    def load_saved_state(self, checkpoint, generator_key, disriminator_key, optimizer_key):
        self.netG.load_state_dict(checkpoint[generator_key])
        self.netD.load_state_dict(checkpoint[disriminator_key])
        self.optimizerG.load_state_dict(checkpoint[generator_key + optimizer_key])
        self.optimizerD.load_state_dict(checkpoint[disriminator_key + optimizer_key])
    
    def save_states(self, epoch, path, generator_key, disriminator_key, optimizer_key):
        save_dict = {'epoch': epoch}
        netG_state_dict = self.netG.state_dict()
        netD_state_dict = self.netD.state_dict()
        optimizerG_state_dict = self.optimizerG.state_dict()
        optimizerD_state_dict = self.optimizerD.state_dict()
        
        save_dict[generator_key] = netG_state_dict
        save_dict[disriminator_key] = netD_state_dict
        save_dict[generator_key + optimizer_key] = optimizerG_state_dict
        save_dict[disriminator_key + optimizer_key] = optimizerD_state_dict
    
        torch.save(save_dict, path)
        print("Saved model state:", len(save_dict)) 
    
    def get_gen_state_dicts(self):
        return self.netG.state_dict(), self.optimizerG.state_dict()
    
    def get_disc_state_dicts(self):
        return self.netD.state_dict(), self.optimizerD.state_dict()