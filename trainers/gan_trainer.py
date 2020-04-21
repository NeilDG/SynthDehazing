# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 19:41:58 2019

@author: delgallegon
"""

from model import sample_gan
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

class GANTrainer:
    def __init__(self, gan_version, gan_iteration, gpu_device, writer, lr = 0.0002, weight_decay = 0.0, betas = (0.5, 0.999)):
        self.gpu_device = gpu_device
        self.lr = lr
        self.gan_version = gan_version
        self.gan_iteration = gan_iteration
        self.writer = writer
        self.visualized = False
        
        # Number of channels in the training images. For color images this is 3
        self.num_channels = 3
        
        # Size of z latent vector (i.e. size of generator input)
        self.input_latent_size = 100
        
        # Size of feature maps in generator
        self.gen_feature_size = constants.IMAGE_SIZE
        
        # Size of feature maps in discriminator
        self.disc_feature_size = constants.IMAGE_SIZE
        
        # Create batch of latent vectors that we will use to visualize
        #  the progression of the generator
        self.fixed_noise = torch.randn(64, self.input_latent_size, 1, 1, device=self.gpu_device)
    
        self.netG = sample_gan.Generator(self.num_channels, self.input_latent_size, self.gen_feature_size).to(self.gpu_device)
        print(self.netG)
        
        self.netD = sample_gan.Discriminator(self.num_channels, self.disc_feature_size).to(self.gpu_device)
        print(self.netD)
        
        self.optimizerD = optim.Adam(self.netD.parameters(), lr, betas)
        self.optimizerG = optim.Adam(self.netG.parameters(), lr, betas)
        
        self.G_losses = []
        self.D_losses = []
    
    def compute_loss(self, pred, target):
        loss = nn.BCELoss() #binary cross-entropy loss
        return loss(pred, target)
        
    # Input = image
    # Performs a discriminator forward-backward pass, then a generator forward-backward pass
    def train(self, input, iteration):
        real_label = 1
        fake_label = 0
        
        self.netG.train()
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        self.netD.zero_grad()
        # Format batch
        b_size = input.size(0)
        label = torch.full((b_size,), real_label, device = self.gpu_device)
        # Forward pass real batch through D
        output = self.netD(input).view(-1)
        # Calculate loss on all-real batch
        #print("[REAL] Label shape: ", np.shape(label))
        #print("[REAL] Output shape: ", np.shape(output))
        errD_real = self.compute_loss(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        #D_x = output.mean().item()
    
        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, self.input_latent_size, 1, 1, device=self.gpu_device)
        # Generate fake image batch with G
        fake = self.netG(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = self.netD(fake.detach()).view(-1)
        #print("[FAKE] Label shape: ", np.shape(label))
        #print("[FAKE] Output shape: ", np.shape(output))
        # Calculate D's loss on the all-fake batch
        errD_fake = self.compute_loss(output, label)
        # Calculate the gradients for this batch
        errD_fake.backward()
        #D_G_z1 = output.mean().item()
        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake
        # Update D
        self.optimizerD.step()
    
        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        self.netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = self.netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = self.compute_loss(output, label)
        # Calculate gradients for G
        errG.backward()
        #D_G_z2 = output.mean().item()
        # Update G
        self.optimizerG.step()
        
        # Save Losses for plotting later
        self.G_losses.append(errG.item())
        self.D_losses.append(errD.item())
        
        #print("Iteration: ", iteration, " G loss: ", errG.item(), " D loss: ", errD.item())

    def verify(self):        
        # Check how the generator is doing by saving G's output on fixed_noise
        with torch.no_grad():
            fake = self.netG(self.fixed_noise).detach().cpu()
 
        plt.figure(figsize=(constants.FIG_SIZE,constants.FIG_SIZE))
        plt.axis("off")
        ims = np.transpose(vutils.make_grid(fake, nrow = 8, padding=2, normalize=True).cpu(),(1,2,0))
        plt.imshow(ims)
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