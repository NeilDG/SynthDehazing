# -*- coding: utf-8 -*-
"""
DC Gan tutorial from pytorch
Created on Fri Apr 17 10:49:44 2020

@author: delgallegon
"""

from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from torch.utils.tensorboard import SummaryWriter
from IPython.display import HTML

from model import sample_gan
from loaders import dataset_loader
import constants

def main():
    # Set random seed for reproducibility
    manualSeed = 999
    #manualSeed = random.randint(1, 10000) # use if you want new results
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    
    # Batch size during training
    batch_size = 2048
    
    # Number of channels in the training images. For color images this is 3
    nc = 3
    
    # Size of z latent vector (i.e. size of generator input)
    nz = 100
    
    # Size of feature maps in generator
    ngf = constants.IMAGE_SIZE
    
    # Size of feature maps in discriminator
    ndf = constants.IMAGE_SIZE
    
    # Number of training epochs
    num_epochs = 2000
    
    # Learning rate for optimizers
    lr = 0.0002
    
    # Beta1 hyperparam for Adam optimizers
    beta1 = 0.5
    
    # Number of GPUs available. Use 0 for CPU mode.
    ngpu = 1
    
    # Create the dataloader
    dataloader = dataset_loader.load_dataset(batch_size, 500)
    
    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    
    # Plot some training images
    file_name_batch, real_batch = next(iter(dataloader))
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch.to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
    plt.show()
    
    # Create the generator
    netG = sample_gan.Generator(nc, nz, ngf).to(device)
    print(netG)
    
    netD = sample_gan.Discriminator(nc, ndf).to(device)
    print(netD)
    
    loss = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)
    
    # Establish convention for real and fake labels during training
    real_label = 1
    fake_label = 0
    
    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
    
    #load checkpt
    GAN_VERSION = "gan_v1.02"
    GAN_ITERATION = "1"
    OPTIMIZER_KEY = "optimizer"
    CHECKPATH = 'checkpoint/' + GAN_VERSION +'.pt'
    GENERATOR_KEY = "generator"
    DISCRIMINATOR_KEY = "discriminator"
    start_epoch = 1
    if(False): 
        checkpoint = torch.load(CHECKPATH)
        start_epoch = checkpoint['epoch'] + 1  
        netG.load_state_dict(checkpoint[GENERATOR_KEY])
        optimizerG.load_state_dict(checkpoint[GENERATOR_KEY + OPTIMIZER_KEY])
        
        netD.load_state_dict(checkpoint[DISCRIMINATOR_KEY])
        optimizerD.load_state_dict(checkpoint[DISCRIMINATOR_KEY + OPTIMIZER_KEY])
 
        print("Loaded checkpt ",CHECKPATH, "Current epoch: ", start_epoch)
        print("===================================================")
     
    #initialize tensorboard writer
    writer = SummaryWriter('train_plot')
    
    # Training Loop

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    
    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(start_epoch, num_epochs):
        # For each batch in the dataloader
        for i, (file_name, data) in enumerate(dataloader, 0):
    
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_gpu = data.to(device)
            b_size = real_gpu.size(0)
            label = torch.full((b_size,), real_label, device=device)
            # Forward pass real batch through D
            output = netD(real_gpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = loss(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()
    
            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = loss(output, label)
            # Calculate the gradients for this batch
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()
    
            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = loss(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()
    
            # Output training stats
            if i % 20 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
    
            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())
    
        # Check how the generator is doing by saving G's output on fixed_noise
        with torch.no_grad():
            fake = netG(fixed_noise).detach().cpu()
        img_list.append(vutils.make_grid(fake, nrow = 8, padding=2, normalize=True))  
 
        fig = plt.figure(figsize=(16,16))
        plt.axis("off")
        ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
        plt.show()
        
        
        #save every X epoch
        save_dict = {'epoch': epoch}
                
        netG_state_dict = netG.state_dict()
        netD_state_dict = netD.state_dict()
        optimizerG_state_dict = optimizerG.state_dict()
        optimizerD_state_dict = optimizerD.state_dict()
        
        save_dict[GENERATOR_KEY] = netG_state_dict
        save_dict[DISCRIMINATOR_KEY] = netD_state_dict
        save_dict[GENERATOR_KEY + OPTIMIZER_KEY] = optimizerG_state_dict
        save_dict[DISCRIMINATOR_KEY + OPTIMIZER_KEY] = optimizerD_state_dict
    
        torch.save(save_dict, CHECKPATH)
        print("Saved model state:", len(save_dict))   
        
        log_weights("gen", netG, writer, epoch)
        log_weights("disc", netD, writer, epoch)
        
        # plt.title("Generator and Discriminator Loss During Training")
        # plt.plot(G_losses,label="G")
        # plt.plot(D_losses,label="D")
        # plt.xlabel("iterations")
        # plt.ylabel("Loss")
        # plt.legend()
        # plt.show()
        
        #plot to tensorboard per epoch
        ave_G_loss = sum(G_losses) / (len(G_losses) * 1.0)
        ave_D_loss = sum(D_losses) / (len(D_losses) * 1.0)

        writer.add_scalars(GAN_VERSION +'/loss' + "/" + GAN_ITERATION, {'g_train_loss' :ave_G_loss, 'd_train_loss' : ave_D_loss},
                           global_step = epoch)
        writer.close()

def log_weights(model_name, model, writer, current_epoch):
        #log update in weights
        for module_name,module in model.named_modules():
            for name, param in module.named_parameters():
                if(module_name != ""):
                    #print("Layer added to tensorboard: ", module_name + '/weights/' +name)
                    writer.add_histogram(model_name + "/" + module_name + '/' +name, param.data, global_step = current_epoch)
                        
#FIX for broken pipe num_workers issue.
if __name__=="__main__": 
    main()
    