# -*- coding: utf-8 -*-
"""
Class for producing figures
Created on Sat May  2 09:09:21 2020

@author: delgallegon
"""

import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from torch.utils.tensorboard import SummaryWriter
from loaders import dataset_loader
from trainers import cyclic_gan_trainer
from trainers import style_gan_trainer
import constants

def view_train_results(batch_size, gan_version, gan_iteration):
    
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    writer = SummaryWriter('train_plot')
    
    gt = cyclic_gan_trainer.GANTrainer(gan_version, gan_iteration, device, writer)
    
    checkpoint = torch.load(constants.CHECKPATH)  
    gt.load_saved_state(checkpoint, constants.GENERATOR_KEY, constants.DISCRIMINATOR_KEY, constants.OPTIMIZER_KEY)
 
    print("Loaded results checkpt ",constants.CHECKPATH)
    print("===================================================")
    
    dataloader = dataset_loader.load_synth_dataset(batch_size, -1)
    item_number = 0
    for i, (name, normal_img, homog_img, topdown_img) in enumerate(dataloader, 0):
        normal_tensor = normal_img.to(device)
        homog_tensor = homog_img.to(device)
        topdown_tensor = topdown_img.to(device)
        item_number = item_number + 1
        gt.verify_and_save(normal_tensor, homog_tensor, topdown_tensor, item_number)

def style_transfer(batch_size, style_version, style_iteration):
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    writer = SummaryWriter('train_plot')
    
    gt = style_gan_trainer.GANTrainer(style_version, style_iteration, device, writer)
    checkpoint = torch.load(constants.STYLE_CHECKPATH)
    gt.load_saved_state(checkpoint, constants.GENERATOR_KEY, constants.DISCRIMINATOR_KEY, constants.OPTIMIZER_KEY)
 
    print("Loaded results checkpt ",constants.STYLE_CHECKPATH)
    print("===================================================")
    
    dataloader = dataset_loader.load_style_dataset(batch_size, -1)
    
    # Plot some training images
    name_batch, vemon_batch, homog_batch, gta_batch = next(iter(dataloader))
    plt.figure(figsize=(constants.FIG_SIZE,constants.FIG_SIZE))
    plt.axis("off")
    plt.title("Training - VEMON Images")
    plt.imshow(np.transpose(vutils.make_grid(vemon_batch.to(device)[:constants.batch_size], nrow = 8, padding=2, normalize=True).cpu(),(1,2,0)))
    plt.show()
    
    plt.figure(figsize=(constants.FIG_SIZE,constants.FIG_SIZE))
    plt.axis("off")
    plt.title("Training - GTA Images")
    plt.imshow(np.transpose(vutils.make_grid(gta_batch.to(device)[:constants.batch_size], nrow = 8, padding=2, normalize=True).cpu(),(1,2,0)))
    plt.show()
    
    item_number = 0
    for i, (name, vemon_batch, homog_batch, gta_batch) in enumerate(dataloader, 0):
        vemon_tensor = vemon_batch.to(device)
        item_number = item_number + 1
        gt.vemon_verify(vemon_tensor, item_number)
    
def vemon_infer(batch_size, gan_version, gan_iteration): 
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    writer = SummaryWriter('train_plot')
    
    gt = cyclic_gan_trainer.GANTrainer(gan_version, gan_iteration, device, writer)
    
    checkpoint = torch.load(constants.CHECKPATH)  
    gt.load_saved_state(checkpoint, constants.GENERATOR_KEY, constants.DISCRIMINATOR_KEY, constants.OPTIMIZER_KEY)
 
    print("Loaded results checkpt ",constants.CHECKPATH)
    print("===================================================")
    
    dataloader = dataset_loader.load_vemon_dataset(batch_size, -1)
    
    # Plot some training images
    name_batch, normal_batch, homog_batch, topdown_batch = next(iter(dataloader))
    plt.figure(figsize=(constants.FIG_SIZE,constants.FIG_SIZE))
    plt.axis("off")
    plt.title("Training - Normal Images")
    plt.imshow(np.transpose(vutils.make_grid(normal_batch.to(device)[:batch_size], nrow = 16, padding=2, normalize=True).cpu(),(1,2,0)))
    plt.show()
    
    plt.figure(figsize=(constants.FIG_SIZE,constants.FIG_SIZE))
    plt.axis("off")
    plt.title("Training - Homog Images")
    plt.imshow(np.transpose(vutils.make_grid(homog_batch.to(device)[:batch_size], nrow = 16, padding=2, normalize=True).cpu(),(1,2,0)))
    plt.show()
    
    item_number = 0
    for i, (name, normal_img, homog_img, topdown_img) in enumerate(dataloader, 0):
        normal_tensor = normal_img.to(device)
        homog_tensor = homog_img.to(device)
        item_number = item_number + 1
        gt.vemon_verify(normal_tensor, homog_tensor, item_number)

def main():
    #view_train_results(constants.infer_size, constants.GAN_VERSION, constants.GAN_ITERATION)
    style_transfer(constants.infer_size, constants.STYLE_GAN_VERSION, constants.STYLE_ITERATION)

#FIX for broken pipe num_workers issue.
if __name__=="__main__": 
    main()        
        
        