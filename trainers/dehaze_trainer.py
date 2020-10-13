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
from utils import pytorch_colors
from utils import tensor_utils

class DehazeTrainer:
    
    def __init__(self, gan_version, gan_iteration, gpu_device, gen_blocks, g_lr, d_lr):
        self.gpu_device = gpu_device
        self.g_lr = g_lr
        self.d_lr = d_lr
        self.gan_version = gan_version
        self.gan_iteration = gan_iteration
        self.visdom_reporter = plot_utils.VisdomReporter()
        
        #self.G_A = cg.Generator(input_nc = 3, output_nc = 3, n_residual_blocks=gen_blocks).to(self.gpu_device)
        self.G_A = cg.Generator(input_nc = 1, output_nc = 1, n_residual_blocks=gen_blocks).to(self.gpu_device)
        #self.D_A = cg.Discriminator(input_nc = 3).to(self.gpu_device)
        self.D_A = cg.Discriminator(input_nc = 1).to(self.gpu_device)
        
        self.optimizerG = torch.optim.Adam(itertools.chain(self.G_A.parameters()), lr = self.g_lr)
        self.optimizerD = torch.optim.Adam(itertools.chain(self.D_A.parameters()), lr = self.d_lr)
        self.initialize_dict()
        
    
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
        self.caption_dict[constants.LIKENESS_LOSS_KEY] = "Clarity loss per iteration"
        self.caption_dict[constants.D_A_FAKE_LOSS_KEY] = "D(A) fake loss per iteration"
        self.caption_dict[constants.D_A_REAL_LOSS_KEY] = "D(A) real loss per iteration"
    
    def update_penalties(self, clarity_weight, adv_weight):
        #what penalties to use for losses?
        self.adv_weight = adv_weight
        self.clarity_weight = clarity_weight
        
        #save hyperparameters for bookeeping
        HYPERPARAMS_PATH = "checkpoint/" + constants.DEHAZER_VERSION + "_" + constants.ITERATION + ".config"
        with open(HYPERPARAMS_PATH, "w") as f:
            print("Version: ", constants.DEHAZER_CHECKPATH, file = f)
            print("Learning rate for G: ", str(self.g_lr), file = f)
            print("Learning rate for D: ", str(self.d_lr), file = f)
            print("====================================", file = f)
            print("Adv weight: ", str(self.adv_weight), file = f)
            print("Clarity weight: ", str(self.clarity_weight), file = f)
            print("Brightness enhance: ", str(constants.brightness_enhance), file = f)
            print("Contrast enhance: ", str(constants.contrast_enhance), file=f)
    
    def adversarial_loss(self, pred, target):
        loss = nn.L1Loss()
        return loss(pred, target)
    
    def clarity_loss(self, pred, target):
        #loss = ssim_loss.SSIM()
        #return 1 - loss(pred, target)
        loss = nn.MSELoss()
        return loss(pred, target)
    
    def train(self, dirty_tensor, clean_tensor):
        clean_like = self.G_A(dirty_tensor)
        
        self.D_A.train()
        self.optimizerD.zero_grad()
        
        prediction = self.D_A(clean_tensor)
        noise_value = random.uniform(0.8, 1.0)
        real_tensor = torch.ones_like(prediction) * noise_value #add noise value to avoid perfect predictions for real
        fake_tensor = torch.zeros_like(prediction)
        
        D_A_real_loss = self.adversarial_loss(self.D_A(clean_tensor), real_tensor) * self.adv_weight
        D_A_fake_loss = self.adversarial_loss(self.D_A(clean_like.detach()), fake_tensor) * self.adv_weight

        errD = D_A_real_loss + D_A_fake_loss
        if(errD.item() > 0.1):
            errD.backward()
            self.optimizerD.step()
        
        self.G_A.train()
        self.optimizerG.zero_grad()
        
        clarity_loss = self.clarity_loss(self.G_A(dirty_tensor), clean_tensor) * self.clarity_weight
        prediction = self.D_A(self.G_A(dirty_tensor))
        real_tensor = torch.ones_like(prediction)
        adv_loss = self.adversarial_loss(prediction, real_tensor) * self.adv_weight
        
        errG = clarity_loss + adv_loss
        errG.backward()
        self.optimizerG.step()
            
        #what to put to losses dict for visdom reporting?
        self.losses_dict[constants.G_LOSS_KEY].append(errG.item())
        self.losses_dict[constants.D_OVERALL_LOSS_KEY].append(errD.item())
        self.losses_dict[constants.LIKENESS_LOSS_KEY].append(clarity_loss.item())
        self.losses_dict[constants.D_A_FAKE_LOSS_KEY].append(D_A_fake_loss.item())
        self.losses_dict[constants.D_A_REAL_LOSS_KEY].append(D_A_real_loss.item())
    
    def visdom_report(self, iteration,synth_dirty_tensor, synth_clean_tensor, test_tensor_1):
        with torch.no_grad():
            synth_clean_like = self.G_A(synth_dirty_tensor)
        
            #report to visdom
            self.visdom_reporter.plot_finegrain_loss("dehazing_loss", iteration, self.losses_dict, self.caption_dict)
            self.visdom_reporter.plot_image(synth_dirty_tensor, "Training Hazy images")
            self.visdom_reporter.plot_image(synth_clean_tensor, "Training Clean images")
            self.visdom_reporter.plot_image(synth_clean_like, "Training Clean-like images")
            self.visdom_reporter.plot_image(test_tensor_1, "Test Hazy images - VEMON")
            self.visdom_reporter.plot_image(self.G_A(test_tensor_1), "Test Clean-like images - VEMON")
            # self.visdom_reporter.plot_image(test_tensor_2, "Test Hazy images - I-Haze")
            # self.visdom_reporter.plot_image(self.G_A(test_tensor_2),"Test Clean-like images - I-Haze")
            # self.visdom_reporter.plot_image(test_tensor_3, "Test Hazy images - Public")
            # self.visdom_reporter.plot_image(self.G_A(test_tensor_3), "Test Clean-like images - Public")
    
    def infer_single(self, dark_dirty_tensor):
        with torch.no_grad():
            dark_clean_tensor = self.G_A(dark_dirty_tensor)
        
        return dark_clean_tensor  
    
    def dehaze_infer(self, denoise_model, dirty_tensor, file_number):
        LOCATION = os.getcwd() + "/figures/"
        with torch.no_grad():
            denoise_like = denoise_model(dirty_tensor) #first pass
            clean_like = self.G_A(denoise_like).detach() #second pass to dehazing network
        
        #resize tensors for better viewing
        resized_normal = nn.functional.interpolate(dirty_tensor, scale_factor = 4.0, mode = "bilinear", recompute_scale_factor = True)
        resized_denoise = nn.functional.interpolate(denoise_like, scale_factor = 4.0, mode = "bilinear", recompute_scale_factor = True)
        resized_fake = nn.functional.interpolate(clean_like, scale_factor = 4.0, mode = "bilinear", recompute_scale_factor = True)
        
        print("New shapes: %s %s" % (np.shape(resized_normal), np.shape(resized_fake)))
        
        fig, ax = plt.subplots(3, 1)
        fig.set_size_inches(constants.FIG_SIZE)
        fig.tight_layout()
        
        ims = np.transpose(vutils.make_grid(resized_normal, nrow = 8, padding=2, normalize=True).cpu(),(1,2,0))
        ax[0].set_axis_off()
        ax[0].imshow(ims)
        
        ims = np.transpose(vutils.make_grid(resized_denoise, nrow = 8, padding=2, normalize=True).cpu(),(1,2,0))
        ax[1].set_axis_off()
        ax[1].imshow(ims)
        
        ims = np.transpose(vutils.make_grid(resized_fake, nrow = 8, padding=2, normalize=True).cpu(),(1,2,0))
        ax[2].set_axis_off()
        ax[2].imshow(ims)
        
        plt.subplots_adjust(left = 0.06, wspace=0.0, hspace=0.15) 
        plt.savefig(LOCATION + "result_" + str(file_number) + ".png")
        plt.show()
            
    def load_saved_state(self, iteration, checkpoint, generator_key, disriminator_key, optimizer_key):
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