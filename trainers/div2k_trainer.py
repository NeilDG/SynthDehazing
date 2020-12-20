# -*- coding: utf-8 -*-
# Template trainer. Do not use this for actual training.

import os
from model import style_transfer_gan as transfer_gan
from model import vanilla_cycle_gan as discrim_gan
import constants
import torch
import random
import itertools
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision.utils as vutils
from custom_losses import ssim_loss
from custom_losses import vgg_loss_model as vgg
from utils import logger
from utils import plot_utils
from utils import pytorch_colors
from utils import tensor_utils

class Div2kTrainer:
    
    def __init__(self, gan_version, gan_iteration, gpu_device, g_lr, d_lr):
        self.gpu_device = gpu_device
        self.g_lr = g_lr
        self.d_lr = d_lr
        self.gan_version = gan_version
        self.gan_iteration = gan_iteration
        self.G_A = discrim_gan.Generator(n_residual_blocks=10).to(self.gpu_device)
        self.G_B = discrim_gan.Generator(n_residual_blocks=10).to(self.gpu_device)
        self.D_A = discrim_gan.Discriminator().to(self.gpu_device) #use CycleGAN's discriminator
        self.D_B = discrim_gan.Discriminator().to(self.gpu_device)
        
        self.visdom_reporter = plot_utils.VisdomReporter()
        self.optimizerG = torch.optim.Adam(itertools.chain(self.G_A.parameters(), self.G_B.parameters()), lr = self.g_lr)
        self.optimizerD = torch.optim.Adam(itertools.chain(self.D_A.parameters(), self.D_B.parameters()), lr = self.d_lr)
        self.initialize_dict()
        
    
    def initialize_dict(self):
        #what to store in visdom?
        self.losses_dict = {}
        self.losses_dict[constants.G_LOSS_KEY] = []
        self.losses_dict[constants.D_OVERALL_LOSS_KEY] = []
        self.losses_dict[constants.IDENTITY_LOSS_KEY] = []
        self.losses_dict[constants.LIKENESS_LOSS_KEY] = []
        self.losses_dict[constants.COLOR_SHIFT_LOSS_KEY] = []
        self.losses_dict[constants.G_ADV_LOSS_KEY] = []
        self.losses_dict[constants.D_A_FAKE_LOSS_KEY] = []
        self.losses_dict[constants.D_A_REAL_LOSS_KEY] = []
        self.losses_dict[constants.D_B_FAKE_LOSS_KEY] = []
        self.losses_dict[constants.D_B_REAL_LOSS_KEY] = []
        self.losses_dict[constants.CYCLE_LOSS_KEY] = []
        
        self.caption_dict = {}
        self.caption_dict[constants.G_LOSS_KEY] = "G loss per iteration"
        self.caption_dict[constants.D_OVERALL_LOSS_KEY] = "D loss per iteration"
        self.caption_dict[constants.IDENTITY_LOSS_KEY] = "Identity loss per iteration"
        self.caption_dict[constants.LIKENESS_LOSS_KEY] = "Likeness loss per iteration"
        self.caption_dict[constants.COLOR_SHIFT_LOSS_KEY] = "Color shift loss per iteration"
        self.caption_dict[constants.G_ADV_LOSS_KEY] = "G adv loss per iteration"
        self.caption_dict[constants.D_A_FAKE_LOSS_KEY] = "D(A) fake loss per iteration"
        self.caption_dict[constants.D_A_REAL_LOSS_KEY] = "D(A) real loss per iteration"
        self.caption_dict[constants.D_B_FAKE_LOSS_KEY] = "D(B) fake loss per iteration"
        self.caption_dict[constants.D_B_REAL_LOSS_KEY] = "D(B) real loss per iteration"
        self.caption_dict[constants.CYCLE_LOSS_KEY] = "Cycle loss per iteration"
        
    
    def update_penalties(self, adv_weight, id_weight, likeness_weight, cycle_weight, color_shift_weight):
        #what penalties to use for losses?
        self.adv_weight = adv_weight
        self.id_weight = id_weight
        self.likeness_weight = likeness_weight
        self.cycle_weight = cycle_weight
        self.color_shift_weight = color_shift_weight
        
        #save hyperparameters for bookeeping
        HYPERPARAMS_PATH = "checkpoint/" + constants.COLOR_TRANSFER_VERSION + "_" + constants.ITERATION + ".config"
        with open(HYPERPARAMS_PATH, "w") as f:
            print("Version: ", constants.COLOR_TRANSFER_CHECKPATH, file = f)
            print("Learning rate for G: ", str(self.g_lr), file = f)
            print("Learning rate for D: ", str(self.d_lr), file = f)
            print("====================================", file = f)
            print("Adv weight: ", str(self.adv_weight), file = f)
            print("Identity weight: ", str(self.id_weight), file = f)
            print("Likeness weight: ", str(self.likeness_weight), file = f)
            print("Color shift weight: ", str(self.color_shift_weight), file=f)
            print("Cycle weight: ", str(self.cycle_weight), file = f)
            print("====================================", file = f)
            print("Brightness enhance: ", str(constants.brightness_enhance), file = f)
            print("Contrast enhance: ", str(constants.contrast_enhance), file = f)
            
        
    
    def adversarial_loss(self, pred, target):
        loss = nn.MSELoss()
        return loss(pred, target)
    
    def identity_loss(self, pred, target):
        loss = nn.L1Loss()
        return loss(pred, target)
    
    def cycle_loss(self, pred, target):
        loss = nn.L1Loss()
        return loss(pred, target)
        #loss = ssim_loss.SSIM()
        #return 1 - loss(pred, target)
    
    def likeness_loss(self, pred, target):
        loss = nn.L1Loss()
        return loss(pred, target)
        # loss = vgg.VGGPerceptualLoss().to(self.gpu_device)
        # return loss(pred, target)

    def color_shift_loss(self, pred, target):
        pred_lab = pytorch_colors.rgb_to_lab(pred.detach())
        target_lab = pytorch_colors.rgb_to_lab(target.detach())

        (y,u,v) = torch.chunk(pred_lab.transpose(0,1), 3)
        pred_ab = torch.cat((u,v))

        (y,u,v) = torch.chunk(target_lab.transpose(0,1), 3)
        target_ab = torch.cat((u,v))


        pred_ab = torch.cat((torch.zeros_like(y), pred_ab))
        target_ab = torch.cat((torch.zeros_like(y), target_ab))
        pred_ab = pred_ab.transpose(0,1)
        target_ab = target_ab.transpose(0,1)

        #impose color penalty to tensor for autograd by canceling out original pred/target pair
        pred = pred + pred_ab
        target = target + target_ab

        loss = nn.L1Loss()
        return loss(pred, target)

    def train(self, dirty_tensor, clean_tensor):
        clean_like = self.G_A(dirty_tensor)
        dirty_like = self.G_B(clean_tensor)
        
        self.D_A.train()
        self.D_B.train()
        self.optimizerD.zero_grad()
        
        prediction = self.D_A(clean_tensor)
        real_tensor = torch.ones_like(prediction)
        fake_tensor = torch.zeros_like(prediction)
        
        D_A_real_loss = self.adversarial_loss(self.D_A(clean_tensor), real_tensor) * self.adv_weight
        D_A_fake_loss = self.adversarial_loss(self.D_A(clean_like.detach()), fake_tensor) * self.adv_weight
        
        prediction = self.D_B(dirty_tensor)
        real_tensor = torch.ones_like(prediction)
        fake_tensor = torch.zeros_like(prediction)
        
        D_B_real_loss = self.adversarial_loss(self.D_B(dirty_tensor), real_tensor) * self.adv_weight
        D_B_fake_loss = self.adversarial_loss(self.D_B(dirty_like.detach()), fake_tensor) * self.adv_weight
        
        errD = D_A_real_loss + D_A_fake_loss + D_B_real_loss + D_B_fake_loss
        if(errD.item() > 0.1):
            errD.backward()
            self.optimizerD.step()
        
        self.G_A.train()
        self.G_B.train()
        self.optimizerG.zero_grad()
        
        identity_like = self.G_A(clean_tensor)
        clean_like = self.G_A(dirty_tensor)
        dirty_like = self.G_B(clean_like)
        
        identity_loss = self.identity_loss(identity_like, clean_tensor) * self.id_weight
        #A_likeness_loss = self.likeness_loss(clean_like, clean_tensor) * self.likeness_weight
        A_color_shift_loss = self.color_shift_loss(clean_like, clean_tensor) * self.color_shift_weight
        A_cycle_loss = self.cycle_loss(dirty_like, dirty_tensor) * self.cycle_weight
    
        dirty_like = self.G_B(clean_tensor)
        B_likeness_loss = self.likeness_loss(dirty_like, dirty_tensor) * self.likeness_weight
        B_color_shift_loss = self.color_shift_loss(dirty_like, dirty_like) * self.color_shift_weight
        B_cycle_loss = self.cycle_loss(self.G_A(dirty_like), clean_tensor) * self.cycle_weight
        
        prediction = self.D_A(clean_like)
        real_tensor = torch.ones_like(prediction)
        A_adv_loss = self.adversarial_loss(prediction, real_tensor) * self.adv_weight
        
        prediction = self.D_B(dirty_like)
        real_tensor = torch.ones_like(prediction)
        B_adv_loss = self.adversarial_loss(prediction, real_tensor) * self.adv_weight
        
        errG = identity_loss + B_likeness_loss + A_color_shift_loss + B_color_shift_loss + A_adv_loss + B_adv_loss + A_cycle_loss + B_cycle_loss
        errG.backward()
        self.optimizerG.step()
        
        #what to put to losses dict for visdom reporting?
        self.losses_dict[constants.G_LOSS_KEY].append(errG.item())
        self.losses_dict[constants.D_OVERALL_LOSS_KEY].append(errD.item())
        self.losses_dict[constants.IDENTITY_LOSS_KEY].append(identity_loss.item())
        self.losses_dict[constants.LIKENESS_LOSS_KEY].append(B_likeness_loss.item())
        self.losses_dict[constants.COLOR_SHIFT_LOSS_KEY].append(A_color_shift_loss.item() + B_color_shift_loss.item())
        self.losses_dict[constants.G_ADV_LOSS_KEY].append(A_adv_loss.item() + B_adv_loss.item())
        self.losses_dict[constants.D_A_FAKE_LOSS_KEY].append(D_A_fake_loss.item())
        self.losses_dict[constants.D_A_REAL_LOSS_KEY].append(D_A_real_loss.item())
        self.losses_dict[constants.D_B_FAKE_LOSS_KEY].append(D_B_fake_loss.item())
        self.losses_dict[constants.D_B_REAL_LOSS_KEY].append(D_B_real_loss.item())
        self.losses_dict[constants.CYCLE_LOSS_KEY].append(A_cycle_loss.item() + B_cycle_loss.item())
    
    def visdom_report(self, iteration, dirty_tensor, clean_tensor, test_dirty_tensor, test_clean_tensor):
        with torch.no_grad():
            clean_like = self.G_A(dirty_tensor)
            test_clean_like = self.G_A(test_dirty_tensor)
            test_dirty_like = self.G_B(test_clean_like)
        
        #report to visdom
        self.visdom_reporter.plot_finegrain_loss("color transfer loss", iteration, self.losses_dict, self.caption_dict)
        self.visdom_reporter.plot_image((dirty_tensor), "Training Dirty images")
        self.visdom_reporter.plot_image((clean_tensor), "Training Clean images")
        self.visdom_reporter.plot_image((clean_like), "Training Clean-like images")
        self.visdom_reporter.plot_image((test_dirty_tensor), "Test Dirty images")
        self.visdom_reporter.plot_image((test_dirty_like), "Test Dirty-like images")
        self.visdom_reporter.plot_image((test_clean_tensor), "Test Clean images")
        self.visdom_reporter.plot_image((test_clean_like), "Test Clean-like images")
    
    def produce_image(self, dirty_tensor):
        with torch.no_grad():
            clean_like = self.G_A(dirty_tensor, self.denoise_model(dirty_tensor))
            resized_fake = nn.functional.interpolate(clean_like, scale_factor = 1.0, mode = "bilinear", recompute_scale_factor = True)
        
        return resized_fake
   
    def infer(self, dirty_tensor, file_number):
        LOCATION = os.getcwd() + "/figures/"
        with torch.no_grad():
            clean_like = self.G_A(dirty_tensor).detach()
        
        #resize tensors for better viewing
        resized_normal = nn.functional.interpolate(dirty_tensor, scale_factor = 4.0, mode = "bilinear", recompute_scale_factor = True)
        resized_fake = nn.functional.interpolate(clean_like, scale_factor = 4.0, mode = "bilinear", recompute_scale_factor = True)
        
        print("New shapes: %s %s" % (np.shape(resized_normal), np.shape(resized_fake)))
        
        fig, ax = plt.subplots(2, 1)
        fig.set_size_inches((32, 16))
        fig.tight_layout()
        
        ims = np.transpose(vutils.make_grid(resized_normal, nrow = 8, padding=2, normalize=True).cpu(),(1,2,0))
        ax[0].set_axis_off()
        ax[0].imshow(ims)
        
        ims = np.transpose(vutils.make_grid(resized_fake, nrow = 8, padding=2, normalize=True).cpu(),(1,2,0))
        ax[1].set_axis_off()
        ax[1].imshow(ims)
        
        plt.subplots_adjust(left = 0.06, wspace=0.0, hspace=0.15) 
        plt.savefig(LOCATION + "result_" + str(file_number) + ".png")
        plt.show()
        
    def load_saved_state(self, iteration, checkpoint, generator_key, discriminator_key, optimizer_key):
        self.G_A.load_state_dict(checkpoint[generator_key + "A"])
        self.G_B.load_state_dict(checkpoint[generator_key + "B"])
        self.D_A.load_state_dict(checkpoint[discriminator_key + "A"])
        self.D_B.load_state_dict(checkpoint[discriminator_key + "B"])
        self.optimizerG.load_state_dict(checkpoint[generator_key + optimizer_key])
        self.optimizerD.load_state_dict(checkpoint[discriminator_key + optimizer_key])
    
    def save_states(self, epoch, iteration, path, generator_key, disriminator_key, optimizer_key):
        save_dict = {'epoch': epoch, 'iteration': iteration}
        netGA_state_dict = self.G_A.state_dict()
        netGB_state_dict = self.G_B.state_dict()
        netDA_state_dict = self.D_A.state_dict()
        netDB_state_dict = self.D_B.state_dict()
        
        optimizerG_state_dict = self.optimizerG.state_dict()
        optimizerD_state_dict = self.optimizerD.state_dict()
        
        save_dict[generator_key + "A"] = netGA_state_dict
        save_dict[generator_key + "B"] = netGB_state_dict
        save_dict[disriminator_key + "A"] = netDA_state_dict
        save_dict[disriminator_key + "B"] = netDB_state_dict
        
        save_dict[generator_key + optimizer_key] = optimizerG_state_dict
        save_dict[disriminator_key + optimizer_key] = optimizerD_state_dict
    
        torch.save(save_dict, path)
        print("Saved model state: %s Epoch: %d" % (len(save_dict), (epoch + 1)))