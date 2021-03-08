# -*- coding: utf-8 -*-
# Template trainer. Do not use this for actual training.

import os
from model import vanilla_cycle_gan as cg
from model import style_transfer_gan as sg
from model import dehaze_discriminator as dh
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
import kornia
from custom_losses import rmse_log_loss

class TransmissionTrainer:
    
    def __init__(self, gan_version, gan_iteration, gpu_device, g_lr = 0.0002, d_lr = 0.0002):
        self.gpu_device = gpu_device
        self.g_lr = g_lr
        self.d_lr = d_lr
        self.gan_version = gan_version
        self.gan_iteration = gan_iteration
        self.G_A = cg.Generator(input_nc = 3, output_nc = 1, n_residual_blocks = 8).to(self.gpu_device)
        #self.G_A = sg.Generator(input_nc=3, output_nc=1, n_residual_blocks = 10).to(self.gpu_device)
        #self.D_A = cg.Discriminator(input_nc = 1).to(self.gpu_device)  # use CycleGAN's discriminator
        self.D_A = dh.Discriminator(input_nc = 1).to(self.gpu_device)

        self.visdom_reporter = plot_utils.VisdomReporter()
        self.initialize_dict()

        self.optimizerG = torch.optim.Adam(itertools.chain(self.G_A.parameters()), lr=self.g_lr)
        self.optimizerD = torch.optim.Adam(itertools.chain(self.D_A.parameters()), lr=self.d_lr)
        self.schedulerG = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizerG, patience = 1000, threshold = 0.00005)
        self.schedulerD = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizerD, patience = 1000, threshold=0.00005)
        
    
    def initialize_dict(self):
        # what to store in visdom?
        self.losses_dict = {}
        self.losses_dict[constants.G_LOSS_KEY] = []
        self.losses_dict[constants.D_OVERALL_LOSS_KEY] = []
        self.losses_dict[constants.LIKENESS_LOSS_KEY] = []
        self.losses_dict[constants.EDGE_LOSS_KEY] = []
        self.losses_dict[constants.G_ADV_LOSS_KEY] = []
        self.losses_dict[constants.D_A_FAKE_LOSS_KEY] = []
        self.losses_dict[constants.D_A_REAL_LOSS_KEY] = []
        #self.losses_dict[constants.D_B_FAKE_LOSS_KEY] = []
        #self.losses_dict[constants.D_B_REAL_LOSS_KEY] = []

        self.caption_dict = {}
        self.caption_dict[constants.G_LOSS_KEY] = "G loss per iteration"
        self.caption_dict[constants.D_OVERALL_LOSS_KEY] = "D loss per iteration"
        self.caption_dict[constants.LIKENESS_LOSS_KEY] = "Likeness loss per iteration"
        self.caption_dict[constants.EDGE_LOSS_KEY] = "Edge loss per iteration"
        self.caption_dict[constants.G_ADV_LOSS_KEY] = "G adv loss per iteration"
        self.caption_dict[constants.D_A_FAKE_LOSS_KEY] = "D(A) fake loss per iteration"
        self.caption_dict[constants.D_A_REAL_LOSS_KEY] = "D(A) real loss per iteration"
        #self.caption_dict[constants.D_B_FAKE_LOSS_KEY] = "D(B) fake loss per iteration"
        #self.caption_dict[constants.D_B_REAL_LOSS_KEY] = "D(B) real loss per iteration"
        
    
    def update_penalties(self, adv_weight, likeness_weight, edge_weight, comments):
        #what penalties to use for losses?
        self.adv_weight = adv_weight
        self.likeness_weight = likeness_weight
        self.edge_weight = edge_weight

        # save hyperparameters for bookkeeping
        HYPERPARAMS_PATH = "checkpoint/" + constants.TRANSMISSION_VERSION + "_" + constants.ITERATION + ".config"
        with open(HYPERPARAMS_PATH, "w") as f:
            print("Version: ", constants.TRANSMISSION_ESTIMATOR_CHECKPATH, file=f)
            print("Comment: ", comments, file=f)
            print("Learning rate for G: ", str(self.g_lr), file=f)
            print("Learning rate for D: ", str(self.d_lr), file=f)
            print("====================================", file=f)
            print("Adv weight: ", str(self.adv_weight), file=f)
            print("Likeness weight: ", str(self.likeness_weight), file=f)
            print("Edge weight: ", str(self.edge_weight), file=f)
    
    def adversarial_loss(self, pred, target):
        # loss = nn.MSELoss()
        # return loss(pred, target)
        loss = nn.BCELoss()
        return loss(pred, target)

    def likeness_loss(self, pred, target):
        loss = nn.MSELoss()
        return loss(pred, target)

    def edge_loss(self, pred, target):
        loss = nn.L1Loss()
        pred_grad = kornia.filters.spatial_gradient(pred)
        target_grad = kornia.filters.spatial_gradient(target)

        return loss(pred_grad, target_grad)

    def train(self, rgb_tensor, depth_tensor):
        depth_like = self.G_A(rgb_tensor)
        #rgb_like = self.G_B(depth_tensor)

        self.D_A.train()
        #self.D_B.train()
        self.optimizerD.zero_grad()

        prediction = self.D_A(depth_tensor)
        real_tensor = torch.ones_like(prediction)
        fake_tensor = torch.zeros_like(prediction)

        D_A_real_loss = self.adversarial_loss(self.D_A(depth_tensor), real_tensor) * self.adv_weight
        D_A_fake_loss = self.adversarial_loss(self.D_A(depth_like.detach()), fake_tensor) * self.adv_weight

        #prediction = self.D_B(rgb_tensor)
        #real_tensor = torch.ones_like(prediction)
        #fake_tensor = torch.zeros_like(prediction)

        #D_B_real_loss = self.adversarial_loss(self.D_B(rgb_tensor), real_tensor) * self.adv_weight
        #D_B_fake_loss = self.adversarial_loss(self.D_B(rgb_like.detach()), fake_tensor) * self.adv_weight

        errD = D_A_real_loss + D_A_fake_loss
        if (errD.item() > 0.1):
            errD.backward()
            self.optimizerD.step()
            self.schedulerD.step(errD)

        self.G_A.train()
        #self.G_B.train()
        self.optimizerG.zero_grad()

        #print("Shape: ", np.shape(rgb_tensor), np.shape(depth_tensor))
        A_likeness_loss = self.likeness_loss(self.G_A(rgb_tensor), depth_tensor) * self.likeness_weight
        A_edge_loss = self.edge_loss(self.G_A(rgb_tensor), depth_tensor) * self.edge_weight

        prediction = self.D_A(self.G_A(rgb_tensor))
        real_tensor = torch.ones_like(prediction)
        A_adv_loss = self.adversarial_loss(prediction, real_tensor) * self.adv_weight

        errG = A_likeness_loss + A_adv_loss + A_edge_loss
        errG.backward()
        self.optimizerG.step()
        self.schedulerG.step(errG)

        # what to put to losses dict for visdom reporting?
        self.losses_dict[constants.G_LOSS_KEY].append(errG.item())
        self.losses_dict[constants.D_OVERALL_LOSS_KEY].append(errD.item())
        self.losses_dict[constants.LIKENESS_LOSS_KEY].append(A_likeness_loss.item())
        self.losses_dict[constants.EDGE_LOSS_KEY].append(A_edge_loss.item())
        self.losses_dict[constants.G_ADV_LOSS_KEY].append(A_adv_loss.item())
        self.losses_dict[constants.D_A_FAKE_LOSS_KEY].append(D_A_fake_loss.item())
        self.losses_dict[constants.D_A_REAL_LOSS_KEY].append(D_A_real_loss.item())
        #self.losses_dict[constants.D_B_FAKE_LOSS_KEY].append(D_B_fake_loss.item())
        #self.losses_dict[constants.D_B_REAL_LOSS_KEY].append(D_B_real_loss.item())
    
    def visdom_report(self, iteration, train_gray_tensor, train_depth_tensor):
        with torch.no_grad():
            train_depth_like = self.G_A(train_gray_tensor)

            # report to visdom
            self.visdom_reporter.plot_grad_flow(self.G_A.named_parameters(), "G_A grad flow")
            self.visdom_reporter.plot_grad_flow(self.D_A.named_parameters(), "D_A grad flow")
            self.visdom_reporter.plot_finegrain_loss("Depth loss", iteration, self.losses_dict, self.caption_dict)
            self.visdom_reporter.plot_image((train_gray_tensor), "Training RGB images")
            self.visdom_reporter.plot_image((train_depth_tensor), "Training Depth images")
            self.visdom_reporter.plot_image((train_depth_like), "Training Depth-like images")

    def visdom_plot_test_image(self, test_rgb_tensor, test_gray_tensor, id):
        with torch.no_grad():
            #test_depth_like = self.G_A(test_gray_tensor)
            test_depth_like = self.G_A(test_rgb_tensor)

        self.visdom_reporter.plot_image(test_rgb_tensor, "Test RGB images - " + str(id))
        self.visdom_reporter.plot_image(test_depth_like, "Test Depth-like images - " + str(id))
    
    def load_saved_state(self, iteration, checkpoint, generator_key, discriminator_key, optimizer_key):
        self.G_A.load_state_dict(checkpoint[generator_key + "A"])
        #self.G_B.load_state_dict(checkpoint[generator_key + "B"])
        self.D_A.load_state_dict(checkpoint[discriminator_key + "A"])
        #self.D_B.load_state_dict(checkpoint[discriminator_key + "B"])
        self.optimizerG.load_state_dict(checkpoint[generator_key + optimizer_key])
        self.optimizerD.load_state_dict(checkpoint[discriminator_key + optimizer_key])

        self.schedulerG.load_state_dict(checkpoint[generator_key + "scheduler"])
        self.schedulerD.load_state_dict(checkpoint[discriminator_key + "scheduler"])
    
    def save_states(self, epoch, iteration, path, generator_key, discriminator_key, optimizer_key):
        save_dict = {'epoch': epoch, 'iteration': iteration}
        netGA_state_dict = self.G_A.state_dict()
        #netGB_state_dict = self.G_B.state_dict()
        netDA_state_dict = self.D_A.state_dict()
        #netDB_state_dict = self.D_B.state_dict()

        optimizerG_state_dict = self.optimizerG.state_dict()
        optimizerD_state_dict = self.optimizerD.state_dict()

        schedulerG_state_dict = self.schedulerG.state_dict()
        schedulerD_state_dict = self.schedulerD.state_dict()

        save_dict[generator_key + "A"] = netGA_state_dict
        #save_dict[generator_key + "B"] = netGB_state_dict
        save_dict[discriminator_key + "A"] = netDA_state_dict
        #save_dict[discriminator_key + "B"] = netDB_state_dict

        save_dict[generator_key + optimizer_key] = optimizerG_state_dict
        save_dict[discriminator_key + optimizer_key] = optimizerD_state_dict

        save_dict[generator_key + "scheduler"] = schedulerG_state_dict
        save_dict[discriminator_key + "scheduler"] = schedulerD_state_dict

        torch.save(save_dict, path)
        print("Saved model state: %s Epoch: %d" % (len(save_dict), (epoch + 1)))