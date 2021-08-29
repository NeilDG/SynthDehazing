# -*- coding: utf-8 -*-

import os
from model import vanilla_cycle_gan as cg
from model import ffa_net as ffa_gan
from model import unet_gan as un
from model import dehaze_discriminator as dh
import constants
import torch
import itertools
import torch.nn as nn
from utils import plot_utils
from utils import tensor_utils
import torch.cuda.amp as amp
import kornia

class TransmissionTrainer:
    
    def __init__(self, gpu_device, batch_size, is_unet, num_blocks, has_dropout, g_lr = 0.0002, d_lr = 0.0002):
        self.gpu_device = gpu_device
        self.g_lr = g_lr
        self.d_lr = d_lr
        self.batch_size = batch_size

        if(is_unet == 1):
            self.G_T = un.UnetGenerator(input_nc=3, output_nc=1, num_downs = num_blocks).to(self.gpu_device)
        else:
            self.G_T = cg.Generator(input_nc = 3, output_nc = 1, n_residual_blocks = num_blocks,  has_dropout=has_dropout).to(self.gpu_device)

        self.D_T = dh.Discriminator(input_nc = 1).to(self.gpu_device)

        self.visdom_reporter = plot_utils.VisdomReporter()
        self.initialize_dict()

        self.optimizerG = torch.optim.Adam(itertools.chain(self.G_T.parameters()), lr=self.g_lr)
        self.optimizerD = torch.optim.Adam(itertools.chain(self.D_T.parameters()), lr=self.d_lr)
        self.schedulerG = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizerG, patience = 100000 / batch_size, threshold = 0.00005)
        self.schedulerD = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizerD, patience = 100000 / batch_size, threshold = 0.00005)

        self.fp16_scaler = amp.GradScaler()  # for automatic mixed precision

        checkpt = torch.load(constants.ALBEDO_CHECKPT)
        self.albedo_G = ffa_gan.FFA(gps=3, blocks=18).to(self.gpu_device)
        self.albedo_G.load_state_dict(checkpt[constants.GENERATOR_KEY + "A"])
        self.albedo_G.eval()
        print("Albedo network loaded.")
        
    
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

        self.caption_dict = {}
        self.caption_dict[constants.G_LOSS_KEY] = "G loss per iteration"
        self.caption_dict[constants.D_OVERALL_LOSS_KEY] = "D loss per iteration"
        self.caption_dict[constants.LIKENESS_LOSS_KEY] = "Likeness loss per iteration"
        self.caption_dict[constants.EDGE_LOSS_KEY] = "Edge loss per iteration"
        self.caption_dict[constants.G_ADV_LOSS_KEY] = "G adv loss per iteration"
        self.caption_dict[constants.D_A_FAKE_LOSS_KEY] = "D(A) fake loss per iteration"
        self.caption_dict[constants.D_A_REAL_LOSS_KEY] = "D(A) real loss per iteration"
        
    
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
        loss = nn.BCEWithLogitsLoss()
        #loss = nn.L1Loss()
        return loss(pred, target)

    def likeness_loss(self, pred, target):
        loss = nn.L1Loss()
        return loss(pred, target)

    def edge_loss(self, pred, target):
        loss = nn.L1Loss()
        pred_grad = kornia.filters.spatial_gradient(pred)
        target_grad = kornia.filters.spatial_gradient(target)

        return loss(pred_grad, target_grad)

    def train(self, iteration, hazy_tensor, transmission_tensor, unlit_enabled = 1):
        with amp.autocast():
            if(unlit_enabled == 1):
                albedo_tensor = self.albedo_G(hazy_tensor)
                depth_like = self.G_T(albedo_tensor)
            else:
                #print("Unlit disabled")
                depth_like = self.G_T(hazy_tensor)

            self.D_T.train()
            self.optimizerD.zero_grad()

            prediction = self.D_T(transmission_tensor)
            real_tensor = torch.ones_like(prediction)
            fake_tensor = torch.zeros_like(prediction)

            D_A_real_loss = self.adversarial_loss(self.D_T(transmission_tensor), real_tensor) * self.adv_weight
            D_A_fake_loss = self.adversarial_loss(self.D_T(depth_like.detach()), fake_tensor) * self.adv_weight

            errD = D_A_real_loss + D_A_fake_loss
            self.fp16_scaler.scale(errD).backward()
            if (iteration % (512 / self.batch_size) == 0):
                self.fp16_scaler.step(self.optimizerD)
                self.schedulerD.step(errD)

            self.G_T.train()
            self.optimizerG.zero_grad()

            A_likeness_loss = self.likeness_loss(self.G_T(hazy_tensor), transmission_tensor) * self.likeness_weight
            A_edge_loss = self.edge_loss(self.G_T(hazy_tensor), transmission_tensor) * self.edge_weight

            prediction = self.D_T(self.G_T(hazy_tensor))
            real_tensor = torch.ones_like(prediction)
            A_adv_loss = self.adversarial_loss(prediction, real_tensor) * self.adv_weight

            errG = A_likeness_loss + A_adv_loss + A_edge_loss
            self.fp16_scaler.scale(errG).backward()
            if (iteration % (512 / self.batch_size) == 0):
                self.fp16_scaler.step(self.optimizerG)
                self.schedulerG.step(errG)
                self.fp16_scaler.update()

            # what to put to losses dict for visdom reporting?
            self.losses_dict[constants.G_LOSS_KEY].append(errG.item())
            self.losses_dict[constants.D_OVERALL_LOSS_KEY].append(errD.item())
            self.losses_dict[constants.LIKENESS_LOSS_KEY].append(A_likeness_loss.item())
            self.losses_dict[constants.EDGE_LOSS_KEY].append(A_edge_loss.item())
            self.losses_dict[constants.G_ADV_LOSS_KEY].append(A_adv_loss.item())
            self.losses_dict[constants.D_A_FAKE_LOSS_KEY].append(D_A_fake_loss.item())
            self.losses_dict[constants.D_A_REAL_LOSS_KEY].append(D_A_real_loss.item())

    def test(self, hazy_tensor, unlit_enabled = 1):
        with torch.no_grad():
            if (unlit_enabled == 1):
                albedo_tensor = self.albedo_G(hazy_tensor)
                transmission_like = self.G_T(albedo_tensor)
            else:
                transmission_like = self.G_T(hazy_tensor)

            return transmission_like
    
    def visdom_report(self, iteration):
        self.visdom_reporter.plot_finegrain_loss("Transmission loss - " +str(constants.TRANSMISSION_VERSION) +str(constants.ITERATION), iteration, self.losses_dict, self.caption_dict)

    def visdom_infer_train(self, train_hazy_tensor, train_transmission_tensor, id):
        with torch.no_grad():
            train_depth_like = self.G_T(self.albedo_G(train_hazy_tensor))

        self.visdom_reporter.plot_image((train_hazy_tensor), str(id) + " Training - RGB " + str(constants.TRANSMISSION_VERSION) + str(constants.ITERATION))
        self.visdom_reporter.plot_image((train_transmission_tensor), str(id) + " Training - Transmission " + str(constants.TRANSMISSION_VERSION) + str(constants.ITERATION))
        self.visdom_reporter.plot_image((train_depth_like), str(id) + " Training -  Transmission-Like " +str(constants.TRANSMISSION_VERSION) +str(constants.ITERATION))

    def visdom_infer_test(self, test_hazy_tensor, id):
        with torch.no_grad():
            test_depth_like = self.G_T(self.albedo_G(test_hazy_tensor))

        self.visdom_reporter.plot_image(test_hazy_tensor, str(id) + " Test - RGB " + str(constants.TRANSMISSION_VERSION) + str(constants.ITERATION))
        self.visdom_reporter.plot_image(test_depth_like, str(id) + " Test - Transmission-Like" + str(constants.TRANSMISSION_VERSION) + str(constants.ITERATION))
    
    def load_saved_state(self, checkpoint):
        self.G_T.load_state_dict(checkpoint[constants.GENERATOR_KEY + "T"])
        self.D_T.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + "T"])
        self.optimizerG.load_state_dict(checkpoint[constants.GENERATOR_KEY + constants.OPTIMIZER_KEY])
        self.optimizerD.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + constants.OPTIMIZER_KEY])

        self.schedulerG.load_state_dict(checkpoint[constants.GENERATOR_KEY + "scheduler"])
        self.schedulerD.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + "scheduler"])
    
    def save_states(self, epoch, iteration):
        save_dict = {'epoch': epoch, 'iteration': iteration}
        netGA_state_dict = self.G_T.state_dict()
        netDA_state_dict = self.D_T.state_dict()

        optimizerG_state_dict = self.optimizerG.state_dict()
        optimizerD_state_dict = self.optimizerD.state_dict()

        schedulerG_state_dict = self.schedulerG.state_dict()
        schedulerD_state_dict = self.schedulerD.state_dict()

        save_dict[constants.GENERATOR_KEY + "T"] = netGA_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + "T"] = netDA_state_dict

        save_dict[constants.GENERATOR_KEY + constants.OPTIMIZER_KEY] = optimizerG_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + constants.OPTIMIZER_KEY] = optimizerD_state_dict

        save_dict[constants.GENERATOR_KEY + "scheduler"] = schedulerG_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + "scheduler"] = schedulerD_state_dict

        torch.save(save_dict, constants.TRANSMISSION_ESTIMATOR_CHECKPATH)
        print("Saved model state: %s Epoch: %d" % (len(save_dict), (epoch + 1)))

    def save_states_unstable(self, epoch, iteration):
        save_dict = {'epoch': epoch, 'iteration': iteration}
        netGA_state_dict = self.G_T.state_dict()
        netDA_state_dict = self.D_T.state_dict()

        optimizerG_state_dict = self.optimizerG.state_dict()
        optimizerD_state_dict = self.optimizerD.state_dict()

        schedulerG_state_dict = self.schedulerG.state_dict()
        schedulerD_state_dict = self.schedulerD.state_dict()

        save_dict[constants.GENERATOR_KEY + "T"] = netGA_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + "T"] = netDA_state_dict

        save_dict[constants.GENERATOR_KEY + constants.OPTIMIZER_KEY] = optimizerG_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + constants.OPTIMIZER_KEY] = optimizerD_state_dict

        save_dict[constants.GENERATOR_KEY + "scheduler"] = schedulerG_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + "scheduler"] = schedulerD_state_dict

        torch.save(save_dict, constants.TRANSMISSION_ESTIMATOR_CHECKPATH + ".checkpt")
        print("Saved checkpt: %s Epoch: %d" % (len(save_dict), (epoch + 1)))

        # #clear plots to avoid potential sudden jumps in visualization due to unstable gradients during early training
        # if(epoch % 5 == 0):
        #     self.losses_dict[constants.G_LOSS_KEY].clear()
        #     self.losses_dict[constants.D_OVERALL_LOSS_KEY].clear()
        #     self.losses_dict[constants.LIKENESS_LOSS_KEY].clear()
        #     self.losses_dict[constants.EDGE_LOSS_KEY].clear()
        #     self.losses_dict[constants.G_ADV_LOSS_KEY].clear()
        #     self.losses_dict[constants.D_A_FAKE_LOSS_KEY].clear()
        #     self.losses_dict[constants.D_A_REAL_LOSS_KEY].clear()