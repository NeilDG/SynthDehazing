# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 17:02:01 2020

@author: delgallegon
"""
from matplotlib.lines import Line2D

import constants
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import visdom

class VisdomReporter:
    def __init__(self):
        if(constants.is_coare == 1):
            self.vis = None
        else:
            self.vis= visdom.Visdom()
        
        self.image_windows = {}
        self.loss_windows = {}
    
    def plot_image(self, img_tensor, caption):
        if(constants.is_coare == 1):
            return
        
        img_group = vutils.make_grid(img_tensor[:constants.display_size], nrow = 8, padding=2, normalize=True).cpu()
        if hash(caption) not in self.image_windows:
            self.image_windows[hash(caption)] = self.vis.images(img_group, opts = dict(caption = caption + " " + str(constants.ITERATION)))
        else:
            self.vis.images(img_group, win = self.image_windows[hash(caption)], opts = dict(caption = caption + " " + str(constants.ITERATION)))

    def plot_grad_flow(self, named_parameters, caption):
        '''Plots the gradients flowing through different layers in the net during training.
        Can be used for checking for possible gradient vanishing / exploding problems.

        Usage: Plug this function in Trainer class after loss.backwards() as
        "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
        ave_grads = []
        max_grads = []
        layers = []
        for n, p in named_parameters:
            if (p.requires_grad) and ("bias" not in n):
                layers.append(n)
                ave_grads.append(p.grad.abs().mean())
                max_grads.append(p.grad.abs().max())
        plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="r")
        plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
        plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="g")
        plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(left=0, right=len(ave_grads))
        plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow")
        plt.grid(True)
        plt.legend([Line2D([0], [0], color="r", lw=4),
                    Line2D([0], [0], color="b", lw=4),
                    Line2D([0], [0], color="g", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])

        if hash(caption) not in self.loss_windows:
            self.loss_windows[hash(caption)] = self.vis.matplot(plt, opts = dict(caption = caption))
        else:
            self.vis.matplot(plt, win = self.loss_windows[hash(caption)], opts = dict(caption = caption))

    def plot_finegrain_loss(self, loss_key, iteration, losses_dict, caption_dict):        
        if(constants.is_coare == 1):
            #TODO: fix issue on matplot user permission for COARE
            return
        
        loss_keys = list(losses_dict.keys())
        caption_keys = list(caption_dict.keys())
        colors = ['r', 'g', 'black', 'darkorange', 'olive', 'palevioletred', 'rosybrown', 'cyan', 'slategray', 'darkmagenta', 'linen', 'chocolate']
        index = 0
        
        x = [i for i in range(iteration, iteration + len(losses_dict[constants.G_LOSS_KEY]))]
        COLS = 3; ROWS = 4
        fig, ax = plt.subplots(ROWS, COLS, sharex=True)
        fig.set_size_inches(9, 9)
        fig.tight_layout()
        
        for i in range(ROWS):
            for j in range(COLS):
                if(index < len(loss_keys)):
                    ax[i, j].plot(x, losses_dict[loss_keys[index]], color = colors[index], label = str(caption_dict[caption_keys[index]]))
                    index = index + 1
                else:
                    break
    
        fig.legend(loc = 'lower right')
        if loss_key not in self.loss_windows:
            self.loss_windows[loss_key] = self.vis.matplot(plt, opts = dict(caption = "Losses" + " " + str(constants)))
        else:
            self.vis.matplot(plt, win = self.loss_windows[loss_key], opts = dict(caption = "Losses" + " " + str(constants))) 
          
        plt.show()
        
    # def plot_finegrain_loss(self, loss_key, iteration, losses_dict):        
    #     if(constants.is_coare == 1):
    #         #TODO: fix issue on matplot user permission for COARE
    #         return
        
    #     x = [i for i in range(iteration, iteration + len(losses_dict[constants.LIKENESS_LOSS_KEY]))]
    #     fig, ax = plt.subplots(4, 3, sharex=True)
    #     fig.set_size_inches(9, 9)
    #     fig.tight_layout()
        
    #     ax[0,0].plot(x, losses_dict[constants.REALNESS_LOSS_KEY], color = 'r', label = "Realness loss per iteration")
    #     ax[0,1].plot(x, losses_dict[constants.LIKENESS_LOSS_KEY], color = 'g', label = "Clarity loss per iteration")
    #     ax[0,2].plot(x, losses_dict[constants.G_LOSS_KEY], color = 'black', label = "G Overall loss per iteration")
    #     ax[1,0].plot(x, losses_dict[constants.D_OVERALL_LOSS_KEY], color = 'darkorange', label = "D overall loss per iteration")
    #     ax[1,1].plot(x, losses_dict[constants.G_ADV_LOSS_KEY], color = 'olive', label = "G Adv loss per iteration")
    #     ax[1,2].plot(x, losses_dict[constants.D_A_FAKE_LOSS_KEY], color = 'palevioletred', label = "D(A) fake loss")
    #     ax[2,0].plot(x, losses_dict[constants.D_A_REAL_LOSS_KEY], color = 'rosybrown', label = "D(A) real loss")
    #     ax[2,1].plot(x, losses_dict[constants.D_B_FAKE_LOSS_KEY], color = 'cyan', label = "D(B) fake loss")
    #     ax[2,2].plot(x, losses_dict[constants.D_B_REAL_LOSS_KEY], color = 'slategray', label = "D(B) real loss")
    
    #     fig.legend(loc = 'lower right')
    #     if loss_key not in self.loss_windows:
    #         self.loss_windows[loss_key] = self.vis.matplot(plt, opts = dict(caption = "Losses" + " " + str(constants)))
    #     else:
    #        self.vis.matplot(plt, win = self.loss_windows[loss_key], opts = dict(caption = "Losses" + " " + str(constants))) 
          
    #     plt.show()