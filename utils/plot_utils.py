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

SALIKSIK_SERVER = "192.168.134.223" #IMPORTANT: No HTTP

class VisdomReporter:
    _sharedInstance = None

    @staticmethod
    def initialize():
        VisdomReporter._sharedInstance = VisdomReporter()

    @staticmethod
    def getInstance():
        return VisdomReporter._sharedInstance

    def __init__(self):
        if(constants.is_coare == 1):
            self.vis = visdom.Visdom(SALIKSIK_SERVER, use_incoming_socket=False) #TODO: Note that this is set to TRUE for observation.
        else:
            self.vis= visdom.Visdom()
        
        self.image_windows = {}
        self.loss_windows = {}
    
    def plot_image(self, img_tensor, caption):
        # if(constants.is_coare == 1):
        #     return
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
        # if(constants.is_coare == 1):
        #     #TODO: fix issue on matplot user permission for COARE
        #     return
        
        loss_keys = list(losses_dict.keys())
        caption_keys = list(caption_dict.keys())
        colors = ['r', 'g', 'black', 'darkorange', 'olive', 'palevioletred', 'rosybrown', 'cyan', 'slategray', 'darkmagenta', 'linen', 'chocolate']
        index = 0
        
        x = [i for i in range(iteration, iteration + len(losses_dict[constants.D_OVERALL_LOSS_KEY]))]
        COLS = 3; ROWS = 4
        fig, ax = plt.subplots(ROWS, COLS, sharex=True)
        fig.set_size_inches(9, 9)
        fig.tight_layout()
        
        for i in range(ROWS):
            for j in range(COLS):
                if(index < len(loss_keys)):
                    if(index == 1):
                        ax[i, j].plot(x, losses_dict[loss_keys[index]], color=colors[index], label= loss_key + " " +str(caption_dict[caption_keys[index]]))
                    else:
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

    def plot_train_test_loss(self, loss_key, iteration, train_losses, test_losses, train_caption, test_caption):
        colors = ['r', 'g', 'black', 'darkorange', 'olive', 'palevioletred', 'rosybrown', 'cyan', 'slategray', 'darkmagenta', 'linen', 'chocolate']

        x1 = [i for i in range(iteration, iteration + len(train_losses))]
        x2 = [i for i in range(iteration, iteration + len(test_losses))]

        plt.plot(x1, train_losses, color=colors[0], label=str(train_caption))
        plt.plot(x2, test_losses, color=colors[1], label=str(test_caption))
        plt.legend(loc='lower right')

        if loss_key not in self.loss_windows:
            self.loss_windows[loss_key] = self.vis.matplot(plt, opts=dict(caption="Losses" + " " + str(constants)))
        else:
            self.vis.matplot(plt, win=self.loss_windows[loss_key], opts=dict(caption="Losses" + " " + str(constants)))

        plt.show()

    def plot_airlight_comparison(self, loss_key, iteration, airlight_loss, airlight_captions):
        colors = ['r', 'g', 'black', 'darkorange', 'olive', 'palevioletred', 'rosybrown', 'cyan', 'slategray', 'darkmagenta', 'linen', 'chocolate']

        x1 = [i for i in range(iteration, iteration + len(airlight_loss[0]))]
        x2 = [i for i in range(iteration, iteration + len(airlight_loss[1]))]

        plt.plot(x1, airlight_loss[0], color=colors[0], label=str(airlight_captions[0]))
        plt.plot(x2, airlight_loss[1], color=colors[1], label=str(airlight_captions[1]))
        plt.legend(loc='lower right')

        if loss_key not in self.loss_windows:
            self.loss_windows[loss_key] = self.vis.matplot(plt, opts=dict(caption="Losses" + " " + str(constants)))
        else:
            self.vis.matplot(plt, win=self.loss_windows[loss_key], opts=dict(caption="Losses" + " " + str(constants)))

        plt.show()

    def plot_psnr_ssim_loss(self, loss_key, iteration, losses_dict, caption_dict, base_key):
        # if (constants.is_coare == 1):
        #     # TODO: fix issue on matplot user permission for COARE
        #     return

        loss_keys = list(losses_dict.keys())
        caption_keys = list(caption_dict.keys())

        colors = ['r', 'g', 'black', 'darkorange', 'olive', 'palevioletred', 'rosybrown', 'cyan', 'slategray',
                  'darkmagenta', 'linen', 'chocolate']

        x = [i for i in range(iteration, iteration + len(losses_dict[base_key]))]
        COLS = 2;
        fig, ax = plt.subplots(1, COLS, sharex=True)
        fig.set_size_inches(5, 5)
        fig.tight_layout()

        for j in range(COLS):
            ax[j].plot(x, losses_dict[loss_keys[j]], color=colors[j], label=str(caption_dict[caption_keys[j]]))

        fig.legend(loc='lower right')
        if loss_key not in self.loss_windows:
            self.loss_windows[loss_key] = self.vis.matplot(plt, opts=dict(caption="Losses" + " " + str(constants)))
        else:
            self.vis.matplot(plt, win=self.loss_windows[loss_key], opts=dict(caption="Losses" + " " + str(constants)))

        plt.show()