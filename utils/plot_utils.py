# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 17:02:01 2020

@author: delgallegon
"""
import constants
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import visdom

class VisdomReporter:
    def __init__(self):
        if(constants.is_coare == 1):
            return #do nothing. can't run visdom on COARE
        
        self.vis= visdom.Visdom()
        self.image_windows = {}
        self.loss_windows = {}
    
    #post current image to vis
    def plot_image(self, dirty_tensor, dirty_like, clean_tensor, clean_like):
        if(constants.is_coare == 1):
            return #do nothing. can't run visdom on COARE
        
        DIRTY_KEY = "A"
        DIRTY_LIKE_KEY = "B"
        CLEAN_KEY = "C"
        CLEAN_LIKE_KEY = "D"
        
        dirty_group = vutils.make_grid(dirty_tensor[:constants.display_size], nrow = 8, padding=2, normalize=True).cpu()
        dirty_like_group = vutils.make_grid(dirty_like[:constants.display_size], nrow = 8, padding=2, normalize=True).cpu()
        clean_group = vutils.make_grid(clean_tensor[:constants.display_size], nrow = 8, padding=2, normalize=True).cpu()
        clean_like_group = vutils.make_grid(clean_like[:constants.display_size], nrow = 8, padding=2, normalize=True).cpu()
        
        if DIRTY_KEY not in self.image_windows:
            self.image_windows[DIRTY_KEY] = self.vis.images(dirty_group, opts = dict(caption = "Dirty images"))
        else:
            self.vis.images(dirty_group, win = self.image_windows[DIRTY_KEY], opts = dict(caption = "Dirty images"))
        
        if DIRTY_LIKE_KEY not in self.image_windows:
            self.image_windows[DIRTY_LIKE_KEY] = self.vis.images(dirty_like_group, opts = dict(caption = "Fake dirty images"))
        else:
            self.vis.images(dirty_like_group, win = self.image_windows[DIRTY_LIKE_KEY], opts = dict(caption = "Fake dirty images"))
        
        if CLEAN_KEY not in self.image_windows:
            self.image_windows[CLEAN_KEY] = self.vis.images(clean_group, opts = dict(caption = "Clean images"))
        else:
            self.vis.images(clean_group, win = self.image_windows[CLEAN_KEY], opts = dict(caption = "Clean images"))
        
        if CLEAN_LIKE_KEY not in self.image_windows:
            self.image_windows[CLEAN_LIKE_KEY] = self.vis.images(clean_like_group, opts = dict(caption = "Fake clean images"))
        else:
            self.vis.images(clean_like_group, win = self.image_windows[CLEAN_LIKE_KEY], opts = dict(caption = "Fake clean images"))
    
    def plot_finegrain_loss(self, iteration, G_loss_dict, D_loss_dict):
        if(constants.is_coare == 1):
            return #do nothing. can't run visdom on COARE
        
        LOSS_KEY = "FINEGRAIN_LOSS"
        x = [i for i in range(iteration, iteration + len(G_loss_dict[constants.IDENTITY_LOSS_KEY]))]
        fig, ax1 = plt.subplots()
        fig.tight_layout()
        
        ax1.plot(x, G_loss_dict[constants.IDENTITY_LOSS_KEY], color = 'r', label = "Id loss per iteration"); plt.legend()
        ax2 = ax1.twinx()
        ax2.plot(x, G_loss_dict[constants.CYCLE_LOSS_KEY], color = 'g', label = "Cycle loss per iteration"); plt.legend()
        ax3 = ax2.twinx()
        ax3.plot(x, G_loss_dict[constants.TV_LOSS_KEY], color = 'b', label = "TV loss per iteration"); plt.legend()
        ax4 = ax3.twinx()
        ax4.plot(x, G_loss_dict[constants.ADV_LOSS_KEY], color = 'c', label = "G Adv loss per iteration"); plt.legend()
        ax5 = ax4.twinx()
        ax5.plot(x, D_loss_dict[constants.D_FAKE_LOSS_KEY], color = 'c', label = "D Fake loss per iteration"); plt.legend()
        ax6 = ax5.twinx()
        ax6.plot(x, D_loss_dict[constants.D_REAL_LOSS_KEY], color = 'c', label = "D Real loss per iteration"); plt.legend()
        
        ax1.set_ylim(0, 1.0)
        ax2.set_ylim(0, 1.0)
        ax3.set_ylim(0, 1.0)
        ax4.set_ylim(0, 1.0)
        ax5.set_ylim(0, 1.0)
        ax6.set_ylim(0, 1.0)
        if LOSS_KEY not in self.loss_windows:
            self.loss_windows[LOSS_KEY] = self.vis.matplot(plt)
        else:
           self.vis.matplot(plt, win = self.loss_windows[LOSS_KEY]) 
          
        plt.show()
        
    def plot_loss(self, iteration, G_losses, D_losses):
        if(constants.is_coare == 1):
            return #do nothing. can't run visdom on COARE
        
        LOSS_KEY = "LOSS"
        x = [i for i in range(iteration, iteration + len(G_losses))]
        fig, ax = plt.subplots()
        fig.tight_layout()
        
        ax.plot(x, G_losses, color = 'r', label = "G Losses per iteration"); plt.legend()
        ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.plot(x, D_losses, color = 'g', label = "D Losses per iteration"); plt.legend()
        ax.set_ylim(0, 4.0)
        ax2.set_ylim(0, 4.0)
        if LOSS_KEY not in self.loss_windows:
            self.loss_windows[LOSS_KEY] = self.vis.matplot(plt)
        else:
           self.vis.matplot(plt, win = self.loss_windows[LOSS_KEY]) 
          
        plt.show()