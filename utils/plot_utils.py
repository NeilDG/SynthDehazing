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
        
        
    #post current image to vis
    # def plot_image(self, dirty_tensor, dirty_like, clean_tensor, clean_like):
    #     if(constants.is_coare == 1):
    #         #TODO: Fix COARE deadlock issue
    #         return
        
    #     DIRTY_KEY = "A"
    #     DIRTY_LIKE_KEY = "B"
    #     CLEAN_KEY = "C"
    #     CLEAN_LIKE_KEY = "D"
        
    #     dirty_group = vutils.make_grid(dirty_tensor[:constants.display_size], nrow = 8, padding=2, normalize=True).cpu()
    #     dirty_like_group = vutils.make_grid(dirty_like[:constants.display_size], nrow = 8, padding=2, normalize=True).cpu()
    #     clean_group = vutils.make_grid(clean_tensor[:constants.display_size], nrow = 8, padding=2, normalize=True).cpu()
    #     clean_like_group = vutils.make_grid(clean_like[:constants.display_size], nrow = 8, padding=2, normalize=True).cpu()
        
    #     if DIRTY_KEY not in self.image_windows:
    #         self.image_windows[DIRTY_KEY] = self.vis.images(dirty_group, opts = dict(caption = "Dirty images" + " " + str(constants.STYLE_ITERATION)))
    #     else:
    #         self.vis.images(dirty_group, win = self.image_windows[DIRTY_KEY], opts = dict(caption = "Dirty images" + " " + str(constants.STYLE_ITERATION)))
        
    #     if DIRTY_LIKE_KEY not in self.image_windows:
    #         self.image_windows[DIRTY_LIKE_KEY] = self.vis.images(dirty_like_group, opts = dict(caption = "Fake dirty images" + " " + str(constants.STYLE_ITERATION)))
    #     else:
    #         self.vis.images(dirty_like_group, win = self.image_windows[DIRTY_LIKE_KEY], opts = dict(caption = "Fake dirty images" + " " + str(constants.STYLE_ITERATION)))
        
    #     if CLEAN_KEY not in self.image_windows:
    #         self.image_windows[CLEAN_KEY] = self.vis.images(clean_group, opts = dict(caption = "Clean images" + " " + str(constants.STYLE_ITERATION)))
    #     else:
    #         self.vis.images(clean_group, win = self.image_windows[CLEAN_KEY], opts = dict(caption = "Clean images" + " " + str(constants.STYLE_ITERATION)))
        
    #     if CLEAN_LIKE_KEY not in self.image_windows:
    #         self.image_windows[CLEAN_LIKE_KEY] = self.vis.images(clean_like_group, opts = dict(caption = "Fake clean images" + " " + str(constants.STYLE_ITERATION)))
    #     else:
    #         self.vis.images(clean_like_group, win = self.image_windows[CLEAN_LIKE_KEY], opts = dict(caption = "Fake clean images" + " " + str(constants.STYLE_ITERATION)))
    
    # def plot_denoise_test_image(self, dirty_tensor, clean_tensor, clean_like):    
    #     if(constants.is_coare == 1):
    #         #TODO: Fix COARE deadlock issue
    #         return
        
    #     DIRTY_KEY = "test_A"
    #     CLEAN_KEY = "test_C"
    #     CLEAN_LIKE_KEY = "test_D"
        
    #     dirty_group = vutils.make_grid(dirty_tensor[:constants.test_display_size], nrow = 8, padding=2, normalize=True).cpu()
    #     clean_group = vutils.make_grid(clean_tensor[:constants.test_display_size], nrow = 8, padding=2, normalize=True).cpu()
    #     clean_like_group = vutils.make_grid(clean_like[:constants.test_display_size], nrow = 8, padding=2, normalize=True).cpu()
        
    #     if DIRTY_KEY not in self.image_windows:
    #         self.image_windows[DIRTY_KEY] = self.vis.images(dirty_group, opts = dict(caption = "Dirty images" + " " + str(constants.ITERATION)))
    #     else:
    #         self.vis.images(dirty_group, win = self.image_windows[DIRTY_KEY], opts = dict(caption = "Dirty images" + " " + str(constants.ITERATION)))
                
    #     if CLEAN_KEY not in self.image_windows:
    #         self.image_windows[CLEAN_KEY] = self.vis.images(clean_group, opts = dict(caption = "Clean images" + " " + str(constants.ITERATION)))
    #     else:
    #         self.vis.images(clean_group, win = self.image_windows[CLEAN_KEY], opts = dict(caption = "Clean images" + " " + str(constants.ITERATION)))
        
    #     if CLEAN_LIKE_KEY not in self.image_windows:
    #         self.image_windows[CLEAN_LIKE_KEY] = self.vis.images(clean_like_group, opts = dict(caption = "Fake clean images" + " " + str(constants.ITERATION)))
    #     else:
    #         self.vis.images(clean_like_group, win = self.image_windows[CLEAN_LIKE_KEY], opts = dict(caption = "Fake clean images" + " " + str(constants.ITERATION)))
    
    
    # def plot_test_image(self, dirty_tensor, dirty_like, clean_tensor, clean_like):    
    #     if(constants.is_coare == 1):
    #         #TODO: Fix COARE deadlock issue
    #         return
        
    #     DIRTY_KEY = "test_A"
    #     DIRTY_LIKE_KEY = "test_B"
    #     CLEAN_KEY = "test_C"
    #     CLEAN_LIKE_KEY = "test_D"
    #     SYNTH_CLEAN_KEY = "synth_test_A"
        
    #     dirty_group = vutils.make_grid(dirty_tensor[:constants.test_display_size], nrow = 8, padding=2, normalize=True).cpu()
    #     dirty_like_group = vutils.make_grid(dirty_like[:constants.test_display_size], nrow = 8, padding=2, normalize=True).cpu()
    #     clean_group = vutils.make_grid(clean_tensor[:constants.test_display_size], nrow = 8, padding=2, normalize=True).cpu()
    #     clean_like_group = vutils.make_grid(clean_like[:constants.test_display_size], nrow = 8, padding=2, normalize=True).cpu()
        
    #     if DIRTY_KEY not in self.image_windows:
    #         self.image_windows[DIRTY_KEY] = self.vis.images(dirty_group, opts = dict(caption = "Dirty images" + " " + str(constants.ITERATION)))
    #     else:
    #         self.vis.images(dirty_group, win = self.image_windows[DIRTY_KEY], opts = dict(caption = "Dirty images" + " " + str(constants.ITERATION)))
        
    #     if DIRTY_LIKE_KEY not in self.image_windows:
    #         self.image_windows[DIRTY_LIKE_KEY] = self.vis.images(dirty_like_group, opts = dict(caption = "Fake dirty images" + " " + str(constants.ITERATION)))
    #     else:
    #         self.vis.images(dirty_like_group, win = self.image_windows[DIRTY_LIKE_KEY], opts = dict(caption = "Fake dirty images" + " " + str(constants.ITERATION)))
        
    #     if CLEAN_KEY not in self.image_windows:
    #         self.image_windows[CLEAN_KEY] = self.vis.images(clean_group, opts = dict(caption = "Clean images" + " " + str(constants.ITERATION)))
    #     else:
    #         self.vis.images(clean_group, win = self.image_windows[CLEAN_KEY], opts = dict(caption = "Clean images" + " " + str(constants.ITERATION)))
        
    #     if CLEAN_LIKE_KEY not in self.image_windows:
    #         self.image_windows[CLEAN_LIKE_KEY] = self.vis.images(clean_like_group, opts = dict(caption = "Fake clean images" + " " + str(constants.ITERATION)))
    #     else:
    #         self.vis.images(clean_like_group, win = self.image_windows[CLEAN_LIKE_KEY], opts = dict(caption = "Fake clean images" + " " + str(constants.ITERATION)))
        
    # def plot_image(self, dirty_tensor, clean_tensor, clean_like):
    #     if(constants.is_coare == 1):
    #         #TODO: Fix COARE deadlock issue
    #         return
        
    #     DIRTY_KEY = "A"
    #     DIRTY_LIKE_KEY = "B"
    #     CLEAN_KEY = "C"
    #     CLEAN_LIKE_KEY = "D"
        
    #     dirty_group = vutils.make_grid(dirty_tensor[:constants.display_size], nrow = 8, padding=2, normalize=True).cpu()
    #     clean_group = vutils.make_grid(clean_tensor[:constants.display_size], nrow = 8, padding=2, normalize=True).cpu()
    #     clean_like_group = vutils.make_grid(clean_like[:constants.display_size], nrow = 8, padding=2, normalize=True).cpu()
        
    #     if DIRTY_KEY not in self.image_windows:
    #         self.image_windows[DIRTY_KEY] = self.vis.images(dirty_group, opts = dict(caption = "Dirty images" + " " + str(constants.ITERATION)))
    #     else:
    #         self.vis.images(dirty_group, win = self.image_windows[DIRTY_KEY], opts = dict(caption = "Dirty images" + " " + str(constants.ITERATION)))
        
    #     if CLEAN_KEY not in self.image_windows:
    #         self.image_windows[CLEAN_KEY] = self.vis.images(clean_group, opts = dict(caption = "Clean images" + " " + str(constants.ITERATION)))
    #     else:
    #         self.vis.images(clean_group, win = self.image_windows[CLEAN_KEY], opts = dict(caption = "Clean images" + " " + str(constants.ITERATION)))
        
    #     if CLEAN_LIKE_KEY not in self.image_windows:
    #         self.image_windows[CLEAN_LIKE_KEY] = self.vis.images(clean_like_group, opts = dict(caption = "Fake clean images" + " " + str(constants.ITERATION)))
    #     else:
    #         self.vis.images(clean_like_group, win = self.image_windows[CLEAN_LIKE_KEY], opts = dict(caption = "Fake clean images" + " " + str(constants.ITERATION)))
    
    # def plot_image(self, synth_dirty_tensor, synth_clean_tensor, synth_clean_like, real_dirty_tensor,
    #                real_clean_like):
    #     if(constants.is_coare == 1):
    #         #TODO: Fix COARE deadlock issue
    #         return
        
    #     SYNTH_DIRTY_KEY = "A"
    #     SYNTH_CLEAN_KEY = "C"
    #     SYNTH_CLEAN_LIKE_KEY = "D"
    #     REAL_DIRTY_KEY = "E"
    #     REAL_DIRTY_LIKE_KEY = "F"
    #     REAL_CLEAN_KEY = "G"
    #     REAL_CLEAN_LIKE_KEY = "H"
        
    #     synth_dirty_group = vutils.make_grid(synth_dirty_tensor[:constants.display_size], nrow = 8, padding=2, normalize=True).cpu()
    #     synth_clean_group = vutils.make_grid(synth_clean_tensor[:constants.display_size], nrow = 8, padding=2, normalize=True).cpu()
    #     real_dirty_group = vutils.make_grid(real_dirty_tensor[:constants.display_size], nrow = 8, padding=2, normalize=True).cpu()
        
    #     synth_cleanlike_group = vutils.make_grid(synth_clean_like[:constants.display_size], nrow = 8, padding=2, normalize=True).cpu()
    #     #real_dirtylike_group = vutils.make_grid(real_dirty_like[:constants.display_size], nrow = 8, padding=2, normalize=True).cpu()
    #     real_cleanlike_group = vutils.make_grid(real_clean_like[:constants.display_size], nrow = 8, padding=2, normalize=True).cpu()
        
    #     if SYNTH_DIRTY_KEY not in self.image_windows:
    #         self.image_windows[SYNTH_DIRTY_KEY] = self.vis.images(synth_dirty_group, opts = dict(caption = "Dirty images (synth)" + " " + str(constants.ITERATION)))
    #     else:
    #         self.vis.images(synth_dirty_group, win = self.image_windows[SYNTH_DIRTY_KEY], opts = dict(caption = "Dirty images (synth)" + " " + str(constants.ITERATION)))
           
    #     if SYNTH_CLEAN_KEY not in self.image_windows:
    #         self.image_windows[SYNTH_CLEAN_KEY] = self.vis.images(synth_clean_group, opts = dict(caption = "Clean images (synth)" + " " + str(constants.ITERATION)))
    #     else:
    #         self.vis.images(synth_clean_group, win = self.image_windows[SYNTH_CLEAN_KEY], opts = dict(caption = "Clean images (synth)" + " " + str(constants.ITERATION)))
        
    #     if SYNTH_CLEAN_LIKE_KEY not in self.image_windows:
    #         self.image_windows[SYNTH_CLEAN_LIKE_KEY] = self.vis.images(synth_cleanlike_group, opts = dict(caption = "Clean-like images (synth)" + " " + str(constants.ITERATION)))
    #     else:
    #         self.vis.images(synth_cleanlike_group, win = self.image_windows[SYNTH_CLEAN_LIKE_KEY], opts = dict(caption = "Clean-like images (synth)" + " " + str(constants.ITERATION)))
        
    #     if REAL_DIRTY_KEY not in self.image_windows:
    #         self.image_windows[REAL_DIRTY_KEY] = self.vis.images(real_dirty_group, opts = dict(caption = "Dirty images (real)" + " " + str(constants.ITERATION)))
    #     else:
    #         self.vis.images(real_dirty_group, win = self.image_windows[REAL_DIRTY_KEY], opts = dict(caption = "Dirty images (real)" + " " + str(constants.ITERATION)))
        
    #     # if REAL_DIRTY_LIKE_KEY not in self.image_windows:
    #     #     self.image_windows[REAL_DIRTY_LIKE_KEY] = self.vis.images(real_dirtylike_group, opts = dict(caption = "Dirty-like images (real)" + " " + str(constants.ITERATION)))
    #     # else:
    #     #     self.vis.images(real_dirtylike_group, win = self.image_windows[REAL_DIRTY_LIKE_KEY], opts = dict(caption = "Dirty-like images (real)" + " " + str(constants.ITERATION)))
        
    #     if REAL_CLEAN_LIKE_KEY not in self.image_windows:
    #         self.image_windows[REAL_CLEAN_LIKE_KEY] = self.vis.images(real_cleanlike_group, opts = dict(caption = "Clean-like images (real)" + " " + str(constants.ITERATION)))
    #     else:
    #         self.vis.images(real_cleanlike_group, win = self.image_windows[REAL_CLEAN_LIKE_KEY], opts = dict(caption = "Clean-like images (real)" + " " + str(constants.ITERATION)))
    
    def plot_finegrain_loss(self, loss_key, iteration, losses_dict, caption_dict):        
        if(constants.is_coare == 1):
            #TODO: fix issue on matplot user permission for COARE
            return
        
        x = [i for i in range(iteration, iteration + len(losses_dict[constants.G_LOSS_KEY]))]
        COLS = 3; ROWS = 4
        fig, ax = plt.subplots(ROWS, COLS, sharex=True)
        fig.set_size_inches(9, 9)
        fig.tight_layout()
        
        loss_keys = losses_dict.keys()
        caption_values = caption_dict.values()
        
        print(loss_keys)
        
        index = 0
        for i in range(ROWS):
            for j in range(COLS):
                ax[i, j].plot(x, loss_values[loss_keys], color = 'r', label = str(caption_values[index]))
                index = index + 1
    
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