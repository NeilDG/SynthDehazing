# -*- coding: utf-8 -*-
"""
Created on Sat May 16 21:44:51 2020

@author: delgallegon
"""
from loaders import dataset_loader
import matplotlib.pyplot as plt
import constants
import torch
import torch.nn as nn
import numpy as np
import cv2
import torchvision.transforms as vtransforms
import torchvision.utils as vutils
from utils import tensor_utils

def visualize_color_distribution(batch_size):
    count = 2000
    dataloader = dataset_loader.load_debug_dataset(batch_size, count)
    
    # Plot some training images
    name_batch, normal_batch, homog_batch, topdown_batch = next(iter(dataloader))
    plt.figure(figsize=(constants.FIG_SIZE,constants.FIG_SIZE))
    plt.axis("off")
    plt.title("Training - Normal Images")
    plt.imshow(np.transpose(vutils.make_grid(normal_batch[:batch_size], nrow = 16, padding=2, normalize=True),(1,2,0)))
    plt.show()
    
    plt.figure(figsize=(constants.FIG_SIZE,constants.FIG_SIZE))
    plt.axis("off")
    plt.title("Training - Homog Images")
    plt.imshow(np.transpose(vutils.make_grid(homog_batch[:batch_size], nrow = 16, padding=2, normalize=True),(1,2,0)))
    plt.show()
    
    rgb_count = np.empty((count, 3), dtype=np.int32)
    index = 0
    for i, (name, normal_batch, homog_batch, topdown_batch) in enumerate(dataloader, 0):
        for n, normal_tensor, homog_tensor, topdown_tensor in zip(name, normal_batch, homog_batch, topdown_batch):
            normal_img = tensor_utils.convert_to_opencv(normal_tensor)
            homog_img = tensor_utils.convert_to_opencv(homog_tensor)
            topdown_img = tensor_utils.convert_to_opencv(topdown_tensor)
            
            rgb_count[index, 0] += np.mean(normal_img[:,:,0] * 255.0)
            rgb_count[index, 1] += np.mean(normal_img[:,:,1] * 255.0)
            rgb_count[index, 2] += np.mean(normal_img[:,:,2] * 255.0)
            
            if(np.mean(normal_img[:,:,0] * 255.0) < 120.0 or np.mean(normal_img[:,:,1] * 255.0) < 120.0 or 
               np.mean(normal_img[:,:,2] * 255.0) < 120.0):
                plt.imshow(normal_img)
                plt.show()
                print(n)
            
            index = index + 1
    
    plt.scatter(np.random.normal(0, 1.0, count), rgb_count[:,0], color=(1,0,0))
    plt.scatter(np.random.normal(0, 1.0, count), rgb_count[:,1], color=(0,1,0))
    plt.scatter(np.random.normal(0, 1.0, count), rgb_count[:,2], color=(0,0,1))
    plt.show()


def main():
    visualize_color_distribution(constants.infer_size)
    
if __name__=="__main__": 
    main()   