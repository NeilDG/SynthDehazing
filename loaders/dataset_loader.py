# -*- coding: utf-8 -*-
"""
Dataset loader
Created on Fri Jun  7 19:01:36 2019

@author: delgallegon
"""

import torch
from torch.utils import data
from loaders import image_dataset
import constants
import os
from torchvision import transforms


def assemble_train_data(num_image_to_load = -1):
    rgb_list = [];
    
    images = os.listdir(constants.DATASET_PATH)
    image_len = len(images)
    
    if(num_image_to_load > 0):
        image_len = num_image_to_load
    
    for i in range(image_len): #len(images)
        rgbImagePath = constants.DATASET_PATH + images[i]
        rgb_list.append(rgbImagePath)  
        
    return rgb_list

def load_dataset(batch_size = 8, num_image_to_load = -1):
    rgb_list = assemble_train_data(num_image_to_load = num_image_to_load)
    print("Length of train images: ", len(rgb_list))

    train_dataset = image_dataset.TorchImageDataset(rgb_list)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=8,
        shuffle=True
    )
    
    return train_loader