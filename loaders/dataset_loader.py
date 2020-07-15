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
from utils import logger

print = logger.log
def assemble_train_data(num_image_to_load = -1):
    a_list = []; b_list = []
    
    for (root, dirs, files) in os.walk(constants.DATASET_VEMON_PATH):
        for f in files:
            file_name = os.path.join(root, f)
            #print(file_name)
            a_list.append(file_name)
            if(num_image_to_load != -1 and len(a_list) == num_image_to_load):
                break  
    
    for (root, dirs, files) in os.walk(constants.DATASET_DIV2K_PATH):
        for f in files:
            file_name = os.path.join(root, f)
            b_list.append(file_name)
            if(num_image_to_load != -1 and len(b_list) == num_image_to_load):
                break
    
    return a_list, b_list

def load_test_dataset(batch_size = 8, num_image_to_load = -1):
    a_list, b_list = assemble_train_data(num_image_to_load)
    print("Length of images: %d, %d." % (len(a_list), len(b_list)))

    data_loader = torch.utils.data.DataLoader(
        image_dataset.TestDataset(a_list, b_list),
        batch_size=batch_size,
        num_workers=constants.num_workers,
        shuffle=True
    )
    
    return data_loader

def load_train_dataset(batch_size = 8, num_image_to_load = -1):
    a_list, b_list = assemble_train_data(num_image_to_load)
    print("Length of images: %d, %d." % (len(a_list), len(b_list)))

    data_loader = torch.utils.data.DataLoader(
        image_dataset.StyleDataset(a_list, b_list),
        batch_size=batch_size,
        num_workers=3,
        shuffle=True
    )
    
    return data_loader