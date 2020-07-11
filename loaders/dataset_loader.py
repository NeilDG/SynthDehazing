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
def assemble_msg_data(num_image_to_load = -1):
    gta_list = [];  normal_list = []
    
    for (root, dirs, files) in os.walk(constants.DATASET_VEMON_PATH):
        for f in files:
            file_name = os.path.join(root, f)
            #print(file_name)
            normal_list.append(file_name)
            if(num_image_to_load != -1 and len(normal_list) == num_image_to_load):
                break  
    
    for (root, dirs, files) in os.walk(constants.DATASET_GTA_PATH_2):
        for f in files:
            file_name = os.path.join(root, f)
            #print(file_name)
            gta_list.append(file_name)
            if(len(normal_list) == len(gta_list)):
                break
    
    return normal_list, gta_list

def load_msg_dataset(batch_size = 8, num_image_to_load = -1):
    normal_list, gta_list = assemble_msg_data(num_image_to_load)

    print("Length of MSG images: %d, %d." % (len(normal_list), len(gta_list)))
    
    test_dataset = image_dataset.StyleDataset(normal_list, gta_list)
    train_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=constants.num_workers,
        shuffle=True
    )
    
    return train_loader

def load_test_dataset(batch_size = 8, num_image_to_load = -1):
    normal_list, gta_list = assemble_msg_data(num_image_to_load)

    #print("Length of test images: %d, %d." % (len(normal_list), len(gta_list)))
    
    test_dataset = image_dataset.TestDataset(normal_list, gta_list)
    train_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=2,
        shuffle=True
    )
    
    return train_loader