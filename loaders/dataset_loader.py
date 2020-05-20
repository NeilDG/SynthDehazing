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

def assemble_test_data(num_image_to_load = -1):
    normal_list = []; homog_list = []
    
    #load normal images
    images = os.listdir(constants.DATASET_VEMON_FRONT_PATH)
    image_len = len(images)
    
    if(num_image_to_load > 0):
        image_len = num_image_to_load
        
    for i in range(image_len):
        img_path = constants.DATASET_VEMON_FRONT_PATH + images[i]
        normal_list.append(img_path)
        
        img_path = constants.DATASET_VEMON_HOMOG_PATH + images[i]
        homog_list.append(img_path)
    
    return normal_list, homog_list
        
def assemble_synth_train_data(num_image_to_load = -1):
    normal_list = []; topdown_list = []; homog_list = []
    
    #load normal images
    images = os.listdir(constants.DATASET_BIRD_NORMAL_PATH)
    image_len = len(images)
    
    if(num_image_to_load > 0):
        image_len = num_image_to_load
        
    for i in range(image_len):
        img_path = constants.DATASET_BIRD_NORMAL_PATH + images[i]
        normal_list.append(img_path)
        
    #load homog images
    images = os.listdir(constants.DATASET_BIRD_HOMOG_PATH)
    image_len = len(images)
    
    if(num_image_to_load > 0):
        image_len = num_image_to_load
    
    for i in range(image_len):
        img_path = constants.DATASET_BIRD_HOMOG_PATH + images[i]
        homog_list.append(img_path)
    
    #load topdown images
    images = os.listdir(constants.DATASET_BIRD_GROUND_TRUTH_PATH)
    image_len = len(images)
    
    if(num_image_to_load > 0):
        image_len = num_image_to_load
    
    for i in range(image_len):
        img_path = constants.DATASET_BIRD_GROUND_TRUTH_PATH + images[i]
        topdown_list.append(img_path)
        
    return normal_list, homog_list, topdown_list

def assemble_normal_data(num_image_to_load = -1):
    normal_list = []
    
    for (root, dirs, files) in os.walk(constants.DATASET_BIRD_ALTERNATIVE_PATH):
        for f in files:
            if(".jpg" in f and ("_b.jpg" in f) == False):
                file_name = os.path.join(root, f)
                #print(file_name)
                normal_list.append(file_name)
                if(num_image_to_load != -1 and len(normal_list) == num_image_to_load):
                    return normal_list
        
    return normal_list
def assemble_topdown_data(num_image_to_load = -1):
    topdown_list = []
    
    for (root, dirs, files) in os.walk(constants.DATASET_BIRD_ALTERNATIVE_PATH):
        for f in files:
            if("_b.jpg" in f):
                file_name = os.path.join(root, f)
                #print(file_name)
                topdown_list.append(file_name)
                if(num_image_to_load != -1 and len(topdown_list) == num_image_to_load):
                    return topdown_list
        
    return topdown_list


def load_synth_dataset(batch_size = 8, num_image_to_load = -1):
    normal_list, homog_list, topdown_list = assemble_synth_train_data(num_image_to_load = num_image_to_load)
    print("Length of train images: ", len(normal_list), len(homog_list), len(topdown_list))

    train_dataset = image_dataset.TorchImageDataset(normal_list, homog_list, topdown_list)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=12,
        shuffle=True
    )
    
    return train_loader

def load_vemon_dataset(batch_size = 8, num_image_to_load = -1):
    topdown_list = assemble_topdown_data(num_image_to_load)
    normal_list, homog_list = assemble_test_data(len(topdown_list)) #equalize topdown list length to loaded VEMON data
    
    print("Length of VEMON images: ", len(normal_list), len(homog_list), len(topdown_list))
    
    
    test_dataset = image_dataset.VemonImageDataset(normal_list, homog_list, topdown_list)
    train_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=12,
        shuffle=True
    )
    
    return train_loader