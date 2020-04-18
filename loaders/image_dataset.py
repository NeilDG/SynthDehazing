# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 18:14:21 2019

Pytorch image dataset
@author: delgallegon
"""
import torch
import cv2
import numpy as np
from torch.utils import data
from matplotlib import pyplot as plt
from ast import literal_eval
import torchvision.transforms as transforms
import constants

class TorchImageDataset(data.Dataset):
    
    def __init__(self, rgb_image_list):
        self.rgb_image_list = rgb_image_list
        self.image_transform_op = transforms.Compose([
                                   transforms.ToPILImage(),
                                   transforms.Resize(constants.IMAGE_SIZE),
                                   transforms.CenterCrop(constants.IMAGE_SIZE),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ])
        
    def __getitem__(self, idx):
        img_id = self.rgb_image_list[idx]
        img = cv2.imread(img_id); img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #because matplot uses RGB, openCV is BGR
        
        if(self.image_transform_op):
            img = self.image_transform_op(img)
                
        path_segment = img_id.split("/")
        file_name = path_segment[len(path_segment) - 1]
        return file_name, img
    
    def __len__(self):
        return len(self.rgb_image_list)

