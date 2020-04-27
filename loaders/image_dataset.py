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
    
    def __init__(self, normal_list, topdown_list):
        self.normal_list = normal_list
        self.topdown_list = topdown_list
        
        self.normal_transform_op = transforms.Compose([
                                   transforms.ToPILImage(),
                                   transforms.Resize(constants.NORMAL_IMAGE_SIZE),
                                   transforms.CenterCrop(constants.NORMAL_IMAGE_SIZE),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ])
        
        self.topdown_transform_op = transforms.Compose([
                                   transforms.ToPILImage(),
                                   transforms.Resize(constants.TOPDOWN_IMAGE_SIZE),
                                   transforms.CenterCrop(constants.TOPDOWN_IMAGE_SIZE),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ])
        
    def __getitem__(self, idx):
        img_id = self.normal_list[idx]
        normal_img = cv2.imread(img_id); normal_img = cv2.cvtColor(normal_img, cv2.COLOR_BGR2RGB) #because matplot uses RGB, openCV is BGR
        
        if(self.normal_transform_op):
            normal_img = self.normal_transform_op(normal_img)
        
        img_id = self.topdown_list[idx]
        topdown_img = cv2.imread(img_id); topdown_img = cv2.cvtColor(topdown_img, cv2.COLOR_BGR2RGB)
        
        if(self.topdown_transform_op):
            topdown_img = self.topdown_transform_op(topdown_img)
            
        path_segment = img_id.split("/")
        file_name = path_segment[len(path_segment) - 1]
        return file_name, normal_img, topdown_img
    
    def __len__(self):
        return len(self.normal_list)

