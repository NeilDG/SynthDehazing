# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 20:11:02 2020

@author: delgallegon
"""
import os
import torch
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch.utils import data
import torch.nn as nn
from utils import tensor_utils
from torchvision.utils import save_image
import torchvision.transforms as transforms
import constants

DATASET_DIV2K_PATH = "E:/DIV2K_train_HR/"
SAVE_PATH = "E:/VEMON_Transfer/train/C/"

class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
    
def assemble_img_list(path, num_image_to_load = -1):
    img_list = []
    
    for (root, dirs, files) in os.walk(path):
        for f in files:
            file_name = os.path.join(root, f)
            img_list.append(file_name)
    
    return img_list


def create_div2k_data():
    div2k_data = assemble_img_list(DATASET_DIV2K_PATH)
    count = 0
    for k in range(len(div2k_data)):
        normal_img = cv2.imread(div2k_data[k])
        normal_img = cv2.cvtColor(normal_img, cv2.COLOR_BGR2RGB)
        
        transform_op_1 = transforms.Compose([
                                    transforms.ToPILImage(),
                                    transforms.TenCrop((512, 512))
                                    ])
        
        final_op = transforms.Compose([transforms.ToPILImage(), 
                                       transforms.RandomHorizontalFlip(),
                                       transforms.Resize((512, 512)),
                                       transforms.RandomCrop(constants.TEST_IMAGE_SIZE),
                                       transforms.ToTensor()])
        
        for i in range(89):
            file_name = SAVE_PATH + "div2k_patch_%d.png" % count
            final_img = final_op(normal_img)
            save_image(final_img, file_name)
            print("Saved: ", file_name)
            count = count + 1
            
  
def create_gta_noisy_data():
    NOISY_SAVE_PATH = "E:/Noisy GTA/noisy/"
    CLEAN_SAVE_PATH = "E:/Noisy GTA/clean/"
    gta_clean_data = assemble_img_list(constants.DATASET_GTA_PATH_2)
    count = 0
    for k in range(len(gta_clean_data)):
        normal_img = cv2.imread(gta_clean_data[k])
        normal_img = cv2.cvtColor(normal_img, cv2.COLOR_BGR2RGB)
        transform_op = transforms.Compose([transforms.ToPILImage(),
                                           transforms.ColorJitter(brightness=(1.25, 1.8)),
                                           transforms.ToTensor(),
                                           AddGaussianNoise(std=0.15)]
                                          )
        
        final_img = transform_op(normal_img)
        file_name = NOISY_SAVE_PATH + "gta_noisy_%d.png" % count
        save_image(final_img, file_name)
        print("Saved: ", file_name)
        
        file_name = CLEAN_SAVE_PATH + "gta_clean_%d.png" % count
        save_image(transforms.ToTensor()(normal_img), file_name)
        
        count = count + 1

def main():
    create_gta_noisy_data()
    #create_div2k_data()
        
if __name__=="__main__": 
    main()   