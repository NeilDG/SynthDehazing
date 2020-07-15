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
from utils import tensor_utils
from torchvision.utils import save_image
import torchvision.transforms as transforms
import constants

DATASET_DIV2K_PATH = "E:/DIV2K_train_HR/"
SAVE_PATH = "E:/VEMON_Transfer/train/C/"

class Div2kDataset(data.Dataset):
    def __init__(self, img_list,patchSize = constants.BIRD_IMAGE_SIZE):
        self.img_list = img_list
        
        #resized = (int(constants.TEST_IMAGE_SIZE[0] * 1.3), int(constants.TEST_IMAGE_SIZE[1] * 1.3))
        
        self.transform_op = transforms.Compose([
                                    transforms.ToPILImage(),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor()
                                    ])

    def __getitem__(self, idx):
        img_id = self.img_list[idx]
        path_segment = img_id.split("/")
        file_name = path_segment[len(path_segment) - 1]
        
        print(file_name)
        normal_img = cv2.imread(img_id); normal_img = cv2.cvtColor(normal_img, cv2.COLOR_BGR2RGB) #because matplot uses RGB, openCV is BGR
        
        if(self.transform_op):
            normal_img = self.transform_op(normal_img)
        return file_name, normal_img
    
    def __len__(self):
        return len(self.img_list)

def assemble_div2k_data(num_image_to_load = -1):
    div2k_list = []
    
    for (root, dirs, files) in os.walk(DATASET_DIV2K_PATH):
        for f in files:
            file_name = os.path.join(root, f)
            div2k_list.append(file_name)
    
    return div2k_list


def main():
    div2k_data = assemble_div2k_data()
    count = 0;
    for k in range(len(div2k_data)):
        normal_img = cv2.imread(div2k_data[k])
        normal_img = cv2.cvtColor(normal_img, cv2.COLOR_BGR2RGB)
        
        transform_op_1 = transforms.Compose([
                                    transforms.ToPILImage(),
                                    transforms.TenCrop((512, 512))
                                    ])
        
        final_op = transforms.Compose([transforms.ToPILImage(), 
                                       transforms.RandomHorizontalFlip(),
                                       transforms.RandomCrop(constants.TEST_IMAGE_SIZE),
                                       transforms.ToTensor(),
])
        
        for i in range(89):
            file_name = SAVE_PATH + "div2k_patch_%d.png" % count
            final_img = final_op(normal_img)
            save_image(final_img, file_name)
            print("Saved: ", file_name)
            count = count + 1
            
        # normal_batch = transform_op_1(final_op)
        # for i in range(len(normal_batch)):
        #     for j in range(50):
        #         file_name = SAVE_PATH + "div2k_patch_%d.png" % count
        #         final_img = final_op(normal_batch[i])
        #         save_image(final_img, file_name)
        #         print("Saved: ", file_name)
        #         count = count + 1
        
            

if __name__=="__main__": 
    main()   