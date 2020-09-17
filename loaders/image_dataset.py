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
from utils import tensor_utils


class GlobalConfig():
    def __init__(self):
        self.i = 0
        self.j = 0
        self.h = 0
        self.w = 0
    
    def set_vars(i, j, h, w):
        self.i = i
        self.j = j
        self.h = h
        self.w = w
    
    def get_vars():
        return self.i, self.j, self.h, self.w

class Div2kDataset(data.Dataset):
    def __init__(self, vemon_list, div2k_list):
        self.vemon_list = vemon_list
        self.div2k_list = div2k_list
        
        resized = (int(constants.TEST_IMAGE_SIZE[0] * 1.01), int(constants.TEST_IMAGE_SIZE[1] * 1.01))
        self.transform_op = transforms.Compose([
                                    transforms.ToPILImage(),
                                    #transforms.Resize(resized),
                                    transforms.RandomCrop(constants.BIRD_IMAGE_SIZE),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                    ])
        
        
    
    def __getitem__(self, idx):
        img_id = self.vemon_list[idx]
        path_segment = img_id.split("/")
        file_name = path_segment[len(path_segment) - 1]
        
        normal_img = cv2.imread(img_id); normal_img = cv2.cvtColor(normal_img, cv2.COLOR_BGR2RGB) #because matplot uses RGB, openCV is BGR
        
        img_id = self.div2k_list[idx]
        topdown_img = cv2.imread(img_id); topdown_img = cv2.cvtColor(topdown_img, cv2.COLOR_BGR2RGB)
        
        if(self.transform_op):            
            normal_img = self.transform_op(normal_img)
            topdown_img = self.transform_op(topdown_img)
            
        return file_name, normal_img, topdown_img
    
    def __len__(self):
        return len(self.vemon_list)
    
class HazeDataset(data.Dataset):
    def __init__(self, hazy_list, clear_list):
        self.hazy_list = hazy_list
        self.clear_list = clear_list
        
        self.initial_transform_op = transforms.Compose([
                                    transforms.ToPILImage(mode= 'L'),
                                    transforms.Resize(constants.TEST_IMAGE_SIZE),
                                    ])
            
        self.final_transform_op = transforms.Compose([transforms.ToTensor(),
                                                      transforms.Normalize((0.5), (0.5))])
        
        
    
    def __getitem__(self, idx):
        img_id = self.hazy_list[idx]
        path_segment = img_id.split("/")
        file_name = path_segment[len(path_segment) - 1]
        
        hazy_img = cv2.imread(img_id); hazy_img = cv2.cvtColor(hazy_img, cv2.COLOR_BGR2YUV)
        hazy_img = tensor_utils.get_y_channel(hazy_img)
        
        img_id = self.clear_list[idx]
        clear_img = cv2.imread(img_id); clear_img = cv2.cvtColor(clear_img, cv2.COLOR_BGR2YUV)
        clear_img = tensor_utils.get_y_channel(clear_img)
                 
        hazy_img = self.initial_transform_op(hazy_img)
        clear_img = self.initial_transform_op(clear_img)
        
        crop_indices = transforms.RandomCrop.get_params(hazy_img, output_size=constants.BIRD_IMAGE_SIZE)
        i, j, h, w = crop_indices
        
        hazy_img = transforms.functional.crop(hazy_img, i, j, h, w)
        clear_img = transforms.functional.crop(clear_img, i, j, h, w)
        
        hazy_img = self.final_transform_op(hazy_img)
        clear_img = self.final_transform_op(clear_img)
            
                
        return file_name, hazy_img, clear_img
    
    def __len__(self):
        return len(self.hazy_list)

class ColorDataset(data.Dataset):
    def __init__(self, rgb_list):
        self.rgb_list = rgb_list
        
        self.rgb_transform_op = transforms.Compose([
                                    transforms.ToPILImage(),
                                    transforms.Resize(constants.TEST_IMAGE_SIZE)
                                    ])  
        
        self.gray_transform_op = transforms.Compose([
                                    transforms.ToPILImage(mode= 'L'),
                                    transforms.Resize(constants.TEST_IMAGE_SIZE)
                                    ])
            
        self.final_transform_op = transforms.Compose([transforms.ToTensor(),
                                                      transforms.Normalize((0.5), (0.5))])
        
    
    def __getitem__(self, idx):
        img_id = self.rgb_list[idx]
        path_segment = img_id.split("/")
        file_name = path_segment[len(path_segment) - 1]
        
        img = cv2.imread(img_id); img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        gray_img = tensor_utils.get_y_channel(img)
        #yuv_img = tensor_utils.get_uv_channel(img)
        
        gray_img = self.gray_transform_op(gray_img)
        
        crop_indices = transforms.RandomCrop.get_params(gray_img, output_size=constants.BIRD_IMAGE_SIZE)
        i, j, h, w = crop_indices
        
        yuv_img = self.rgb_transform_op(img)
        
        gray_img = transforms.functional.crop(gray_img, i, j, h, w)
        yuv_img = transforms.functional.crop(yuv_img, i, j, h, w)
        
        gray_img = self.final_transform_op(gray_img)
        yuv_img = self.final_transform_op(yuv_img)
        
        return file_name, gray_img, yuv_img
    
    def __len__(self):
        return len(self.rgb_list)

class TestDataset(data.Dataset):
    def __init__(self, vemon_list, gta_list):
        self.vemon_list = vemon_list
        self.gta_list = gta_list
        
        self.transform_op = transforms.Compose([
                                    transforms.ToPILImage(),
                                    transforms.Resize(constants.TEST_IMAGE_SIZE),
                                    transforms.CenterCrop(constants.TEST_IMAGE_SIZE),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                    ])
        
        
    
    def __getitem__(self, idx):
        img_id = self.vemon_list[idx]
        path_segment = img_id.split("/")
        file_name = path_segment[len(path_segment) - 1]
        
        normal_img = cv2.imread(img_id); normal_img = cv2.cvtColor(normal_img, cv2.COLOR_BGR2RGB)
        
        img_id = self.gta_list[idx]
        topdown_img = cv2.imread(img_id); topdown_img = cv2.cvtColor(topdown_img, cv2.COLOR_BGR2RGB)
        
        if(self.transform_op):
            normal_img = self.transform_op(normal_img)
            topdown_img = self.transform_op(topdown_img)
        return file_name, normal_img, topdown_img
    
    def __len__(self):
        return len(self.vemon_list)

class ColorTestDataset(data.Dataset):
    def __init__(self, rgb_list):
        self.rgb_list = rgb_list
        
        self.transform_op = transforms.Compose([
                                    transforms.ToPILImage(),
                                    transforms.Resize(constants.TEST_IMAGE_SIZE),
                                    transforms.CenterCrop(constants.TEST_IMAGE_SIZE),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5), (0.5))
                                    ])
    def __getitem__(self, idx):
        img_id = self.rgb_list[idx]
        path_segment = img_id.split("/")
        file_name = path_segment[len(path_segment) - 1]
        
        img = cv2.imread(img_id); img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        gray_img = tensor_utils.get_y_channel(img)
        #uv_img = tensor_utils.get_uv_channel(img)
        
        gray_img = self.transform_op(gray_img)
        yuv_img = self.transform_op(img)
        
        return file_name, gray_img, yuv_img
    
    def __len__(self):
        return len(self.rgb_list)
    
class DarkChannelTestDataset(data.Dataset):
    def __init__(self, vemon_list, gta_list):
        self.vemon_list = vemon_list
        self.gta_list = gta_list
        
        self.transform_op = transforms.Compose([
                                    transforms.ToPILImage(),
                                    transforms.Resize(constants.TEST_IMAGE_SIZE),
                                    transforms.CenterCrop(constants.TEST_IMAGE_SIZE),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5), (0.5))
                                    ])
        
        
    
    def __getitem__(self, idx):
        img_id = self.vemon_list[idx]
        path_segment = img_id.split("/")
        file_name = path_segment[len(path_segment) - 1]
        
        normal_img = cv2.imread(img_id); normal_img = cv2.cvtColor(normal_img, cv2.COLOR_BGR2YUV)
        normal_img = tensor_utils.get_y_channel(normal_img)
        
        img_id = self.gta_list[idx]
        topdown_img = cv2.imread(img_id); topdown_img = cv2.cvtColor(topdown_img, cv2.COLOR_BGR2YUV)
        topdown_img = tensor_utils.get_y_channel(topdown_img)
        
        if(self.transform_op):
            normal_img = self.transform_op(normal_img)
            topdown_img = self.transform_op(topdown_img)
        return file_name, normal_img, topdown_img
    
    def __len__(self):
        return len(self.vemon_list)