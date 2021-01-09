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

#model-based transmission dataset. Only accepts the clear RGB image and depth image.
class TransmissionDataset_Single(data.Dataset):
    def __init__(self, image_list_a, image_list_b):
        self.image_list_a = image_list_a
        self.image_list_b = image_list_b

        self.initial_img_op = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256))
        ])

        self.final_transform_op = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ])

        self.depth_transform_op = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ])

    def __getitem__(self, idx):
        img_id = self.image_list_a[idx]
        path_segment = img_id.split("/")
        file_name = path_segment[len(path_segment) - 1]

        clear_img = cv2.imread(img_id);
        clear_img = cv2.cvtColor(clear_img, cv2.COLOR_BGR2RGB)  # because matplot uses RGB, openCV is BGR
        clear_img = cv2.normalize(clear_img, dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        img_id = self.image_list_b[idx]
        img_b = cv2.imread(img_id);
        img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
        img_b = cv2.normalize(img_b, dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        T = tensor_utils.generate_transmission(1 - img_b, np.random.uniform(0.4, 2.5))
        img_b = cv2.normalize(T, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        #formulate hazy img
        atmosphere = np.random.uniform(0.5, 1.2)
        hazy_img_like = np.zeros_like(clear_img)
        T = np.resize(T, np.shape(clear_img[:, :, 0]))
        hazy_img_like[:, :, 0] = (T * clear_img[:, :, 0]) + atmosphere * (1 - T)
        hazy_img_like[:, :, 1] = (T * clear_img[:, :, 1]) + atmosphere * (1 - T)
        hazy_img_like[:, :, 2] = (T * clear_img[:, :, 2]) + atmosphere * (1 - T)

        img_a = cv2.normalize(hazy_img_like, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        img_a = self.initial_img_op(img_a)
        img_b = self.initial_img_op(img_b)

        img_a = self.final_transform_op(img_a)
        img_b = self.depth_transform_op(img_b)

        return file_name, img_a, img_b #hazy img, transmission map

    def __len__(self):
        return len(self.image_list_a)

class TransmissionDataset(data.Dataset):
    def __init__(self, image_list_a, image_list_b):
        self.image_list_a = image_list_a
        self.image_list_b = image_list_b

        self.initial_img_op = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256))
        ])

        self.final_transform_op = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ])

        self.depth_transform_op = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ])

    def __getitem__(self, idx):
        img_id = self.image_list_a[idx]
        path_segment = img_id.split("/")
        file_name = path_segment[len(path_segment) - 1]

        img_a = cv2.imread(img_id);
        #img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)  # because matplot uses RGB, openCV is BGR
        img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB)  # because matplot uses RGB, openCV is BGR

        img_id = self.image_list_b[idx]
        img_b = cv2.imread(img_id);
        img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
        img_b = cv2.normalize(img_b, dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        img_b = tensor_utils.generate_transmission(img_b, np.random.uniform(0.4, 1.8))
        img_b = cv2.normalize(img_b, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        #img_b = cv2.resize(img_b, (256, 256))

        img_a = self.initial_img_op(img_a)
        img_b = self.initial_img_op(img_b)

        # crop_indices = transforms.RandomCrop.get_params(img_a, output_size=(128, 128))
        # i, j, h, w = crop_indices
        #
        # img_a = transforms.functional.crop(img_a, i, j, h, w)
        # img_b = transforms.functional.crop(img_b, i, j, h, w)

        img_a = self.final_transform_op(img_a)
        img_b = self.depth_transform_op(img_b)

        return file_name, img_a, img_b

    def __len__(self):
        return len(self.image_list_a)

class TransmissionTestDataset(data.Dataset):
    def __init__(self, img_list_a):
        self.img_list_a = img_list_a

        self.final_transform_op = transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.depth_transform_op = transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5)),
        ])

    def __getitem__(self, idx):
        img_id = self.img_list_a[idx]
        path_segment = img_id.split("/")
        file_name = path_segment[len(path_segment) - 1]

        img_a = cv2.imread(img_id);
        img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB)  # because matplot uses RGB, openCV is BGR
        #img_size = np.shape(img_a)
        #img_a = cv2.resize(img_a, (int(img_size[1] / 4), int(img_size[0] / 4)))
        #img_a = cv2.resize(img_a, (512, 512))

        gray_img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)

        img_a = self.final_transform_op(img_a)
        gray_img_a = self.depth_transform_op(gray_img_a)

        return file_name, img_a, gray_img_a

    def __len__(self):
        return len(self.img_list_a)

class DepthDataset(data.Dataset):
    def __init__(self, image_list_a, image_list_b):
        self.image_list_a = image_list_a
        self.image_list_b = image_list_b

        self.final_transform_op = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.depth_transform_op = transforms.Compose([
            transforms.ToPILImage('L'),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ])

    def __getitem__(self, idx):
        img_id = self.image_list_a[idx]
        path_segment = img_id.split("/")
        file_name = path_segment[len(path_segment) - 1]

        img_a = cv2.imread(img_id);
        img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB)  # because matplot uses RGB, openCV is BGR

        img_id = self.image_list_b[idx]
        img_b = cv2.imread(img_id);
        img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)

        img_a = self.final_transform_op(img_a)
        img_b = self.depth_transform_op(img_b)

        return file_name, img_a, img_b

    def __len__(self):
        return len(self.image_list_a)



class DepthTestDataset(data.Dataset):
    def __init__(self, img_list_a, img_list_b):
        self.img_list_a = img_list_a
        self.img_list_b = img_list_b

        self.final_transform_op = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(constants.TEST_IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.depth_transform_op = transforms.Compose([
            transforms.ToPILImage('L'),
            transforms.Resize(constants.TEST_IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ])

    def __getitem__(self, idx):
        img_id = self.img_list_a[idx]
        path_segment = img_id.split("/")
        file_name = path_segment[len(path_segment) - 1]

        img_a = cv2.imread(img_id);
        img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB)  # because matplot uses RGB, openCV is BGR

        img_id = self.img_list_b[idx]
        img_b = cv2.imread(img_id);
        img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)

        img_a = self.final_transform_op(img_a)
        img_b = self.depth_transform_op(img_b)

        return file_name, img_a, img_b

    def __len__(self):
        return len(self.img_list_a)

class ColorTransferDataset(data.Dataset):
    def __init__(self, image_list_a, image_list_b):
        self.image_list_a = image_list_a
        self.image_list_b = image_list_b
        
        self.final_transform_op = transforms.Compose([
                                    transforms.ToPILImage(),
                                    transforms.Resize((256, 256)),
                                    transforms.RandomCrop(constants.PATCH_IMAGE_SIZE),
                                    transforms.RandomVerticalFlip(),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                    ])
        
        
    
    def __getitem__(self, idx):
        img_id = self.image_list_a[idx]
        path_segment = img_id.split("/")
        file_name = path_segment[len(path_segment) - 1]
        
        img_a = cv2.imread(img_id); img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB) #because matplot uses RGB, openCV is BGR
        
        img_id = self.image_list_b[idx]
        img_b = cv2.imread(img_id); img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB)
        
        img_a = self.final_transform_op(img_a)
        img_b = self.final_transform_op(img_b)
            
        return file_name, img_a, img_b
    
    def __len__(self):
        return len(self.image_list_a)


class ColorTransferTestDataset(data.Dataset):
    def __init__(self, img_list_a, img_list_b):
        self.img_list_a = img_list_a
        self.img_list_b = img_list_b

        self.final_transform_op = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(constants.TEST_IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, idx):
        img_id = self.img_list_a[idx]
        path_segment = img_id.split("/")
        file_name = path_segment[len(path_segment) - 1]

        img_a = cv2.imread(img_id);
        img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB)  # because matplot uses RGB, openCV is BGR

        img_id = self.img_list_b[idx]
        img_b = cv2.imread(img_id);
        img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB)

        img_a = self.final_transform_op(img_a)
        img_b = self.final_transform_op(img_b)

        return file_name, img_a, img_b

    def __len__(self):
        return len(self.img_list_a)

class DarkChannelHazeDataset(data.Dataset):
    def __init__(self, hazy_list, clear_list):
        self.hazy_list = hazy_list
        self.clear_list = clear_list
        self.transform_op = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5), (0.5))])

        # self.initial_transform_op = transforms.Compose([
        #                             transforms.ToPILImage(mode = 'L'),
        #                             transforms.Resize(constants.TEST_IMAGE_SIZE),
        #                             ])
        #
        # self.final_transform_op = transforms.Compose([transforms.ToTensor(),
        #                                               transforms.Normalize((0.5), (0.5))])
        
        
    
    def __getitem__(self, idx):
        img_id = self.hazy_list[idx]
        path_segment = img_id.split("/")
        file_name = path_segment[len(path_segment) - 1]
        
        hazy_img = cv2.imread(img_id); hazy_img = cv2.cvtColor(hazy_img, cv2.COLOR_BGR2YUV)
        hazy_img = tensor_utils.get_y_channel(hazy_img)

        #hazy_img = cv2.imread(img_id); hazy_img = cv2.cvtColor(hazy_img, cv2.COLOR_BGR2RGB)
        #hazy_img = tensor_utils.get_dark_channel(hazy_img, 8)
        
        img_id = self.clear_list[idx]
        #clear_img = cv2.imread(img_id); clear_img = cv2.cvtColor(clear_img, cv2.COLOR_BGR2RGB)
        #clear_img = tensor_utils.get_dark_channel(clear_img, 8)
        clear_img = cv2.imread(img_id); clear_img = cv2.cvtColor(clear_img, cv2.COLOR_BGR2YUV)
        clear_img = tensor_utils.get_y_channel(clear_img)
                 
        # hazy_img = self.initial_transform_op(hazy_img)
        # clear_img = self.initial_transform_op(clear_img)
        #
        # crop_indices = transforms.RandomCrop.get_params(hazy_img, output_size=constants.PATCH_IMAGE_SIZE)
        # i, j, h, w = crop_indices
        #
        # hazy_img = transforms.functional.crop(hazy_img, i, j, h, w)
        # clear_img = transforms.functional.crop(clear_img, i, j, h, w)
        #
        # hazy_img = self.final_transform_op(hazy_img)
        # clear_img = self.final_transform_op(clear_img)
        hazy_img = self.transform_op(hazy_img)
        clear_img = self.transform_op(clear_img)

        return file_name, hazy_img, clear_img
    
    def __len__(self):
        return len(self.hazy_list)


class DarkChannelTestDataset(data.Dataset):
    def __init__(self, hazy_list, clear_list):
        self.hazy_list = hazy_list
        self.clear_list = clear_list

        self.transform_op = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(constants.TEST_IMAGE_SIZE),
            transforms.CenterCrop(constants.TEST_IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ])

    def __getitem__(self, idx):
        img_id = self.hazy_list[idx]
        path_segment = img_id.split("/")
        file_name = path_segment[len(path_segment) - 1]

        hazy_img = cv2.imread(img_id);
        hazy_img = cv2.cvtColor(hazy_img, cv2.COLOR_BGR2YUV)
        hazy_img = tensor_utils.get_y_channel(hazy_img)
        #hazy_img = tensor_utils.get_dark_channel(hazy_img, constants.DC_FILTER_SIZE)

        img_id = self.clear_list[idx]
        clear_img = cv2.imread(img_id);
        clear_img = cv2.cvtColor(clear_img, cv2.COLOR_BGR2YUV)
        clear_img = tensor_utils.get_y_channel(clear_img)
        #clear_img = tensor_utils.get_dark_channel(clear_img, constants.DC_FILTER_SIZE)

        if (self.transform_op):
            hazy_img = self.transform_op(hazy_img)
            clear_img = self.transform_op(clear_img)
        return file_name, hazy_img, clear_img

    def __len__(self):
        return len(self.hazy_list)

class HazeDataset(data.Dataset):
    def __init__(self, hazy_list, clear_list):
        self.hazy_list = hazy_list
        self.clear_list = clear_list
        
        self.initial_transform_op = transforms.Compose([
                                    transforms.ToPILImage(),
                                    transforms.Resize(constants.TEST_IMAGE_SIZE),
                                    ])
            
        self.final_transform_op = transforms.Compose([transforms.ToTensor(),
                                                      transforms.Normalize((0.5), (0.5), (0.5))])
        
        
    
    def __getitem__(self, idx):
        img_id = self.hazy_list[idx]
        path_segment = img_id.split("/")
        file_name = path_segment[len(path_segment) - 1]
        
        hazy_img = cv2.imread(img_id); hazy_img = cv2.cvtColor(hazy_img, cv2.COLOR_BGR2RGB)
        
        img_id = self.clear_list[idx]
        clear_img = cv2.imread(img_id); clear_img = cv2.cvtColor(clear_img, cv2.COLOR_BGR2RGB)

        #img_id = self.real_hazy_list[idx]
        #real_hazy_img = cv2.imread(img_id); real_hazy_img = cv2.cvtColor(real_hazy_img, cv2.COLOR_BGR2RGB)
                 
        hazy_img = self.initial_transform_op(hazy_img)
        clear_img = self.initial_transform_op(clear_img)
        #real_hazy_img  = self.initial_transform_op(real_hazy_img)
        
        crop_indices = transforms.RandomCrop.get_params(hazy_img, output_size=constants.PATCH_IMAGE_SIZE)
        i, j, h, w = crop_indices
        
        hazy_img = transforms.functional.crop(hazy_img, i, j, h, w)
        clear_img = transforms.functional.crop(clear_img, i, j, h, w)
        #real_hazy_img = transforms.functional.crop(real_hazy_img, i, j, h, w)

        hazy_img = transforms.functional.adjust_brightness(hazy_img, constants.brightness_enhance)
        clear_img = transforms.functional.adjust_brightness(clear_img, constants.brightness_enhance)
        #real_hazy_img = transforms.functional.adjust_brightness(real_hazy_img, constants.brightness_enhance)

        hazy_img = transforms.functional.adjust_contrast(hazy_img, constants.contrast_enhance)
        hazy_img = transforms.functional.adjust_contrast(hazy_img, constants.contrast_enhance)
        #real_hazy_img = transforms.functional.adjust_brightness(real_hazy_img, constants.brightness_enhance)
        
        hazy_img = self.final_transform_op(hazy_img)
        clear_img = self.final_transform_op(clear_img)
        #real_hazy_img = self.final_transform_op(real_hazy_img)
                
        return file_name, hazy_img, clear_img
    
    def __len__(self):
        return len(self.hazy_list)

class HazeTestDataset(data.Dataset):
    def __init__(self, rgb_list):
        self.rgb_list = rgb_list
        
        self.initial_transform_op = transforms.Compose([
                                    transforms.ToPILImage(),
                                    transforms.Resize(constants.TEST_IMAGE_SIZE),
                                    transforms.CenterCrop(constants.TEST_IMAGE_SIZE)
                                    ])
        
        self.final_transform_op = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5), (0.5))
                                    ])
    def __getitem__(self, idx):
        img_id = self.rgb_list[idx]
        path_segment = img_id.split("/")
        file_name = path_segment[len(path_segment) - 1]
        
        img = cv2.imread(img_id); img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.initial_transform_op(img)
        img = self.final_transform_op(img)
        
        return file_name, img
    
    def __len__(self):
        return len(self.rgb_list)
    
class ColorDataset(data.Dataset):
    def __init__(self, rgb_list_a):
        self.rgb_list = rgb_list_a
        
        self.rgb_transform_op = transforms.Compose([
                                    transforms.ToPILImage(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5), (0.5), (0.5))
                                    ])  
        
        self.gray_transform_op = transforms.Compose([
                                    transforms.ToPILImage(mode= 'L'),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5), (0.5), (0.5))
                                    ])
            
        # self.final_transform_op = transforms.Compose([transforms.ToTensor(),
        #                                               transforms.Normalize((0.5), (0.5))])
        
    
    def __getitem__(self, idx):
        img_id = self.rgb_list[idx]
        path_segment = img_id.split("/")
        file_name = path_segment[len(path_segment) - 1]
        
        img_a = cv2.imread(img_id); img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2YUV)
        gray_img_a = tensor_utils.get_y_channel(img_a)
        gray_img_a = self.gray_transform_op(gray_img_a)
        colored_img_a = self.rgb_transform_op(img_a)
        
        return file_name, gray_img_a, colored_img_a
    
    def __len__(self):
        return len(self.rgb_list)

class ColorTestDataset(data.Dataset):
    def __init__(self, rgb_list_a):
        self.rgb_list = rgb_list_a
        
        self.transform_op = transforms.Compose([
                                    transforms.ToPILImage(),
                                    transforms.Resize(constants.TEST_IMAGE_SIZE),
                                    transforms.RandomCrop(constants.TEST_IMAGE_SIZE),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5), (0.5), (0.5))
                                    ])
    def __getitem__(self, idx):
        img_id = self.rgb_list[idx]
        path_segment = img_id.split("/")
        file_name = path_segment[len(path_segment) - 1]

        img_a = cv2.imread(img_id); img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB)
        gray_img_a = tensor_utils.get_y_channel(img_a)
        gray_img_a = self.transform_op(gray_img_a)
        colored_img_a = self.transform_op(img_a)
        
        return file_name, gray_img_a, colored_img_a
    
    def __len__(self):
        return len(self.rgb_list)


class LatentDataset(data.Dataset):
    def __init__(self, image_list_a):
        self.image_list_a = image_list_a

        self.final_transform_op = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, idx):
        img_id = self.image_list_a[idx]
        path_segment = img_id.split("/")
        file_name = path_segment[len(path_segment) - 1]

        img_a = cv2.imread(img_id);
        img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB)  # because matplot uses RGB, openCV is BGR

        img_a = self.final_transform_op(img_a)

        return file_name, img_a

    def __len__(self):
        return len(self.image_list_a)