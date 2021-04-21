# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 18:14:21 2019

Pytorch image dataset
@author: delgallegon
"""
import math

import torch
import cv2
import numpy as np
from torch.utils import data
from matplotlib import pyplot as plt
from ast import literal_eval
import torchvision.transforms as transforms
import constants
from utils import tensor_utils

class TransmissionAlbedoDataset(data.Dataset):
    def __init__(self, image_list_a, depth_dir, crop_size, should_crop):
        self.image_list_a = image_list_a
        self.depth_dir = depth_dir
        self.crop_size = crop_size
        self.should_crop = should_crop

        self.initial_img_op = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256))
        ])

        self.final_transform_op = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
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

        img_id = self.depth_dir + file_name
        img_b = cv2.imread(img_id)
        img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
        img_b = cv2.resize(img_b, np.shape(clear_img[:, :, 0]))
        img_b = cv2.normalize(img_b, dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        T = tensor_utils.generate_transmission(1 - img_b, np.random.uniform(0.0, 2.5)) #also include clear samples
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

        if(self.should_crop):
            crop_indices = transforms.RandomCrop.get_params(img_a, output_size=self.crop_size)
            i, j, h, w = crop_indices

            img_a = transforms.functional.crop(img_a, i, j, h, w)
            img_b = transforms.functional.crop(img_b, i, j, h, w)

        img_a = self.final_transform_op(img_a)
        img_b = self.depth_transform_op(img_b)

        airlight_tensor = torch.tensor(atmosphere, dtype=torch.float32)
        return file_name, img_a, img_b #hazy albedo img, transmission map

    def __len__(self):
        return len(self.image_list_a)

class TransmissionAlbedoDatasetTest(data.Dataset):
    def __init__(self, image_list_a):
        self.image_list_a = image_list_a

        self.initial_img_op = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256))
        ])

        self.final_transform_op = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, idx):
        img_id = self.image_list_a[idx]
        path_segment = img_id.split("/")
        file_name = path_segment[len(path_segment) - 1]

        img_a = cv2.imread(img_id);
        img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB)  # because matplot uses RGB, openCV is BGR
        img_a = cv2.normalize(img_a, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        img_a = self.initial_img_op(img_a)
        img_a = self.final_transform_op(img_a)

        return file_name, img_a

    def __len__(self):
        return len(self.image_list_a)

class AirlightDataset(data.Dataset):
    ATMOSPHERE_MIN = 0.5
    ATMOSPHERE_MAX = 1.2

    @staticmethod
    def atmosphere_mean():
        return (AirlightDataset.ATMOSPHERE_MIN + AirlightDataset.ATMOSPHERE_MAX) / 2.0;

    @staticmethod
    def atmosphere_std():
        return math.sqrt(pow(AirlightDataset.ATMOSPHERE_MAX - AirlightDataset.ATMOSPHERE_MIN, 2) / 12)

    def __init__(self, image_albedo_list, clear_dir, depth_dir, crop_size, should_crop):
        self.image_list_a = image_albedo_list
        self.depth_dir = depth_dir
        self.clear_dir = clear_dir
        self.crop_size = crop_size
        self.should_crop = should_crop

        self.initial_img_op = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256))
        ])

        self.final_transform_op = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, idx):
        img_id = self.image_list_a[idx]
        path_segment = img_id.split("/")
        file_name = path_segment[len(path_segment) - 1]

        #albedo hazy img
        albedo_img = cv2.imread(img_id);
        albedo_img = cv2.cvtColor(albedo_img, cv2.COLOR_BGR2RGB)  # because matplot uses RGB, openCV is BGR
        albedo_img = cv2.normalize(albedo_img, dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        img_id = self.depth_dir + file_name
        depth_img = cv2.imread(img_id)
        depth_img = cv2.cvtColor(depth_img, cv2.COLOR_BGR2GRAY)
        depth_img = cv2.resize(depth_img, np.shape(albedo_img[:, :, 0]))
        depth_img = cv2.normalize(depth_img, dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        T = tensor_utils.generate_transmission(1 - depth_img, np.random.uniform(0.0, 2.5)) #also include clear samples

        #formulate hazy img
        #atmosphere = np.random.uniform(AirlightDataset.ATMOSPHERE_MIN, AirlightDataset.ATMOSPHERE_MAX)
        atmosphere = np.random.normal(AirlightDataset.atmosphere_mean(), AirlightDataset.atmosphere_std() + 1.5)
        T = np.resize(T, np.shape(albedo_img[:, :, 0]))
        albedo_hazy_img = np.zeros_like(albedo_img)
        albedo_hazy_img[:, :, 0] = (T * albedo_img[:, :, 0]) + atmosphere * (1 - T)
        albedo_hazy_img[:, :, 1] = (T * albedo_img[:, :, 1]) + atmosphere * (1 - T)
        albedo_hazy_img[:, :, 2] = (T * albedo_img[:, :, 2]) + atmosphere * (1 - T)
        albedo_hazy_img = cv2.normalize(albedo_hazy_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        #styled hazy img
        img_id = self.clear_dir + file_name
        styled_img = cv2.imread(img_id)
        styled_img = cv2.cvtColor(styled_img, cv2.COLOR_BGR2RGB)
        styled_img = cv2.resize(styled_img, np.shape(albedo_hazy_img[:, :, 0]))
        styled_img = cv2.normalize(styled_img, dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        #formulate hazy img
        T = np.resize(T, np.shape(styled_img[:, :, 0]))
        styled_hazy_img = np.zeros_like(styled_img)
        styled_hazy_img[:, :, 0] = (T * styled_img[:, :, 0]) + atmosphere * (1 - T)
        styled_hazy_img[:, :, 1] = (T * styled_img[:, :, 1]) + atmosphere * (1 - T)
        styled_hazy_img[:, :, 2] = (T * styled_img[:, :, 2]) + atmosphere * (1 - T)
        styled_hazy_img = cv2.normalize(styled_hazy_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        albedo_hazy_img = self.initial_img_op(albedo_hazy_img)
        styled_hazy_img = self.initial_img_op(styled_hazy_img)

        if(self.should_crop):
            crop_indices = transforms.RandomCrop.get_params(styled_hazy_img, output_size=self.crop_size)
            i, j, h, w = crop_indices

            albedo_hazy_img = transforms.functional.crop(albedo_hazy_img, i, j, h, w)
            styled_hazy_img = transforms.functional.crop(styled_hazy_img, i, j, h, w)

        albedo_hazy_img = self.final_transform_op(albedo_hazy_img)
        styled_hazy_img = self.final_transform_op(styled_hazy_img)

        self.depth_transform_op = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        # transmission_img = cv2.normalize(T, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        # transmission_img = cv2.resize(transmission_img, (256, 256))
        # transmission_img = self.depth_transform_op(transmission_img)

        #normalize
        #atmosphere = (atmosphere - AirlightDataset.atmosphere_mean()) / AirlightDataset.atmosphere_std()
        airlight_tensor = torch.tensor(atmosphere, dtype = torch.float32)

        return file_name, albedo_hazy_img, styled_hazy_img, airlight_tensor #hazy albedo img, transmission map, airlight

    def __len__(self):
        return len(self.image_list_a)

#model-based transmission dataset. Only accepts the clear RGB image and depth image.
class TransmissionDataset_Single(data.Dataset):
    def __init__(self, image_list_a, image_list_b, light_list_c, crop_size):
        self.image_list_a = image_list_a
        self.image_list_b = image_list_b
        self.light_list_c = light_list_c
        self.crop_size = crop_size

        self.initial_img_op = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256))
        ])

        self.final_transform_op = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.depth_transform_op = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        self.light_vector_mean = []
        self.light_vector_std = []

        self.compute_mean_std()

    def compute_mean_std(self):
        light_x = []
        light_z = []
        print("Computing mean and std of light vectors")

        for i in range(0, len(self.light_list_c)):
            light_file = open(self.light_list_c[i], "r")
            light_string = light_file.readline()
            light_vector = str.split(light_string, ",")
            light_vector = [float(light_vector[0]), float(light_vector[1])]
            light_x.append(light_vector[0])
            light_z.append(light_vector[1])
            #print("Light vector ",i, " : ", light_vector[0])

        self.light_vector_mean.append(np.mean(light_x))
        self.light_vector_mean.append(np.mean(light_z))

        self.light_vector_std.append(np.std(light_x))
        self.light_vector_std.append(np.std(light_z))

        print("X mean: ", self.light_vector_mean[0], " std: ", self.light_vector_std[0])
        print("Z mean: ", self.light_vector_mean[1], " std: ", self.light_vector_std[1])

    def __getitem__(self, idx):
        img_id = self.image_list_a[idx]
        path_segment = img_id.split("/")
        file_name = path_segment[len(path_segment) - 1]

        clear_img = cv2.imread(img_id);
        clear_img = cv2.cvtColor(clear_img, cv2.COLOR_BGR2RGB)  # because matplot uses RGB, openCV is BGR
        clear_img = cv2.normalize(clear_img, dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        img_id = self.image_list_b[idx]
        img_b = cv2.imread(img_id)
        img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
        img_b = cv2.resize(img_b, np.shape(clear_img[:, :, 0]))
        img_b = cv2.normalize(img_b, dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        T = tensor_utils.generate_transmission(1 - img_b, np.random.uniform(0.0, 2.5)) #also include clear samples
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

        crop_indices = transforms.RandomCrop.get_params(img_a, output_size=self.crop_size)
        i, j, h, w = crop_indices

        img_a = transforms.functional.crop(img_a, i, j, h, w)
        img_b = transforms.functional.crop(img_b, i, j, h, w)

        img_a = self.final_transform_op(img_a)
        img_b = self.depth_transform_op(img_b)

        airlight_tensor = torch.tensor(atmosphere, dtype=torch.float32)

        if(constants.is_coare == 0):
            light_id = int(img_id.split("/")[3].split(".")[0].split("_")[1]) + 5  # offset
        else:
            light_id = int(img_id.split("/")[6].split(".")[0].split("_")[1]) + 5  # offset

        light_path = constants.DATASET_LIGHTCOORDS_PATH_COMPLETE + "lights_" + str(light_id) + ".txt"

        #light_file = open(self.light_list_c[idx], "r")
        light_file = open(light_path, "r")
        light_string = light_file.readline()
        light_vector = str.split(light_string, ",")
        light_vector = [float(light_vector[0]), float(light_vector[1])]

        #normalize data by mean and std
        light_vector[0] = (light_vector[0] - self.light_vector_mean[0]) / self.light_vector_std[0]
        light_vector[1] = (light_vector[1] - self.light_vector_mean[1]) / self.light_vector_std[1]

        light_coords_tensor = torch.tensor(light_vector, dtype = torch.float32)

        return file_name, img_a, img_b, light_coords_tensor, airlight_tensor #hazy img, transmission map, light distance, airlight

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
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
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

        img_a = self.initial_img_op(img_a)
        img_b = self.initial_img_op(img_b)

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

class HazeTestPairedDataset(data.Dataset):
    def __init__(self, hazy_list, clear_list):
        self.hazy_list = hazy_list
        self.clear_list = clear_list

        self.initial_transform_op = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(constants.TEST_IMAGE_SIZE),
            transforms.CenterCrop(constants.TEST_IMAGE_SIZE)
        ])

        self.final_transform_op = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, idx):
        img_id = self.hazy_list[idx]
        path_segment = img_id.split("/")
        file_name = path_segment[len(path_segment) - 1]

        hazy_img = cv2.imread(img_id);
        hazy_img = cv2.cvtColor(hazy_img, cv2.COLOR_BGR2RGB)
        hazy_img = self.initial_transform_op(hazy_img)
        hazy_img = self.final_transform_op(hazy_img)

        img_id = self.clear_list[idx]
        clear_img = cv2.imread(img_id);
        clear_img = cv2.cvtColor(clear_img, cv2.COLOR_BGR2RGB)
        clear_img = self.initial_transform_op(clear_img)
        clear_img = self.final_transform_op(clear_img)

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
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                    ])  
        
        self.gray_transform_op = transforms.Compose([
                                    transforms.ToPILImage(mode= 'L'),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5), (0.5), (0.5))
                                    ])
    
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


class ColorAlbedoDataset(data.Dataset):
    def __init__(self, image_list_a, image_list_b, depth_dir):
        self.image_list_a = image_list_a
        self.image_list_b = image_list_b
        self.depth_dir = depth_dir

        self.initial_transform_op = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(constants.TEST_IMAGE_SIZE),
            transforms.CenterCrop(constants.TEST_IMAGE_SIZE)
        ])

        self.final_transform_op = transforms.Compose([transforms.ToTensor(),
                                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    def __getitem__(self, idx):
        img_id = self.image_list_a[idx]
        path_segment = img_id.split("/")
        file_name = path_segment[len(path_segment) - 1]

        initial_img = cv2.imread(img_id);
        initial_img = cv2.cvtColor(initial_img, cv2.COLOR_BGR2RGB)  # because matplot uses RGB, openCV is BGR

        img_id = self.image_list_b[idx]
        img_b = cv2.imread(img_id);
        img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB)

        img_a = self.initial_transform_op(initial_img)
        img_b = self.initial_transform_op(img_b)

        img_id = self.depth_dir + file_name
        depth_img = cv2.imread(img_id)
        depth_img = cv2.cvtColor(depth_img, cv2.COLOR_BGR2GRAY)
        depth_img = cv2.resize(depth_img, np.shape(initial_img[:, :, 0]))
        depth_img = cv2.normalize(depth_img, dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        T = tensor_utils.generate_transmission(1 - depth_img, np.random.uniform(0.0, 2.5))  # also include clear samples

        # formulate hazy img
        atmosphere = np.random.normal(AirlightDataset.ATMOSPHERE_MIN, AirlightDataset.ATMOSPHERE_MAX)
        T = np.resize(T, np.shape(initial_img[:, :, 0]))
        hazy_img = initial_img
        hazy_img = cv2.normalize(hazy_img, dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        hazy_img[:, :, 0] = (T * hazy_img[:, :, 0]) + atmosphere * (1 - T)
        hazy_img[:, :, 1] = (T * hazy_img[:, :, 1]) + atmosphere * (1 - T)
        hazy_img[:, :, 2] = (T * hazy_img[:, :, 2]) + atmosphere * (1 - T)
        hazy_img = cv2.normalize(hazy_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        hazy_img = self.initial_transform_op(hazy_img)

        crop_indices = transforms.RandomCrop.get_params(img_a, output_size=constants.PATCH_IMAGE_SIZE)
        i, j, h, w = crop_indices

        #img_a = transforms.functional.crop(img_a, i, j, h, w)
        hazy_img = transforms.functional.crop(hazy_img, i, j, h, w)
        img_b = transforms.functional.crop(img_b, i, j, h, w)

        #img_a = self.final_transform_op(img_a)
        hazy_img = self.final_transform_op(hazy_img)
        img_b = self.final_transform_op(img_b)

        return file_name, hazy_img, img_b

    def __len__(self):
        return len(self.image_list_a)


class ColorAlbedoTestDataset(data.Dataset):
    def __init__(self, image_list_a, image_list_b, depth_dir): #depth_dir can be none
        self.image_list_a = image_list_a
        self.image_list_b = image_list_b
        self.depth_dir = depth_dir

        self.final_transform_op = transforms.Compose([transforms.ToPILImage(),
                                                      transforms.Resize(constants.TEST_IMAGE_SIZE),
                                                      transforms.CenterCrop(constants.TEST_IMAGE_SIZE),
                                                      transforms.ToTensor(),
                                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    def __getitem__(self, idx):
        img_id = self.image_list_a[idx]
        path_segment = img_id.split("/")
        file_name = path_segment[len(path_segment) - 1]

        img_a = cv2.imread(img_id);
        img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB)  # because matplot uses RGB, openCV is BGR

        img_id = self.image_list_b[idx]
        img_b = cv2.imread(img_id);
        img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB)

        if(self.depth_dir is not None):
            img_id = self.depth_dir + file_name
            depth_img = cv2.imread(img_id)
            depth_img = cv2.cvtColor(depth_img, cv2.COLOR_BGR2GRAY)
            depth_img = cv2.resize(depth_img, np.shape(img_a[:, :, 0]))
            depth_img = cv2.normalize(depth_img, dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            T = tensor_utils.generate_transmission(1 - depth_img, np.random.uniform(0.0, 2.5))  # also include clear samples

            # formulate hazy img
            atmosphere = np.random.normal(AirlightDataset.ATMOSPHERE_MIN, AirlightDataset.ATMOSPHERE_MAX)
            T = np.resize(T, np.shape(img_a[:, :, 0]))
            hazy_img = img_a
            hazy_img = cv2.normalize(hazy_img, dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            hazy_img[:, :, 0] = (T * hazy_img[:, :, 0]) + atmosphere * (1 - T)
            hazy_img[:, :, 1] = (T * hazy_img[:, :, 1]) + atmosphere * (1 - T)
            hazy_img[:, :, 2] = (T * hazy_img[:, :, 2]) + atmosphere * (1 - T)
            hazy_img = cv2.normalize(hazy_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        img_b = self.final_transform_op(img_b)

        if(self.depth_dir is None):
            img_a = self.final_transform_op(img_a)
            return file_name, img_a, img_b
        else:
            hazy_img = self.final_transform_op(hazy_img)
            return file_name, hazy_img, img_b

    def __len__(self):
        return len(self.image_list_a)