# -*- coding: utf-8 -*-
"""
Created on Sat May 16 21:44:51 2020

@author: delgallegon
"""
from loaders import dataset_loader
import matplotlib.pyplot as plt
import constants
import torch
import torch.nn as nn
import numpy as np
import cv2
import torchvision.transforms as vtransforms
import torchvision.utils as vutils
from utils import tensor_utils
from utils import plot_utils
from loaders import dataset_loader

# def visualize_color_distribution(batch_size):
#     count = 2000
#     dataloader = dataset_loader.load_debug_dataset(batch_size, count)
#
#     # Plot some training images
#     name_batch, normal_batch, homog_batch, topdown_batch = next(iter(dataloader))
#     plt.figure(figsize=(constants.FIG_SIZE,constants.FIG_SIZE))
#     plt.axis("off")
#     plt.title("Training - Normal Images")
#     plt.imshow(np.transpose(vutils.make_grid(normal_batch[:batch_size], nrow = 16, padding=2, normalize=True),(1,2,0)))
#     plt.show()
#
#     plt.figure(figsize=(constants.FIG_SIZE,constants.FIG_SIZE))
#     plt.axis("off")
#     plt.title("Training - Homog Images")
#     plt.imshow(np.transpose(vutils.make_grid(homog_batch[:batch_size], nrow = 16, padding=2, normalize=True),(1,2,0)))
#     plt.show()
#
#     rgb_count = np.empty((count, 3), dtype=np.int32)
#     index = 0
#     for i, (name, normal_batch, homog_batch, topdown_batch) in enumerate(dataloader, 0):
#         for n, normal_tensor, homog_tensor, topdown_tensor in zip(name, normal_batch, homog_batch, topdown_batch):
#             normal_img = tensor_utils.convert_to_opencv(normal_tensor)
#             homog_img = tensor_utils.convert_to_opencv(homog_tensor)
#             topdown_img = tensor_utils.convert_to_opencv(topdown_tensor)
#
#             rgb_count[index, 0] += np.mean(normal_img[:,:,0] * 255.0)
#             rgb_count[index, 1] += np.mean(normal_img[:,:,1] * 255.0)
#             rgb_count[index, 2] += np.mean(normal_img[:,:,2] * 255.0)
#
#             if(np.mean(normal_img[:,:,0] * 255.0) < 120.0 or np.mean(normal_img[:,:,1] * 255.0) < 120.0 or
#                np.mean(normal_img[:,:,2] * 255.0) < 120.0):
#                 plt.imshow(normal_img)
#                 plt.show()
#                 print(n)
#
#             index = index + 1
#
#     plt.scatter(np.random.normal(0, 1.0, count), rgb_count[:,0], color=(1,0,0))
#     plt.scatter(np.random.normal(0, 1.0, count), rgb_count[:,1], color=(0,1,0))
#     plt.scatter(np.random.normal(0, 1.0, count), rgb_count[:,2], color=(0,0,1))
#     plt.show()

def visualize_color_distribution(img_dir_path_a, img_dir_path_b):
    img_list = dataset_loader.assemble_unpaired_data(img_dir_path_a, 500)
    rgb_list = np.empty((len(img_list), 3), dtype=np.float64)
    print("Reading images in ", img_dir_path_a)

    for i in range(len(img_list)):
        img = cv2.imread(img_list[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        red_channel = np.reshape(img[:, :, 0], -1)
        blue_channel = np.reshape(img[:, :, 1], -1)
        green_channel = np.reshape(img[:, :, 2], -1)

        rgb_list[i, 0] = np.round(np.mean(red_channel), 4)
        rgb_list[i, 1] = np.round(np.mean(blue_channel), 4)
        rgb_list[i, 2] = np.round(np.mean(green_channel), 4)

    plt.scatter(x=np.arange(0, len(img_list)), y = rgb_list[:,0], color = (1, 0, 0))
    plt.scatter(x=np.arange(0, len(img_list)), y = rgb_list[:,1], color=(0, 1, 0))
    plt.scatter(x=np.arange(0, len(img_list)), y = rgb_list[:,2], color=(0, 0, 1))
    #plt.show()

    img_list = dataset_loader.assemble_unpaired_data(img_dir_path_b, 500)
    rgb_list = np.empty((len(img_list), 3), dtype=np.float64)
    print("Reading images in ", img_dir_path_b)

    for i in range(len(img_list)):
        img = cv2.imread(img_list[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        red_channel = np.reshape(img[:, :, 0], -1)
        blue_channel = np.reshape(img[:, :, 1], -1)
        green_channel = np.reshape(img[:, :, 2], -1)

        #print(np.shape(red_channel), np.shape(blue_channel), np.shape(green_channel))

        rgb_list[i, 0] = np.round(np.mean(red_channel), 4)
        rgb_list[i, 1] = np.round(np.mean(blue_channel), 4)
        rgb_list[i, 2] = np.round(np.mean(green_channel), 4)

    plt.scatter(x=np.arange(0, len(img_list)), y=rgb_list[:, 0], color=(0.5, 0, 0))
    plt.scatter(x=np.arange(0, len(img_list)), y=rgb_list[:, 1], color=(0, 0.5, 0))
    plt.scatter(x=np.arange(0, len(img_list)), y=rgb_list[:, 2], color=(0, 0, 0.5))

    plt.show()

def visualize_edge_distribution(path_a):
    img_list = dataset_loader.assemble_unpaired_data(path_a, 500)
    edge_list = np.empty((len(img_list), 1), dtype=np.float64)
    print("Reading images in ", path_a)

    for i in range(len(edge_list)):
        img = cv2.imread(img_list[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
        sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
        sobel_img = sobel_x + sobel_y
        sobel_quality = np.round(np.linalg.norm(sobel_img), 4)

        edge_list[i] =  sobel_quality
        print(edge_list[i])

    plt.hist(edge_list)


def main():
    visualize_color_distribution(constants.DATASET_VEMON_PATH_PATCH_32, constants.DATASET_DIV2K_PATH_PATCH)

    visualize_edge_distribution(constants.DATASET_VEMON_PATH_PATCH_32)
    visualize_edge_distribution(constants.DATASET_DIV2K_PATH_PATCH)
    plt.show()
    
if __name__=="__main__": 
    main()   