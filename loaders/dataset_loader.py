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


def assemble_paired_data(path_a, path_b, num_image_to_load=-1):
    a_list = [];
    b_list = []

    for (root, dirs, files) in os.walk(path_a):
        for f in files:
            file_name = os.path.join(root, f)
            # print(file_name)
            a_list.append(file_name)
            if (num_image_to_load != -1 and len(a_list) == num_image_to_load):
                break

    for (root, dirs, files) in os.walk(path_b):
        for f in files:
            file_name = os.path.join(root, f)
            b_list.append(file_name)
            if (num_image_to_load != -1 and len(b_list) == num_image_to_load):
                break

    return a_list, b_list


def assemble_unpaired_data(path_a, num_image_to_load=-1, force_complete=False):
    a_list = []

    loaded = 0
    for (root, dirs, files) in os.walk(path_a):
        for f in files:
            file_name = os.path.join(root, f)
            a_list.append(file_name)
            loaded = loaded + 1
            if (num_image_to_load != -1 and len(a_list) == num_image_to_load):
                break

    while loaded != num_image_to_load and force_complete:
        print("Looking for files in ", path_a)
        for (root, dirs, files) in os.walk(path_a):
            for f in files:
                file_name = os.path.join(root, f)
                a_list.append(file_name)
                loaded = loaded + 1
                if (num_image_to_load != -1 and len(a_list) == num_image_to_load):
                    break

    return a_list


def load_test_dataset(path_a, path_b, batch_size=8, num_image_to_load=-1):
    a_list, b_list = assemble_paired_data(path_a, path_b, num_image_to_load)
    print("Length of test images: %d, %d." % (len(a_list), len(b_list)))

    data_loader = torch.utils.data.DataLoader(
        image_dataset.ColorTransferTestDataset(a_list, b_list),
        batch_size=batch_size,
        num_workers=2,
        shuffle=False
    )

    return data_loader

def load_model_based_transmission_dataset_test(img_a, img_b, light_path, crop_size, batch_size=8, num_image_to_load=-1):
    a_list, b_list = assemble_paired_data(img_a, img_b, num_image_to_load)
    light_list = assemble_unpaired_data(light_path, num_image_to_load=num_image_to_load)
    print("Length of test transmission dataset: %d, %d %d" % (len(a_list), len(b_list), len(light_list)))

    data_loader = torch.utils.data.DataLoader(
        image_dataset.TransmissionDataset_Single(a_list, b_list, light_list, crop_size),
        batch_size=batch_size,
        num_workers=2,
        shuffle=True
    )
    return data_loader


def load_dehazing_dataset(path_a, path_b, return_ground_truth = False, batch_size=8, num_image_to_load=-1):
    a_list = assemble_unpaired_data(path_a, num_image_to_load)
    print("Length of training transmission dataset: %d" % (len(a_list)))

    data_loader = torch.utils.data.DataLoader(
        image_dataset.DehazingDataset(a_list, path_b, (32, 32), True, return_ground_truth),
        batch_size=batch_size,
        num_workers=constants.num_workers,
        shuffle=True
    )
    return data_loader

def load_dehazing_dataset_infer(path_a, path_b, return_ground_truth = False, batch_size=8, num_image_to_load=-1):
    a_list = assemble_unpaired_data(path_a, num_image_to_load)
    print("Length of training transmission dataset: %d" % (len(a_list)))

    data_loader = torch.utils.data.DataLoader(
        image_dataset.DehazingDataset(a_list, path_b, (32, 32), False, return_ground_truth),
        batch_size=batch_size,
        num_workers=constants.num_workers,
        shuffle=False
    )
    return data_loader

def load_transmission_albedo_dataset(path_a, pseudo_path_a, path_b, return_ground_truth = False, batch_size=8, num_image_to_load=-1, num_workers=12):
    a_list = assemble_unpaired_data(path_a, num_image_to_load)
    pseudo_a_list = assemble_unpaired_data(pseudo_path_a, num_image_to_load)
    a_list = a_list + pseudo_a_list
    print("Length of training transmission dataset: %d" % (len(a_list)))

    data_loader = torch.utils.data.DataLoader(
        image_dataset.TransmissionAlbedoDataset(a_list, path_b, (128, 128), True, return_ground_truth),
        #image_dataset.TransmissionAlbedoDataset(a_list, path_b, (256, 256), False, return_ground_truth),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True
    )
    return data_loader

def load_dehaze_dataset_test(path_a, batch_size=8, num_image_to_load=-1):
    a_list = assemble_unpaired_data(path_a, num_image_to_load)
    print("Length of test dataset: %d" % (len(a_list)))

    data_loader = torch.utils.data.DataLoader(
        image_dataset.DehazingDatasetTest(a_list),
        batch_size=batch_size,
        num_workers=1,
        shuffle=True
    )
    return data_loader

def load_dehaze_dataset_test_paired(path_a, path_b, batch_size=8, num_image_to_load=-1):
    a_list, b_list = assemble_paired_data(path_a, path_b, num_image_to_load)
    print("Length of paired test dataset: %d %d" % (len(a_list), len(b_list)))

    data_loader = torch.utils.data.DataLoader(
        image_dataset.DehazingDatasetPaired(a_list, b_list, (256, 256)),
        batch_size=batch_size,
        num_workers=1,
        shuffle=True
    )
    return data_loader


def load_rgb_dataset(path_a, batch_size=8, num_image_to_load=-1):
    a_list = assemble_unpaired_data(path_a, num_image_to_load)
    print("Length of color dataset: %d. " % (len(a_list)))

    rgb_data_loader = torch.utils.data.DataLoader(
        image_dataset.ColorDataset(a_list),
        batch_size=batch_size,
        num_workers=6,
        shuffle=True
    )

    return rgb_data_loader


def load_color_test_dataset(path_a, batch_size=8, num_image_to_load=-1):
    a_list = assemble_unpaired_data(path_a, num_image_to_load)
    print("Length of color dataset: %d." % (len(a_list)))

    rgb_data_loader = torch.utils.data.DataLoader(
        image_dataset.ColorTestDataset(a_list),
        batch_size=batch_size,
        num_workers=2,
        shuffle=True
    )

    return rgb_data_loader

def load_color_train_dataset(path_a, path_c, batch_size=8, num_image_to_load=-1):
    a_list = assemble_unpaired_data(path_a, num_image_to_load / 2)
    c_list = assemble_unpaired_data(path_c, len(a_list), True)

    print("Length of images: %d, %d." % (len(a_list), len(c_list)))

    data_loader = torch.utils.data.DataLoader(
        image_dataset.ColorTransferDataset(a_list, c_list),
        batch_size=batch_size,
        num_workers=6,
        shuffle=True
    )

    return data_loader

def load_color_albedo_train_dataset(path_a, path_c, depth_dir, batch_size=8, num_image_to_load=-1):
    a_list = assemble_unpaired_data(path_a, num_image_to_load / 2)
    c_list = assemble_unpaired_data(path_c, len(a_list), True)

    print("Length of images: %d, %d." % (len(a_list), len(c_list)))

    data_loader = torch.utils.data.DataLoader(
        image_dataset.ColorAlbedoDataset(a_list, c_list, depth_dir),
        batch_size=batch_size,
        num_workers=6,
        shuffle=True
    )

    return data_loader

def load_color_albedo_test_dataset(path_a, path_c, depth_dir, batch_size=8, num_image_to_load=-1):
    a_list = assemble_unpaired_data(path_a, num_image_to_load / 2)
    c_list = assemble_unpaired_data(path_c, len(a_list), True)

    print("Length of images: %d, %d." % (len(a_list), len(c_list)))

    data_loader = torch.utils.data.DataLoader(
        image_dataset.ColorAlbedoTestDataset(a_list, c_list, depth_dir),
        batch_size=batch_size,
        num_workers=2,
        shuffle=True
    )

    return data_loader

def load_airlight_dataset_train(path_a, path_b, return_ground_truth = False, batch_size=8, num_image_to_load=-1):
    a_list = assemble_unpaired_data(path_a, num_image_to_load)
    print("Length of training airlight dataset: %d" % (len(a_list)))

    data_loader = torch.utils.data.DataLoader(
        image_dataset.AirlightDataset(a_list, path_b, (32, 32), True, return_ground_truth),
        batch_size=batch_size,
        num_workers=constants.num_workers,
        shuffle=True
    )
    return data_loader

def load_airlight_dataset_test(path_a, batch_size=8, num_image_to_load=-1):
    a_list = assemble_unpaired_data(path_a, num_image_to_load)
    print("Length of test airlight dataset: %d" % (len(a_list)))

    data_loader = torch.utils.data.DataLoader(
        image_dataset.AirlightDatasetTest(a_list),
        batch_size=batch_size,
        num_workers=1,
        shuffle=True
    )
    return data_loader
