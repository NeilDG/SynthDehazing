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


def assemble_train_data(path_a, path_b, num_image_to_load=-1):
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
        for (root, dirs, files) in os.walk(path_a):
            for f in files:
                file_name = os.path.join(root, f)
                a_list.append(file_name)
                loaded = loaded + 1
                if (num_image_to_load != -1 and len(a_list) == num_image_to_load):
                    break

    return a_list


def load_test_dataset(path_a, path_b, batch_size=8, num_image_to_load=-1):
    a_list, b_list = assemble_train_data(path_a, path_b, num_image_to_load)
    print("Length of test images: %d, %d." % (len(a_list), len(b_list)))

    data_loader = torch.utils.data.DataLoader(
        image_dataset.ColorTransferTestDataset(a_list, b_list),
        batch_size=batch_size,
        num_workers=2,
        shuffle=False
    )

    return data_loader


def load_dark_channel_test_dataset(path_a, path_b, batch_size=8, num_image_to_load=-1):
    a_list, b_list = assemble_train_data(path_a, path_b, num_image_to_load)
    print("Length of dark channel test images: %d, %d." % (len(a_list), len(b_list)))

    data_loader = torch.utils.data.DataLoader(
        image_dataset.DarkChannelTestDataset(a_list, b_list),
        batch_size=batch_size,
        num_workers=2,
        shuffle=False
    )

    return data_loader


def load_dehaze_dataset(path_a, path_b, batch_size=8, num_image_to_load=-1):
    a_list, b_list = assemble_train_data(path_a, path_b, num_image_to_load)
    # c_list = assemble_unpaired_data(path_c, len(b_list), True)
    print("Length of dehazing dataset: %d, %d" % (len(a_list), len(b_list)))
    # print("Length of dehazing dataset: %d, %d, %d" % (len(a_list), len(b_list), len(c_list)))

    data_loader = torch.utils.data.DataLoader(
        image_dataset.DarkChannelHazeDataset(a_list, b_list),
        batch_size=batch_size,
        num_workers=6,
        shuffle=True
    )

    return data_loader


def load_dark_channel_dataset(path_a, path_b, batch_size=8, num_image_to_load=-1):
    a_list, b_list = assemble_train_data(path_a, path_b, num_image_to_load)
    # c_list = assemble_unpaired_data(path_c, len(b_list), True)
    print("Length of dehazing dataset: %d, %d" % (len(a_list), len(b_list)))
    # print("Length of dehazing dataset: %d, %d, %d" % (len(a_list), len(b_list), len(c_list)))

    data_loader = torch.utils.data.DataLoader(
        image_dataset.DarkChannelHazeDataset(a_list, b_list),
        batch_size=batch_size,
        num_workers=6,
        shuffle=True
    )

    return data_loader



def load_dehaze_dataset_test(path_a, batch_size=8, num_image_to_load=-1):
    a_list = assemble_unpaired_data(path_a, num_image_to_load)
    print("Length of dehazing test dataset: %d" % (len(a_list)))

    data_loader = torch.utils.data.DataLoader(
        image_dataset.HazeTestDataset(a_list),
        batch_size=batch_size,
        num_workers=2,
        shuffle=False
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


def load_rgb_test_dataset(path_a, batch_size=8, num_image_to_load=-1):
    a_list = assemble_unpaired_data(path_a, num_image_to_load)
    print("Length of color dataset: %d." % (len(a_list)))

    rgb_data_loader = torch.utils.data.DataLoader(
        image_dataset.ColorTestDataset(a_list),
        batch_size=batch_size,
        num_workers=2,
        shuffle=True
    )

    return rgb_data_loader


def load_color_train_dataset(path_a, path_b, path_c, batch_size=8, num_image_to_load=-1):
    a_list = assemble_unpaired_data(path_a, num_image_to_load / 2)
    b_list = assemble_unpaired_data(path_b, num_image_to_load / 2)
    # specific for Hazy dataset. Combine synth and real data
    a_list = a_list + b_list
    c_list = assemble_unpaired_data(path_c, len(a_list), True)

    print("Length of images: %d, %d." % (len(a_list), len(c_list)))

    data_loader = torch.utils.data.DataLoader(
        image_dataset.ColorTransferDataset(a_list, c_list),
        batch_size=batch_size,
        num_workers=10,
        shuffle=True
    )

    return data_loader

def load_latent_dataset(path_a, batch_size=8, num_image_to_load=-1):
    a_list = assemble_unpaired_data(path_a, num_image_to_load)
    print("Length of dehazing train dataset: %d" % (len(a_list)))

    data_loader = torch.utils.data.DataLoader(
        image_dataset.LatentDataset(a_list),
        batch_size=batch_size,
        num_workers=6,
        shuffle=False
    )

    return data_loader
