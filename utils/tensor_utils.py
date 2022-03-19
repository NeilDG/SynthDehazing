# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 20:28:09 2019

Image and tensor utilities
@author: delgallegon
"""
import numbers

import kornia
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import math
import cv2
from torch.autograd import Variable
import torch
from utils import pytorch_colors
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity
from utils import dark_channel_prior
import constants

# for attaching hooks on pretrained models
class SaveFeatures(nn.Module):
    features = None;

    def __init__(self, m):
        self.hook = m.register_forward_hook(self.hook_fn);

    def hook_fn(self, module, input, output):
        self.features = output;

    def close(self):
        self.hook.remove();


class CombineFeatures(nn.Module):
    features = None;

    def __init(self, m, features):
        self.hook = m.register_forward_hook(self.hook_fn)
        self.features = features

    def hook_fn(self, module, input, output):
        self.features = self.features + output

    def close(self):
        self.hook.remove()


def normalize_to_matplotimg(img_tensor, batch_idx, std, mean):
    img = img_tensor[batch_idx, :, :, :].numpy()
    img = np.moveaxis(img, -1, 0)
    img = np.moveaxis(img, -1, 0)  # for properly displaying image in matplotlib

    img = ((img * std) + mean)  # normalize back to 0-1 range

    img = cv2.convertScaleAbs(img, alpha=(255.0))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def convert_to_matplotimg(img_tensor, batch_idx):
    img = img_tensor[batch_idx, :, :, :].numpy()
    img = np.moveaxis(img, -1, 0)
    img = np.moveaxis(img, -1, 0)  # for properly displaying image in matplotlib

    img = cv2.convertScaleAbs(img, alpha=(255.0))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def convert_to_opencv(img_tensor):
    img = img_tensor
    img = np.moveaxis(img, -1, 0)
    img = np.moveaxis(img, -1, 0)

    return img


def load_true_img(img_path):
    img = cv2.imread(img_path)

    img = cv2.normalize(img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

def load_metrics_compatible_img(img_path, im_size: tuple):
    img = cv2.imread(img_path)
    img = cv2.resize(img, im_size)

    img = cv2.normalize(img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


# loads an image compatible with opencv
def load_image(file_path):
    img = cv2.imread(file_path)
    if (img is not None):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        print("Image ", file_path, " not found.")
    return img


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def preprocess_batch(batch):
    batch = batch.transpose(0, 1)
    (r, g, b) = torch.chunk(batch, 3)
    batch = torch.cat((b, g, r))
    batch = batch.transpose(0, 1)
    return batch


def merge_yuv_results_to_rgb(y_tensor, uv_tensor):
    uv_tensor = uv_tensor.transpose(0, 1)
    y_tensor = y_tensor.transpose(0, 1)

    (u, v) = torch.chunk(uv_tensor, 2)
    yuv_tensor = torch.cat((y_tensor, u, v))
    rgb_tensor = pytorch_colors.lab_to_rgb(yuv_tensor.transpose(0, 1))
    # rgb_tensor = ((rgb_tensor * 0.5) + 0.5) #normalize back to 0-1 range
    rgb_tensor = ((rgb_tensor * 1.0) + 1.0)
    return rgb_tensor


def yuv_to_rgb(yuv_tensor):
    rgb_tensor = pytorch_colors.yuv_to_rgb(yuv_tensor)
    return rgb_tensor


def rgb_to_yuv(rgb_tensor):
    return pytorch_colors.rgb_to_yuv(rgb_tensor)


def change_yuv(y_tensor, yuv_tensor):
    yuv_tensor = yuv_tensor.transpose(0, 1)
    y_tensor = y_tensor.transpose(0, 1)
    (y, u, v) = torch.chunk(yuv_tensor, 3)
    yuv_tensor = torch.cat((y_tensor, u, v))
    return yuv_tensor.transpose(0, 1)


def replace_dark_channel(rgb_tensor, dark_channel_old, dark_channel_new, alpha=0.7, beta=0.7):
    yuv_tensor = pytorch_colors.rgb_to_yuv(rgb_tensor)

    yuv_tensor = yuv_tensor.transpose(0, 1)
    dark_channel_old = dark_channel_old.transpose(0, 1)
    dark_channel_new = dark_channel_new.transpose(0, 1)

    (y, u, v) = torch.chunk(yuv_tensor, 3)

    # deduct old dark channel from all channels and add new one
    # r = r - dark_channel_old + dark_channel_new
    # g = g - dark_channel_old + dark_channel_new
    # b = b - dark_channel_old + dark_channel_new
    y = y - (dark_channel_old * alpha) + (dark_channel_new * beta)

    yuv_tensor = torch.cat((y, u, v))
    rgb_tensor = pytorch_colors.yuv_to_rgb(yuv_tensor.transpose(0, 1))
    return rgb_tensor


def replace_y_channel(rgb_tensor, y_new):
    yuv_tensor = pytorch_colors.rgb_to_yuv(rgb_tensor)

    yuv_tensor = yuv_tensor.transpose(0, 1)
    y_new = y_new.transpose(0, 1)

    (y, u, v) = torch.chunk(yuv_tensor, 3)

    yuv_tensor = torch.cat((y_new, u, v))
    rgb_tensor = pytorch_colors.yuv_to_rgb(yuv_tensor.transpose(0, 1))
    return rgb_tensor

def get_y_channel(I):
    y, u, v = cv2.split(I)
    return y


def get_uv_channel(I):
    y, u, v = cv2.split(I)
    return cv2.merge((u, v))


def compare_transmissions(hazy_img, depth_map):
    T = generate_transmission(1 - depth_map, 1.2, True)  # real-world has 0.1 - 1.8 range only. Unity synth uses 0.03
    dc = get_dark_channel(hazy_img, w=7)
    atmosphere = estimate_atmosphere(hazy_img, dc)
    atmosphere = np.squeeze(atmosphere)

    dcp_transmission = dark_channel_prior.estimate_transmission(hazy_img,
                                                                dark_channel_prior.estimate_atmosphere(hazy_img, dc),
                                                                dc)

    fig, ax = plt.subplots(1, 2)
    fig.tight_layout()
    ax[0].imshow(T)
    ax[1].imshow(dcp_transmission)
    plt.show(block=True)


def perform_dehazing_equation(hazy_img, depth_map):
    # normalize
    T = generate_transmission(1 - depth_map, 1.2, True)  # real-world has 0.1 - 1.8 range only. Unity synth uses 0.03
    atmosphere = estimate_atmosphere(hazy_img, get_dark_channel(hazy_img, w=15))
    atmosphere = np.squeeze(atmosphere)
    # Z = np.multiply((1 - T), np.max(atmosphere))

    print("Min of input: ", np.min(hazy_img), " Max of input: ", np.max(hazy_img),
          "Min of depth: ", np.min(depth_map), " Max of depth: ", np.max(depth_map),
          "Min of T: ", np.min(T), " Max of T: ", np.max(T))

    # print("Min of A: ", np.min(Z), " Max of A:", np.max(Z),
    #     " Mean of A: ", np.mean(Z), " Mean of T: ", np.mean(T))

    # compute clear image with radiance term
    # clear_img = np.ones_like(hazy_img)
    # clear_img[:, :, 0] = ((hazy_img[:, :, 0] - np.full(np.shape(hazy_img[:, :, 0]), atmosphere[0])) / np.maximum(T,
    #                                                                                                              0.5)) + np.full(
    #     np.shape(hazy_img[:, :, 0]), atmosphere[0])
    # clear_img[:, :, 1] = ((hazy_img[:, :, 1] - np.full(np.shape(hazy_img[:, :, 1]), atmosphere[1])) / np.maximum(T,
    #                                                                                                              0.5)) + np.full(
    #     np.shape(hazy_img[:, :, 1]), atmosphere[1])
    # clear_img[:, :, 2] = ((hazy_img[:, :, 2] - np.full(np.shape(hazy_img[:, :, 2]), atmosphere[2])) / np.maximum(T,
    #                                                                                                              0.5)) + np.full(
    #     np.shape(hazy_img[:, :, 2]), atmosphere[2])

    clear_img = np.ones_like(hazy_img)
    T = np.resize(T, np.shape(clear_img[:, :, 0]))
    print("Shapes: ", np.shape(clear_img), np.shape(hazy_img), np.shape(T))
    clear_img[:, :, 0] = (hazy_img[:, :, 0] - (np.full(np.shape(hazy_img[:, :, 0]), atmosphere[0]) * (1 - T))) / T
    clear_img[:, :, 1] = (hazy_img[:, :, 1] - (np.full(np.shape(hazy_img[:, :, 1]), atmosphere[1]) * (1 - T))) / T
    clear_img[:, :, 2] = (hazy_img[:, :, 2] - (np.full(np.shape(hazy_img[:, :, 2]), atmosphere[2]) * (1 - T))) / T

    print("Min of hazy img: ", np.min(hazy_img), " Max of hazy img: ", np.max(hazy_img))
    print("Min of clear img: ", np.min(clear_img), " Max of clear img: ", np.max(clear_img))

    old_img = cv2.normalize(hazy_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    clear_img = np.clip(clear_img, 0.0, 1.0)

    fig, ax = plt.subplots(1, 2)
    fig.tight_layout()
    ax[0].imshow(old_img)
    ax[1].imshow(clear_img)
    plt.show(block=True)


def introduce_haze(hazy_img, clear_img, depth_map):
    LENGTH = 3
    fig, ax = plt.subplots(LENGTH, 1)
    fig.tight_layout()
    beta = 0.0
    for i in range(LENGTH):
        beta = beta + 0.5
        T = generate_transmission(1 - depth_map, beta, True)  # real-world has 0.1 - 1.8 range only. Unity synth uses 0.03
        ax[i].imshow(T)
        ax[i].axis('off')

    plt.show()

    atmosphere_val = np.random.uniform(0.5, 1.2)
    atmosphere = [atmosphere_val, atmosphere_val, atmosphere_val] #atmosphere values can change per channel. but make it uniform for this time
    hazy_img_like = np.zeros_like(clear_img)
    T = np.resize(T, np.shape(clear_img[:, :, 0]))
    hazy_img_like[:, :, 0] = (T * clear_img[:, :, 0]) + atmosphere[0] * (1 - T)
    hazy_img_like[:, :, 1] = (T * clear_img[:, :, 1]) + atmosphere[1] * (1 - T)
    hazy_img_like[:, :, 2] = (T * clear_img[:, :, 2]) + atmosphere[2] * (1 - T)

    # reverse equation
    clear_img_like = np.zeros_like(hazy_img)
    T = np.resize(T, np.shape(clear_img[:, :, 0]))
    print("Shapes: ", np.shape(clear_img), np.shape(hazy_img), np.shape(T))
    clear_img_like[:, :, 0] = (hazy_img_like[:, :, 0] - (np.full(np.shape(hazy_img_like[:, :, 0]), atmosphere[0]) * (1 - T))) / T
    clear_img_like[:, :, 1] = (hazy_img_like[:, :, 1] - (np.full(np.shape(hazy_img_like[:, :, 1]), atmosphere[1]) * (1 - T))) / T
    clear_img_like[:, :, 2] = (hazy_img_like[:, :, 2] - (np.full(np.shape(hazy_img_like[:, :, 2]), atmosphere[2]) * (1 - T))) / T

    hazy_img_like = np.clip(hazy_img_like, 0.0, 1.0)
    clear_img_like = np.clip(clear_img_like, 0.0, 1.0)
    hazy_img_like = cv2.normalize(hazy_img_like, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                  dtype=cv2.CV_8U)
    clear_img_like = cv2.normalize(clear_img_like, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                   dtype=cv2.CV_8U)
    clear_img = cv2.normalize(clear_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    fig, ax = plt.subplots(2, 2)
    fig.tight_layout()
    ax[0, 0].imshow(hazy_img)
    ax[0, 1].imshow(hazy_img_like)
    ax[1, 0].imshow(clear_img)
    ax[1, 1].imshow(clear_img_like)
    plt.show(block=True)

def introduce_haze_albedo(clear_img, depth_map, albedo_img):

    beta = 1.2
    T = generate_transmission(1 - depth_map, beta, True)  # real-world has 0.1 - 1.8 range only. Unity synth uses 0.03

    clear_img_float = cv2.normalize(clear_img, dst = None,alpha = 0, beta = 1, norm_type = cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    albedo_img_float = cv2.normalize(albedo_img, dst = None, alpha=0, beta = 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    #clear_img_float = cv2.cvtColor(clear_img_float, cv2.COLOR_BGR2GRAY)
    #albedo_img_float = cv2.cvtColor(albedo_img_float, cv2.COLOR_BGR2GRAY)

    fig, ax = plt.subplots(1, 2)
    fig.tight_layout()
    ax[0].imshow(clear_img_float)
    ax[1].imshow(albedo_img_float)
    plt.show(block=True)

    atmosphere_val = np.random.uniform(0.5, 1.2)
    atmosphere = [atmosphere_val, atmosphere_val, atmosphere_val]  # atmosphere values can change per channel. but make it uniform for this time
    hazy_img_like = np.zeros_like(clear_img)
    T = np.resize(T, np.shape(clear_img[:, :, 0]))
    hazy_img_like[:, :, 0] = (T * albedo_img_float[:, :, 0]) + atmosphere[0] * (1 - T)
    hazy_img_like[:, :, 1] = (T * albedo_img_float[:, :, 1]) + atmosphere[1] * (1 - T)
    hazy_img_like[:, :, 2] = (T * albedo_img_float[:, :, 2]) + atmosphere[2] * (1 - T)

    # reverse equation
    clear_img_like = np.zeros_like(clear_img)
    T = np.resize(T, np.shape(clear_img[:, :, 0]))
    print("Shapes: ", np.shape(clear_img), np.shape(T))
    clear_img_like[:, :, 0] = (hazy_img_like[:, :, 0] - (np.full(np.shape(hazy_img_like[:, :, 0]), atmosphere[0]) * (1 - T))) / T
    clear_img_like[:, :, 1] = (hazy_img_like[:, :, 1] - (np.full(np.shape(hazy_img_like[:, :, 1]), atmosphere[1]) * (1 - T))) / T
    clear_img_like[:, :, 2] = (hazy_img_like[:, :, 2] - (np.full(np.shape(hazy_img_like[:, :, 2]), atmosphere[2]) * (1 - T))) / T

    clear_img_like = np.clip(clear_img_like, 0.0, 1.0)
    clear_img_like = cv2.normalize(clear_img_like, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                   dtype=cv2.CV_8U)
    clear_img = cv2.normalize(clear_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    fig, ax = plt.subplots(2, 2)
    fig.tight_layout()
    ax[0, 0].imshow(hazy_img_like)
    ax[1, 0].imshow(clear_img)
    ax[1, 1].imshow(clear_img_like)
    plt.show(block=True)

def refine_dehaze_img(hazy_img, clear_img, T):
    T = np.resize(T, np.shape(hazy_img[:, :, 0]))
    hazy_img = cv2.normalize(hazy_img, dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    hazy_mask = np.zeros_like(hazy_img)
    hazy_mask[:, :, 0] = np.multiply(hazy_img[:, :, 0], T)
    hazy_mask[:, :, 1] = np.multiply(hazy_img[:, :, 1], T)
    hazy_mask[:, :, 2] = np.multiply(hazy_img[:, :, 2], T)

    clear_mask = np.zeros_like(clear_img)
    clear_mask[:, :, 0] = np.multiply(clear_img[:, :, 0], T)
    clear_mask[:, :, 1] = np.multiply(clear_img[:, :, 1], T)
    clear_mask[:, :, 2] = np.multiply(clear_img[:, :, 2], T)

    initial_img = np.multiply(hazy_img, (1 - hazy_mask)) + np.multiply(clear_img, clear_mask)

    return initial_img

def mask_haze(hazy_img, clear_img, depth_map):
    T = generate_transmission(depth_map, 0.8)

    hazy_mask = np.zeros_like(hazy_img)
    hazy_mask[:, :, 0] = np.multiply(hazy_img[:, :, 0], T)
    hazy_mask[:, :, 1] = np.multiply(hazy_img[:, :, 1], T)
    hazy_mask[:, :, 2] = np.multiply(hazy_img[:, :, 2], T)

    clear_mask = np.zeros_like(clear_img)
    clear_mask[:, :, 0] = np.multiply(clear_img[:, :, 0], T)
    clear_mask[:, :, 1] = np.multiply(clear_img[:, :, 1], T)
    clear_mask[:, :, 2] = np.multiply(clear_img[:, :, 2], T)

    hazy_mask = cv2.normalize(hazy_mask, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    hazy_img = cv2.normalize(hazy_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    fig, ax = plt.subplots(2, 2)
    fig.tight_layout()
    ax[0, 0].set_axis_off()
    ax[0, 0].imshow(hazy_img)
    ax[0, 1].set_axis_off()
    ax[0, 1].imshow(hazy_mask)
    ax[1, 0].set_axis_off()
    ax[1, 0].imshow(clear_img)
    ax[1, 1].set_axis_off()
    ax[1, 1].imshow(clear_mask)
    plt.show(block=True)

    hazy_mask = cv2.normalize(hazy_mask, dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    clear_mask = cv2.normalize(clear_mask, dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    hazy_img = cv2.normalize(hazy_img, dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    initial_img = np.zeros_like(hazy_img)
    # initial_img = np.multiply(hazy_img, (1 - hazy_mask)) + clear_mask
    initial_img = np.multiply(hazy_img, (1 - hazy_mask))

    atmosphere = estimate_atmosphere(initial_img, get_dark_channel(initial_img, w=15))
    atmosphere = np.squeeze(atmosphere)

    T = estimate_transmission(initial_img, atmosphere, get_dark_channel(initial_img, w=15))
    # T = generate_transmission(depth_map, 0.3)
    T = np.resize(T, np.shape(initial_img[:, :, 0]))

    refined_img = np.zeros_like(hazy_img)
    refined_img[:, :, 0] = (initial_img[:, :, 0] - (
                np.full(np.shape(initial_img[:, :, 0]), atmosphere[0]) * (1 - T))) / T
    refined_img[:, :, 1] = (initial_img[:, :, 1] - (
                np.full(np.shape(initial_img[:, :, 1]), atmosphere[1]) * (1 - T))) / T
    refined_img[:, :, 2] = (initial_img[:, :, 2] - (
                np.full(np.shape(initial_img[:, :, 2]), atmosphere[2]) * (1 - T))) / T

    initial_img = np.clip(initial_img, 0.0, 1.0)
    initial_img = cv2.normalize(initial_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    refined_img = np.clip(refined_img, 0.0, 1.0)
    refined_img = cv2.normalize(refined_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    fig, ax = plt.subplots(1, 2)
    fig.tight_layout()
    ax[0].set_axis_off()
    ax[0].imshow(initial_img)
    ax[1].set_axis_off()
    ax[1].imshow(refined_img)

    plt.show(block=True)


def perform_custom_dehazing_equation(hazy_img, clear_img):
    hazy_img = cv2.cvtColor(hazy_img, cv2.COLOR_RGB2YUV)
    clear_img = cv2.cvtColor(clear_img, cv2.COLOR_BGR2YUV)

    diff = hazy_img[:, :, 0] - clear_img[:, :, 0]
    diff_img = cv2.normalize(diff, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    new_img = np.zeros_like(hazy_img)
    new_img[:, :, 0] = hazy_img[:, :, 0] - diff
    new_img[:, :, 1] = hazy_img[:, :, 1]
    new_img[:, :, 2] = hazy_img[:, :, 2]
    new_img = np.clip(new_img, 0.0, 1.0)

    hazy_img = cv2.normalize(hazy_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    hazy_img = cv2.cvtColor(hazy_img, cv2.COLOR_YUV2RGB)
    new_img = cv2.normalize(new_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    new_img = cv2.cvtColor(new_img, cv2.COLOR_YUV2RGB)
    clear_img = cv2.normalize(clear_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    clear_img = cv2.cvtColor(clear_img, cv2.COLOR_YUV2RGB)

    fig, ax = plt.subplots(1, 3)
    fig.tight_layout()
    ax[0].imshow(hazy_img)
    ax[1].imshow(new_img)
    ax[2].imshow(clear_img)
    plt.show(block=True)

    return new_img


def perform_dehazing_equation_batch(hazy_img, T, filter_strength=0.75, normalize_input=True, normalize_T=True):
    T = np.squeeze(T)

    # normalize data to 0-1 range
    if (normalize_input):
        hazy_img = ((hazy_img * 0.5) + 0.5)

    if (normalize_T):
        T = ((T * 0.5) + 0.5)

    A = np.ones_like(T) - T

    print("Min of input: ", np.min(hazy_img[0]), " Max of input: ", np.max(hazy_img[0]),
          "Min of T: ", np.min(T[0]), " Max of T: ", np.max(T[0]),
          " Mean of A: ", np.mean(A[0]), " Mean of T: ", np.mean(T[0]))

    clear_img = np.zeros_like(hazy_img)
    print("Shapes:", np.shape(clear_img), np.shape(A), np.shape(T))
    clear_img[:, 0, :, :] = ((hazy_img[:, 0, :, :] - A) / np.maximum(T, filter_strength)) + A
    clear_img[:, 1, :, :] = ((hazy_img[:, 1, :, :] - A) / np.maximum(T, filter_strength)) + A
    clear_img[:, 2, :, :] = ((hazy_img[:, 2, :, :] - A) / np.maximum(T, filter_strength)) + A

    return clear_img

def make_rgb(batch):
    batch = batch.transpose(0, 1)
    (b, g, r) = torch.chunk(batch, 3)
    batch = torch.cat((r, g, b))
    batch = batch.transpose(0, 1)
    return batch


def subtract_imagenet_mean_batch(batch):
    """Subtract ImageNet mean pixel-wise from a BGR image."""
    tensortype = type(batch.data)
    mean = tensortype(batch.data.size())
    mean[:, 0, :, :] = 103.939
    mean[:, 1, :, :] = 116.779
    mean[:, 2, :, :] = 123.680
    return batch - Variable(mean).cuda()


def add_imagenet_mean_batch(batch):
    """Add ImageNet mean pixel-wise from a BGR image."""
    tensortype = type(batch.data)
    mean = tensortype(batch.data.size())
    mean[:, 0, :, :] = 103.939
    mean[:, 1, :, :] = 116.779
    mean[:, 2, :, :] = 123.680
    return batch + Variable(mean).cuda()


def imagenet_clamp_batch(batch, low, high):
    batch[:, 0, :, :].data.clamp_(low - 103.939, high - 103.939)
    batch[:, 1, :, :].data.clamp_(low - 116.779, high - 116.779)
    batch[:, 2, :, :].data.clamp_(low - 123.680, high - 123.680)


# computes a z_signal based on image size. Image size must always be a power of 2 and greater than 16x16.
def compute_z_signal(value, batch_size, image_size):
    z_size = (int(image_size[0] / 16), int(image_size[1] / 16))
    torch.manual_seed(value)
    z_signal = torch.randn((batch_size, 100, z_size[0], z_size[1]))
    return z_signal


# computes a z signal to be conacated with another image tensor.
def compute_z_signal_concat(value, batch_size, image_size):
    torch.manual_seed(value)
    z_signal = torch.randn((batch_size, 100, image_size[0], image_size[1]))
    return z_signal


def measure_ssim(img1, img2):
    # preprocessing
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2YUV)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2YUV)

    return structural_similarity(img1, img2, multichannel=True, gaussian_weights=True, sigma=1.5)

class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """

    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)
