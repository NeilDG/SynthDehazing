# -*- coding: utf-8 -*-
"""
Class for producing figures
Created on Sat May  2 09:09:21 2020

@author: delgallegon
"""

import torch.utils.data
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from trainers import transmission_trainer
from loaders import dataset_loader
from model import vanilla_cycle_gan as cycle_gan
from model import ffa_net as ffa
import constants
from torchvision import transforms
import cv2
from utils import tensor_utils
from utils import dark_channel_prior
import os
import glob
from skimage.metrics import peak_signal_noise_ratio
from processing import gist
from custom_losses import vgg_loss_model

def main():
    # os.system("python \"inference.py\" --path=\"E:/Hazy Dataset Benchmark/O-HAZE/hazy/*.jpg\" --output=\"./output/O-Haze/\"")
    # os.system("python \"inference.py\" --path=\"E:/Hazy Dataset Benchmark/I-HAZE/hazy/\" --output=\"./output/I-Haze/\"")
    # os.system("python \"inference.py\" --path=\"E:/Hazy Dataset Benchmark/OTS_BETA/haze/*0.95_0.2.jpg\" --output=\"./output/RESIDE-OTS/\"")

    os.system("python \"color_transfer.py\" --path=\"E:/Synth Hazy 3/clean/*.png\" --output=\"./output/Styled/\"")

if __name__=="__main__":
    main()

        