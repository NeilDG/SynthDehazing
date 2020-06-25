# -*- coding: utf-8 -*-
import os

DATASET_VEMON_PATH = "E:/VEMON_Transfer/train/A/"
DATASET_GTA_PATH = "E:/VEMON_Transfer/train/B/"
DATASET_PLACES_PATH = "E:/Places Dataset/"

BIRD_IMAGE_SIZE = (128, 128) #320 x 192 original
# Spatial size of training images. All images will be resized to this
#   size using a transformer.
NORMAL_IMAGE_SIZE = 128
TOPDOWN_IMAGE_SIZE = 128
FIG_SIZE = NORMAL_IMAGE_SIZE / 4

TENSORBOARD_PATH = os.getcwd() + "/train_plot/"

#========================================================================#
GAN_VERSION = "td_v3.01"
GAN_ITERATION = "1"
OPTIMIZER_KEY = "optimizer"
CHECKPATH = 'checkpoint/' + GAN_VERSION +'.pt'
GENERATOR_KEY = "generator"
DISCRIMINATOR_KEY = "discriminator"


STYLE_GAN_VERSION = "style_v1.02"
STYLE_ITERATION = "3"
STYLE_CHECKPATH = 'checkpoint/' + STYLE_GAN_VERSION + "_" + STYLE_ITERATION +'.pt'
 
 # Set random seed for reproducibility
manualSeed = 999

# Number of training epochs
num_epochs = 30

# Batch size during training
batch_size = 16
infer_size = 128

num_workers = 12

#Running on COARE?
is_coare = 0
    