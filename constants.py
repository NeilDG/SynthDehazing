# -*- coding: utf-8 -*-
import os

DATASET_VEMON_PATH = "E:/VEMON_Transfer/train/A/"
DATASET_GTA_PATH = "E:/VEMON_Transfer/train/B/"
DATASET_GTA_PATH_2= "E:/VEMON Dataset/pending/frames/"
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
STYLE_ITERATION = "9"
STYLE_CHECKPATH = 'checkpoint/' + STYLE_GAN_VERSION + "_" + STYLE_ITERATION +'.pt'

# dictionary keys
IDENTITY_LOSS_KEY = "id"
CYCLE_LOSS_KEY = "cyc"
TV_LOSS_KEY = "tv"
ADV_LOSS_KEY = "adv"

D_REAL_LOSS_KEY = "d_real"
D_FAKE_LOSS_KEY = "d_fake"

# Set random seed for reproducibility
manualSeed = 999

# Number of training epochs
num_epochs = 100

display_size = 16 #must not be larger than batch size
batch_size = 32
infer_size = 128

num_workers = 12

#Running on COARE?
is_coare = 0
    