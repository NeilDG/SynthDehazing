# -*- coding: utf-8 -*-
import os

DATASET_BIRD_NORMAL_PATH = "E:/VEMON Dataset/pending/frames/"
DATASET_BIRD_HOMOG_PATH = "E:/VEMON Dataset/pending/homog_frames/"
DATASET_BIRD_GROUND_TRUTH_PATH = "E:/VEMON Dataset/pending/topdown_frames/"
DATASET_BIRD_ALTERNATIVE_PATH = "E:/GTA Bird Dataset/raw/"

DATASET_VEMON_FRONT_PATH = "E:/VEMON Dataset/frames/"
DATASET_VEMON_HOMOG_PATH = "E:/VEMON Dataset/homog_frames/"

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


STYLE_GAN_VERSION = "style_v1.00"
STYLE_ITERATION = "5"
#STYLE_CHECKPATH = 'checkpoint/' + STYLE_GAN_VERSION + '.pt'
STYLE_CHECKPATH = 'checkpoint/' + STYLE_GAN_VERSION + "_" + STYLE_ITERATION +'.pt'

 # Set random seed for reproducibility
manualSeed = 999

# Number of training epochs
num_epochs = 20

# Batch size during training
batch_size = 4
infer_size = 64

num_workers = 12

#Running on COARE?
is_coare = 0
    