# -*- coding: utf-8 -*-
import os

DATASET_VEMON_PATH = "E:/VEMON_Transfer/train/A/"
DATASET_GTA_PATH = "E:/VEMON_Transfer/train/B/"
DATASET_GTA_PATH_2= "E:/VEMON Dataset/pending/frames/"
DATASET_PLACES_PATH = "E:/Places Dataset/"

BIRD_IMAGE_SIZE = (32, 32) #320 x 192 original
TEST_IMAGE_SIZE = (128, 128)
FIG_SIZE = (BIRD_IMAGE_SIZE[0] / 4, BIRD_IMAGE_SIZE[1] / 4)
TENSORBOARD_PATH = os.getcwd() + "/train_plot/"

#========================================================================#
OPTIMIZER_KEY = "optimizer"
GENERATOR_KEY = "generator"
DISCRIMINATOR_KEY = "discriminator"

STYLE_GAN_VERSION = "style_v1.02"
STYLE_ITERATION = "7"
STYLE_CHECKPATH = 'checkpoint/' + STYLE_GAN_VERSION + "_" + STYLE_ITERATION +'.pt'

# dictionary keys
G_LOSS_KEY = "g_loss"
IDENTITY_LOSS_KEY = "id"
CYCLE_LOSS_KEY = "cyc"
TV_LOSS_KEY = "tv"
ADV_LOSS_KEY = "adv"
PERCEP_LOSS_KEY = "vgg"

D_OVERALL_LOSS_KEY = "d_loss"
D_REAL_LOSS_KEY = "d_real"
D_FAKE_LOSS_KEY = "d_fake"

# Set random seed for reproducibility
manualSeed = 999

# Number of training epochs
num_epochs = 100

test_display_size = 8
display_size = 32 #must not be larger than batch size
batch_size = 32
infer_size = 128

num_workers = 12

#Running on COARE?
is_coare = 0
    