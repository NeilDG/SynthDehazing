# -*- coding: utf-8 -*-
import os

DATASET_VEMON_PATH = "E:/VEMON_Transfer/train/A/"
DATASET_GTA_PATH = "E:/VEMON_Transfer/train/B/"
DATASET_GTA_PATH_2= "E:/VEMON Dataset/pending/frames/"

DATASET_NOISY_GTA_PATH = "E:/Noisy GTA/noisy/"
DATASET_CLEAN_GTA_PATH = "E:/Noisy GTA/clean/"

DATASET_HAZY_PATH = "E:/Synth Hazy/hazy/"
DATASET_CLEAN_PATH = "E:/Synth Hazy/clean/"

DATASET_PLACES_PATH = "E:/Places Dataset/"
DATASET_DIV2K_PATH = "E:/VEMON_Transfer/train/C/"

BIRD_IMAGE_SIZE = (32, 32) #320 x 192 original
TEST_IMAGE_SIZE = (400, 400)
DIV2K_IMAGE_SIZE = (2040, 1404)
FIG_SIZE = (TEST_IMAGE_SIZE[0], TEST_IMAGE_SIZE[1])
TENSORBOARD_PATH = os.getcwd() + "/train_plot/"

#========================================================================#
OPTIMIZER_KEY = "optimizer"
GENERATOR_KEY = "generator"
DISCRIMINATOR_KEY = "discriminator"

DEHAZER_VERSION = "dehazer_v1.04"
COLORIZER_VERSION = "dehazer_v1.04"
ITERATION = "6"

DEHAZER_CHECKPATH = 'checkpoint/' + DEHAZER_VERSION + "_" + ITERATION +'.pt'
COLORIZER_CHECKPATH = 'checkpoint/' + COLORIZER_VERSION + "_" + ITERATION +'.pt'
DENOISE_CHECKPATH = 'checkpoint/gta_denoise_v1.00_1.pt'

# dictionary keys
G_LOSS_KEY = "g_loss"
IDENTITY_LOSS_KEY = "id"
CYCLE_LOSS_KEY = "cyc"
TV_LOSS_KEY = "tv"
G_ADV_LOSS_KEY = "g_adv"
LIKENESS_LOSS_KEY = "likeness"
REALNESS_LOSS_KEY = "realness"


D_OVERALL_LOSS_KEY = "d_loss"
D_A_REAL_LOSS_KEY = "d_real_a"
D_A_FAKE_LOSS_KEY = "d_fake_a"
D_B_REAL_LOSS_KEY = "d_real_b"
D_B_FAKE_LOSS_KEY = "d_fake_b"

# Set random seed for reproducibility
manualSeed = 999

# Number of training epochs
num_epochs = 500

test_display_size = 8
display_size = 16 #must not be larger than batch size
batch_size = 256
infer_size = 32

num_workers = 12

#Running on COARE?
is_coare = 0
    