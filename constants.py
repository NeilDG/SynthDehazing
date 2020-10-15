# -*- coding: utf-8 -*-
import os

DATASET_VEMON_PATH_COMPLETE = "E:/VEMON_Transfer/train/A/"
DATASET_VEMON_PATH_PATCH = "E:/VEMON_Transfer/train/B/"
DATASET_GTA_PATH_2= "E:/VEMON Dataset/pending/frames/"

DATASET_NOISY_GTA_PATH = "E:/Noisy GTA/noisy/"
DATASET_CLEAN_GTA_PATH = "E:/Noisy GTA/clean/"

DATASET_HAZY_PATH_COMPLETE = "E:/Synth Hazy/hazy/"
DATASET_CLEAN_PATH_COMPLETE = "E:/Synth Hazy/clean/"
DATASET_HAZY_PATH_PATCH = "E:/Synth Hazy - Patch/hazy/"
DATASET_CLEAN_PATH_PATCH = "E:/Synth Hazy - Patch/clean/"

DATASET_HAZY_TEST_PATH_1 = "E:/Hazy Dataset Benchmark/I-HAZE/hazy/"
DATASET_HAZY_TEST_PATH_2 = "E:/Hazy Dataset Benchmark/Unannotated/"

DATASET_PLACES_PATH = "E:/Places Dataset/"
DATASET_DIV2K_PATH = "E:/VEMON_Transfer/train/C/"

PATCH_IMAGE_SIZE = (32, 32)
TEST_IMAGE_SIZE = (512, 512)
DIV2K_IMAGE_SIZE = (2040, 1404)
FIG_SIZE = (16, 32)
TENSORBOARD_PATH = os.getcwd() + "/train_plot/"

#========================================================================#
OPTIMIZER_KEY = "optimizer"
GENERATOR_KEY = "generator"
DISCRIMINATOR_KEY = "discriminator"

DEHAZER_VERSION = "dehazer_v1.09"
COLORIZER_VERSION = "colorizer_v1.07"
COLOR_TRANSFER_VERSION = "dehaze_colortransfer_v1.07"

ITERATION = "2"

DEHAZER_CHECKPATH = 'checkpoint/' + DEHAZER_VERSION + "_" + ITERATION +'.pt'
COLORIZER_CHECKPATH = 'checkpoint/' + COLORIZER_VERSION + "_" + ITERATION +'.pt'
COLOR_TRANFER_CHECKPATH = 'checkpoint/' + COLOR_TRANSFER_VERSION + "_" + ITERATION +'.pt'
DENOISE_CHECKPATH = 'checkpoint/gta_denoise_v1.00_1.pt'

# dictionary keys
G_LOSS_KEY = "g_loss"
IDENTITY_LOSS_KEY = "id"
CYCLE_LOSS_KEY = "cyc"
TV_LOSS_KEY = "tv"
G_ADV_LOSS_KEY = "g_adv"
LIKENESS_LOSS_KEY = "likeness"
REALNESS_LOSS_KEY = "realness"
COLOR_SHIFT_LOSS_KEY = "colorshift"

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
infer_size = 16

brightness_enhance = 1.0
contrast_enhance = 1.0

#Running on COARE?
is_coare = 0
    