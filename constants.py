# -*- coding: utf-8 -*-
import os

DATASET_VEMON_PATH_COMPLETE = "E:/VEMON_Transfer/train/full/"
DATASET_VEMON_PATH_PATCH_32 = "E:/VEMON_Transfer/train/32_patch/"
DATASET_VEMON_PATH_PATCH_64 = "E:/VEMON_Transfer/train/64_patch/"
DATASET_VEMON_PATH_PATCH_128 = "E:/VEMON_Transfer/train/128_patch/"

DATASET_HAZY_PATH_COMPLETE = "E:/Synth Hazy/hazy/"
DATASET_CLEAN_PATH_COMPLETE = "E:/Synth Hazy/clean/"
DATASET_DEPTH_PATH_COMPLETE = "E:/Synth Hazy/depth/"

DATASET_HAZY_PATH_PATCH = "E:/Synth Hazy - Patch/hazy/"
DATASET_CLEAN_PATH_PATCH = "E:/Synth Hazy - Patch/clean/"
DATASET_DEPTH_PATH_PATCH = "E:/Synth Hazy - Patch/depth/"

DATASET_OHAZE_PATH_PATCH_HAZY = "E:/O-HAZE - Patch/hazy/"
DATASET_OHAZE_PATH_PATCH_CLEAN = "E:/O-HAZE - Patch/clean/"
DATASET_OHAZE_PATH_CROP_HAZY = "E:/O-Haze - Crop/hazy/"
DATASET_OHAZE_PATH_CROP_CLEAN = "E:/O-Haze - Crop/clean/"

DATASET_OHAZE_HAZY_PATH_COMPLETE = "E:/Hazy Dataset Benchmark/O-HAZE/hazy/"
DATASET_OHAZE_CLEAN_PATH_COMPLETE = "E:/Hazy Dataset Benchmark/O-HAZE/GT/"
DATASET_IHAZE_HAZY_PATH_COMPLETE = "E:/Hazy Dataset Benchmark/I-HAZE/hazy/"
DATASET_IHAZE_CLEAN_PATH_COMPLETE = "E:/Hazy Dataset Benchmark/I-HAZE/GT/"
DATASET_RESIDE_TEST_PATH_COMPLETE = "E:/Hazy Dataset Benchmark/RESIDE-Unannotated/"

DATASET_PLACES_PATH = "E:/Places Dataset/"

DATASET_DIV2K_PATH_PATCH = "E:/Div2k - Patch/"
DATASET_DIV2K_PATH = "E:/DIV2K_train_HR/"

PATCH_IMAGE_SIZE = (32, 32)
TEST_IMAGE_SIZE = (512, 512)
DIV2K_IMAGE_SIZE = (2040, 1404)
FIG_SIZE = (16, 32)
TENSORBOARD_PATH = os.getcwd() + "/train_plot/"

#========================================================================#
OPTIMIZER_KEY = "optimizer"
GENERATOR_KEY = "generator"
DISCRIMINATOR_KEY = "discriminator"
LATENT_VECTOR_KEY = "latent_vector"

DEHAZER_VERSION = "dehazer_v1.13"
COLORIZER_VERSION = "colorizer_v1.08"
COLOR_TRANSFER_VERSION = "color_transfer_v1.11"
TRANSMISSION_VERSION = "transmission_estimator_v1.02"
AIRLIGHT_VERSION = "airlight_estimator_v1.00"
LATENT_VERSION = "latent_v1.00"

ITERATION = "9"

LATENT_CHECKPATH = 'checkpoint/' + LATENT_VERSION + "_" + ITERATION +'.pt'
LATENT_CHECKPATH_64 = 'checkpoint/' + LATENT_VERSION + "_" + ITERATION +'_64.pt'
LATENT_CHECKPATH_128 = 'checkpoint/' + LATENT_VERSION + "_" + ITERATION +'_128.pt'

DEHAZER_CHECKPATH = 'checkpoint/' + DEHAZER_VERSION + "_" + ITERATION +'.pt'
COLORIZER_CHECKPATH = 'checkpoint/' + COLORIZER_VERSION + "_" + ITERATION +'.pt'
COLOR_TRANSFER_CHECKPATH = 'checkpoint/' + COLOR_TRANSFER_VERSION + "_" + ITERATION + '.pt'
TRANSMISSION_ESTIMATOR_CHECKPATH = 'checkpoint/' + TRANSMISSION_VERSION + "_" + ITERATION + '.pt'
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
EDGE_LOSS_KEY = "edge"

D_OVERALL_LOSS_KEY = "d_loss"
D_A_REAL_LOSS_KEY = "d_real_a"
D_A_FAKE_LOSS_KEY = "d_fake_a"
D_B_REAL_LOSS_KEY = "d_real_b"
D_B_FAKE_LOSS_KEY = "d_fake_b"

#DARK CHANNEL FILTER SIZE
DC_FILTER_SIZE = 1

# Set random seed for reproducibility
manualSeed = 999

# Number of training epochs
num_epochs = 500

test_display_size = 8
display_size = 16 #must not be larger than batch size
batch_size = 8
infer_size = 16

brightness_enhance = 1.0
contrast_enhance = 1.0

#Running on COARE?
is_coare = 0
    