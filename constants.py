# -*- coding: utf-8 -*-

#DATASET_PATH = "E:/Dogs_Dataset/"
DATASET_PATH_NORMAL = "E:/ANU Small Dataset/streetview/"
DATASET_PATH_TOPDOWN = "E:/ANU Small Dataset/satview_polish/"

DATASET_BIRD_NORMAL_PATH = "E:/GTA Bird Dataset/crop_img/"
DATASET_BIRD_HOMOG_PATH = "E:/GTA Bird Dataset/homo_img/"
DATASET_BIRD_GROUND_TRUTH_PATH = "E:/GTA Bird Dataset/bird_gt/"

DATASET_VEMON_FRONT_PATH = "E:/VEMON Dataset/frames/"
DATASET_VEMON_HOMOG_PATH = "E:/VEMON Dataset/homog_crop_frames/"

BIRD_IMAGE_SIZE = (128, 128) #320 x 192 original
# Spatial size of training images. All images will be resized to this
#   size using a transformer.
NORMAL_IMAGE_SIZE = 128
TOPDOWN_IMAGE_SIZE = 128
FIG_SIZE = NORMAL_IMAGE_SIZE / 4

SAVE_FIG_PATH = "D:/Users/delgallegon/Documents/GithubProjects/NeuralNets-GenerativeExperiment/"
TENSORBOARD_PATH = "D:/Users/delgallegon/Documents/GithubProjects/NeuralNets-GenerativeExperiment/train_plot/"

#========================================================================#
GAN_VERSION = "td_v3.00"
GAN_ITERATION = "1"
OPTIMIZER_KEY = "optimizer"
CHECKPATH = 'checkpoint/' + GAN_VERSION +'.pt'
GENERATOR_KEY = "generator"
DISCRIMINATOR_KEY = "discriminator"

 # Set random seed for reproducibility
manualSeed = 999

# Number of training epochs
num_epochs = 40

# Batch size during training
batch_size = 32