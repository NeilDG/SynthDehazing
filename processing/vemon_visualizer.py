# -*- coding: utf-8 -*-
"""
Created on Sat May 16 21:44:51 2020

@author: delgallegon
"""
from loaders import dataset_loader
import matplotlib.pyplot as plt
import constants
import torch
import torch.nn as nn
import numpy as np
import cv2
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torchvision.models as models
from utils import tensor_utils
from utils import plot_utils
from loaders import dataset_loader
from model import dehaze_discriminator as dh
from model import vanilla_cycle_gan as cycle_gan

def visualize_color_distribution(img_dir_path_a, img_dir_path_b):
    img_list = dataset_loader.assemble_unpaired_data(img_dir_path_a, 500)
    rgb_list = np.empty((len(img_list), 3), dtype=np.float64)
    print("Reading images in ", img_dir_path_a)

    for i in range(len(img_list)):
        img = cv2.imread(img_list[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        red_channel = np.reshape(img[:, :, 0], -1)
        blue_channel = np.reshape(img[:, :, 1], -1)
        green_channel = np.reshape(img[:, :, 2], -1)

        rgb_list[i, 0] = np.round(np.mean(red_channel), 4)
        rgb_list[i, 1] = np.round(np.mean(blue_channel), 4)
        rgb_list[i, 2] = np.round(np.mean(green_channel), 4)

    plt.scatter(x=np.arange(0, len(img_list)), y = rgb_list[:,0], color = (1, 0, 0))
    plt.scatter(x=np.arange(0, len(img_list)), y = rgb_list[:,1], color=(0, 1, 0))
    plt.scatter(x=np.arange(0, len(img_list)), y = rgb_list[:,2], color=(0, 0, 1))
    #plt.show()

    img_list = dataset_loader.assemble_unpaired_data(img_dir_path_b, 500)
    rgb_list = np.empty((len(img_list), 3), dtype=np.float64)
    print("Reading images in ", img_dir_path_b)

    for i in range(len(img_list)):
        img = cv2.imread(img_list[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        red_channel = np.reshape(img[:, :, 0], -1)
        blue_channel = np.reshape(img[:, :, 1], -1)
        green_channel = np.reshape(img[:, :, 2], -1)

        #print(np.shape(red_channel), np.shape(blue_channel), np.shape(green_channel))

        rgb_list[i, 0] = np.round(np.mean(red_channel), 4)
        rgb_list[i, 1] = np.round(np.mean(blue_channel), 4)
        rgb_list[i, 2] = np.round(np.mean(green_channel), 4)

    plt.scatter(x=np.arange(0, len(img_list)), y=rgb_list[:, 0], color=(0.5, 0, 0))
    plt.scatter(x=np.arange(0, len(img_list)), y=rgb_list[:, 1], color=(0, 0.5, 0))
    plt.scatter(x=np.arange(0, len(img_list)), y=rgb_list[:, 2], color=(0, 0, 0.5))

    plt.show()

def visualize_edge_distribution(path_a):
    img_list = dataset_loader.assemble_unpaired_data(path_a, 500)
    edge_list = np.empty((len(img_list), 1), dtype=np.float64)
    print("Reading images in ", path_a)

    for i in range(len(edge_list)):
        img = cv2.imread(img_list[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
        sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
        sobel_img = sobel_x + sobel_y
        sobel_quality = np.round(np.linalg.norm(sobel_img), 4)

        edge_list[i] =  sobel_quality

    plt.hist(edge_list)


def visualize_haze_equation(path_a, depth_path, path_b):
    img_list, depth_list = dataset_loader.assemble_paired_data(path_a, depth_path, num_image_to_load = 10)
    clear_list = dataset_loader.assemble_unpaired_data(path_b, num_image_to_load = 10)
    print("Reading images in ", path_a, depth_path)

    for i in range(len(img_list)):
        img = cv2.imread(img_list[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256))
        img = cv2.normalize(img, dst = None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        depth_img = cv2.imread(depth_list[i])
        depth_img = cv2.cvtColor(depth_img, cv2.COLOR_BGR2GRAY)
        depth_img = cv2.resize(depth_img, (256, 256))
        depth_img = cv2.normalize(depth_img, dst = None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        clear_img = cv2.imread(clear_list[i])
        clear_img = cv2.cvtColor(clear_img, cv2.COLOR_BGR2RGB)
        clear_img = cv2.resize(clear_img, (256, 256))
        clear_img = cv2.normalize(clear_img, dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        #tensor_utils.compare_transmissions(img, depth_img)
        #tensor_utils.perform_dehazing_equation(img, depth_img)
        #tensor_utils.perform_custom_dehazing_equation(img, clear_img)
        tensor_utils.introduce_haze(img, clear_img, depth_img)
        #tensor_utils.mask_haze(img, clear_img, depth_img)

def show_images(img_tensor, caption):
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    plt.figure(figsize=(8,2))
    plt.axis("off")
    plt.title(caption)
    plt.imshow(np.transpose(
        vutils.make_grid(img_tensor.to(device)[:constants.batch_size], nrow=8, padding=2, normalize=True).cpu(),
        (1, 2, 0)))
    plt.show()


def visualize_feature_distribution(path_a, path_b):
    loader_a = dataset_loader.load_color_test_dataset(path_a, batch_size=16, num_image_to_load=45)
    loader_b = dataset_loader.load_color_test_dataset(path_b, batch_size=16, num_image_to_load=45)

    _, gray_batch_a, rgb_batch_a = next(iter(loader_a))
    _, gray_batch_b, rgb_batch_b = next(iter(loader_b))
    show_images(rgb_batch_a, "A")
    show_images(rgb_batch_b, "B")

    loader_a = dataset_loader.load_color_test_dataset(path_a, batch_size = 1, num_image_to_load = 45)
    loader_b = dataset_loader.load_color_test_dataset(path_b, batch_size = 1, num_image_to_load = 45)

    vgg_model = models.vgg16(pretrained = True)
    vgg_model = nn.Sequential(*list(vgg_model.children())[:-1])

    norm_result_a = []
    norm_result_b = []

    for i, (data_a, data_b) in enumerate(zip(loader_a, loader_b)):
        _, gray_batch_a, rgb_batch_a = data_a
        _, gray_batch_b, rgb_batch_b = data_b

        with torch.no_grad():
            activation_a = vgg_model(rgb_batch_a)
            activation_b = vgg_model(rgb_batch_b)

            norm_result_a.append(np.linalg.norm(activation_a))
            norm_result_b.append(np.linalg.norm(activation_b))

            print(norm_result_a[len(norm_result_a) - 1], norm_result_b[len(norm_result_b) - 1])
            plt.scatter(x=np.arange(0, len(norm_result_a)), y=norm_result_a, color=(0.5, 0, 0))
            plt.scatter(x=np.arange(0, len(norm_result_b)), y=norm_result_b, color=(0, 0.5, 0))

    plt.show()

def visualize_img_to_light_correlation():
    img_list = dataset_loader.assemble_unpaired_data(constants.DATASET_HAZY_PATH_COMPLETE, num_image_to_load=100)
    print("Reading images in ", img_list)

    rgb_list = np.empty((len(img_list), 3), dtype=np.float64)
    light_list = np.empty((len(img_list), 2), dtype=np.float64)
    for i in range(len(img_list)):
        img_id = int(img_list[i].split("/")[3].split(".")[0].split("_")[1]) + 5 #offset

        light_path = constants.DATASET_LIGHTCOORDS_PATH_COMPLETE + "lights_"+str(img_id)+".txt"
        print("Lights path: ", light_path)

        light_file = open(light_path, "r")
        light_string = light_file.readline()
        light_vector = str.split(light_string, ",")
        light_vector = [float(light_vector[0]), float(light_vector[1])]
        print("Light vector: ", light_vector)

        img = cv2.imread(img_list[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        red_channel = np.reshape(img[:, :, 0], -1)
        blue_channel = np.reshape(img[:, :, 1], -1)
        green_channel = np.reshape(img[:, :, 2], -1)

        rgb_list[i, 0] = np.round(np.mean(red_channel), 4)
        rgb_list[i, 1] = np.round(np.mean(blue_channel), 4)
        rgb_list[i, 2] = np.round(np.mean(green_channel), 4)

        light_list[i, 0] = np.round(light_vector[0], 4)
        light_list[i, 1] = np.round(light_vector[1], 4)

    #x_list = np.random.random(len(img_list))
    plt.scatter(x=np.arange(0, len(img_list)), y=rgb_list[:, 0], color=(1, 0, 0))
    plt.scatter(x=np.arange(0, len(img_list)), y=rgb_list[:, 1], color=(0, 1, 0))
    plt.scatter(x=np.arange(0, len(img_list)), y=rgb_list[:, 2], color=(0, 0, 1))
    plt.scatter(x=np.arange(0, len(img_list)), y=light_list[:, 0], color=(1, 1, 0))
    plt.scatter(x=np.arange(0, len(img_list)), y=light_list[:, 1], color=(0, 1, 1))
    plt.show()

def mse(predictions, targets):
    return ((predictions - targets) ** 2).mean()

def perform_lightcoord_predictions(model_checkpt_name):
    img_list = dataset_loader.assemble_unpaired_data(constants.DATASET_CLEAN_PATH_COMPLETE_STYLED, num_image_to_load=1000)
    print("Reading images in ", img_list)

    ABS_PATH_RESULTS = "D:/Users/delgallegon/Documents/GithubProjects/NeuralNets-GenerativeExperiment/results/"
    ABS_PATH_CHECKPOINT = "D:/Users/delgallegon/Documents/GithubProjects/NeuralNets-GenerativeExperiment/checkpoint/"
    #MODEL_CHECKPOINT = "lightcoords_estimator_V1.00_6"
    PATH_TO_FILE = ABS_PATH_RESULTS + str(model_checkpt_name) + ".txt"

    LIGHT_MEAN = [150.29387908289183, 97.35015686388994]
    LIGHT_STD = [108.93871124236733, 72.016361696062]

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    light_estimator = dh.LightCoordsEstimator_V2(input_nc=3, num_layers=4).to(device)
    checkpoint = torch.load(ABS_PATH_CHECKPOINT + model_checkpt_name+ '.pt')
    light_estimator.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY])
    light_estimator.eval()
    print("Light estimator network loaded")
    print("===================================================")

    # load color transfer
    # color_transfer_checkpt = torch.load(ABS_PATH_CHECKPOINT + 'color_transfer_v1.11_2.pt')
    # color_transfer_gan = cycle_gan.Generator(n_residual_blocks=10).to(device)
    # color_transfer_gan.load_state_dict(color_transfer_checkpt[constants.GENERATOR_KEY + "A"])
    # color_transfer_gan.eval()
    # print("Color transfer GAN model loaded.")
    # print("===================================================")

    img_op = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])

    average_mse = 0.0
    with open(PATH_TO_FILE, "w") as f, torch.no_grad():
        for i in range(len(img_list)):
            img_name = img_list[i].split("/")[3].split(".")[0]
            img_id = int(img_name.split("_")[1]) + 5  # offset

            light_path = constants.DATASET_LIGHTCOORDS_PATH_COMPLETE + "lights_" + str(img_id) + ".txt"

            light_file = open(light_path, "r")
            light_string = light_file.readline()
            light_vector = str.split(light_string, ",")
            light_vector = [float(light_vector[0]), float(light_vector[1])]

            img = cv2.imread(img_list[i])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # img_b = cv2.imread(constants.DATASET_DEPTH_PATH_COMPLETE + img_name+".png")
            # img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
            # img_b = cv2.normalize(img_b, dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            # T = tensor_utils.generate_transmission(1 - img_b, np.random.uniform(0.0, 2.5))
            #
            # # formulate hazy img
            # atmosphere = np.random.uniform(0.5, 1.2)
            # hazy_img_like = np.zeros_like(clear_img)
            # T = np.resize(T, np.shape(clear_img[:, :, 0]))
            # hazy_img_like[:, :, 0] = (T * clear_img[:, :, 0]) + atmosphere * (1 - T)
            # hazy_img_like[:, :, 1] = (T * clear_img[:, :, 1]) + atmosphere * (1 - T)
            # hazy_img_like[:, :, 2] = (T * clear_img[:, :, 2]) + atmosphere * (1 - T)
            # img_a = cv2.normalize(hazy_img_like, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            # img_tensor = img_op(img_a).to(device)

            img_tensor = img_op(img).to(device)

            #light_preds = light_estimator(color_transfer_gan(torch.unsqueeze(img_tensor, 0)))
            light_preds = light_estimator(torch.unsqueeze(img_tensor, 0))
            light_preds = torch.squeeze(light_preds).cpu().numpy()

            light_preds[0] = (light_preds[0] * LIGHT_STD[0]) + LIGHT_MEAN[0]
            light_preds[1] = (light_preds[1] * LIGHT_STD[1]) + LIGHT_MEAN[1]

            mse_result = mse(light_vector, light_preds)
            average_mse += mse_result

            print("Img id: " + str(img_id) + " GT light vector: " + str(light_vector) + " Pred light vector: " +str(light_preds)+
                   " MSE: " +str(mse_result), file=f)
            print("Img id: " + str(img_id) + " GT light vector: " + str(light_vector) + " Pred light vector: " + str(light_preds) +
                  " MSE: " + str(mse_result))

        average_mse = np.round(average_mse / len(img_list) * 1.0, 5)
        print("Overall MSE: " + str(average_mse), file = f)



def main():
    # visualize_color_distribution(constants.DATASET_VEMON_PATH_PATCH_32, constants.DATASET_DIV2K_PATH_PATCH)
    # visualize_edge_distribution(constants.DATASET_VEMON_PATH_PATCH_32)
    # visualize_edge_distribution(constants.DATASET_DIV2K_PATH_PATCH)
    # plt.show()
    #
    # visualize_edge_distribution(constants.DATASET_HAZY_PATH_PATCH)
    # visualize_edge_distribution(constants.DATASET_CLEAN_PATH_PATCH)
    # plt.show()

    #visualize_haze_equation(constants.DATASET_HAZY_PATH_COMPLETE, constants.DATASET_DEPTH_PATH_COMPLETE, constants.DATASET_CLEAN_PATH_COMPLETE)
    #visualize_feature_distribution(constants.DATASET
    # _HAZY_PATH_COMPLETE, constants.DATASET_IHAZE_HAZY_PATH_COMPLETE)
    #visualize_img_to_light_correlation()
    #perform_lightcoord_predictions("lightcoords_estimator_V1.00_5")
    #perform_lightcoord_predictions("lightcoords_estimator_V1.00_6")
    perform_lightcoord_predictions("lightcoords_estimator_V1.00_8")
    perform_lightcoord_predictions("lightcoords_estimator_V1.00_9")
    # perform_lightcoord_predictions("lightcoords_estimator_V1.00_13")
    # perform_lightcoord_predictions("lightcoords_estimator_V1.00_14")
    # perform_lightcoord_predictions("lightcoords_estimator_V1.00_15")
    # perform_lightcoord_predictions("lightcoords_estimator_V1.00_16")
    # perform_lightcoord_predictions("lightcoords_estimator_V1.00_17")
    
if __name__=="__main__": 
    main()   