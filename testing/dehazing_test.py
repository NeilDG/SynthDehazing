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
import kornia
from custom_losses import ssim_loss
from loaders.image_dataset import AirlightDataset
from utils import tensor_utils
from utils import plot_utils
from loaders import dataset_loader
from model import dehaze_discriminator as dh
from model import vanilla_cycle_gan as cycle_gan
from model import unet_gan as un
from model import ffa_net as ffa_gan

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


def visualize_haze_equation_albedo(clear_path, depth_path, albedo_path):
    clear_list, depth_list = dataset_loader.assemble_paired_data(clear_path, depth_path, num_image_to_load = 10)
    albedo_list = dataset_loader.assemble_unpaired_data(albedo_path, num_image_to_load = 10)
    print("Reading images in ", albedo_path, depth_path)

    #for i in range(len(albedo_list)):
    for i in range(0, 5):
        albedo_img = cv2.imread(albedo_list[i])
        albedo_img = cv2.cvtColor(albedo_img, cv2.COLOR_BGR2RGB)
        albedo_img = cv2.resize(albedo_img, (256, 256))
        albedo_img = cv2.normalize(albedo_img, dst = None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        depth_img = cv2.imread(depth_list[i])
        depth_img = cv2.cvtColor(depth_img, cv2.COLOR_BGR2GRAY)
        depth_img = cv2.resize(depth_img, (256, 256))
        depth_img = cv2.normalize(depth_img, dst = None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        clear_img = cv2.imread(clear_list[i])
        clear_img = cv2.cvtColor(clear_img, cv2.COLOR_BGR2RGB)
        clear_img = cv2.resize(clear_img, (256, 256))
        clear_img = cv2.normalize(clear_img, dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        tensor_utils.introduce_haze_albedo(clear_img, depth_img, albedo_img)

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
        vutils.make_grid(img_tensor.to(device)[:constants.display_size], nrow=8, padding=2, normalize=True).cpu(),
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

def perform_airlight_predictions(airlight_checkpt_name, albedo_checkpt_name, num_albedo_blocks):
    ABS_PATH_RESULTS = "D:/Users/delgallegon/Documents/GithubProjects/NeuralNets-GenerativeExperiment/results/"
    ABS_PATH_CHECKPOINT = "D:/Users/delgallegon/Documents/GithubProjects/NeuralNets-GenerativeExperiment/checkpoint/"
    PATH_TO_FILE = ABS_PATH_RESULTS + str(airlight_checkpt_name) + ".txt"

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    #A1 = dh.AirlightEstimator_V1(input_nc=3, downsampling_layers = 3, residual_blocks = 7, add_mean = False).to(device)
    #A2 = dh.AirlightEstimator_V1(input_nc=6, downsampling_layers = 3, residual_blocks = 7, add_mean = False).to(device)
    A1 = dh.AirlightEstimator_V2(num_channels = 3, disc_feature_size = 64,  out_features = 3).to(device)
    A2 = dh.AirlightEstimator_V2(num_channels = 6, disc_feature_size = 64, out_features = 3).to(device)
    checkpoint = torch.load(ABS_PATH_CHECKPOINT + airlight_checkpt_name + '.pt')
    A1.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + "A"])
    A2.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + "B"])
    A1.eval()
    A2.eval()
    print("Airlight estimator network loaded")
    print("===================================================")

    #G_albedo = cycle_gan.Generator(n_residual_blocks=8).to(device)
    G_albedo = ffa_gan.FFA(gps=3, blocks=num_albedo_blocks).to(device)
    checkpoint = torch.load(ABS_PATH_CHECKPOINT + albedo_checkpt_name + '.pt')
    G_albedo.load_state_dict(checkpoint[constants.GENERATOR_KEY  + "A"])
    G_albedo.eval()
    print("G albedo network loaded")
    print("===================================================")

    test_loader = dataset_loader.load_airlight_test_dataset(constants.DATASET_ALBEDO_PATH_PSEUDO_TEST, constants.DATASET_CLEAN_PATH_COMPLETE_STYLED_TEST, constants.DATASET_DEPTH_PATH_COMPLETE_TEST, 256, -1)
    average_MSE = [0.0, 0.0, 0.0]

    count = 0
    with open(PATH_TO_FILE, "w") as f, torch.no_grad():
        for i, (test_data) in enumerate(test_loader, 0):
            _, albedo_batch, styled_batch, airlight_batch = test_data
            albedo_batch = albedo_batch.to(device).float()
            styled_batch = styled_batch.to(device).float()
            airlight_batch = airlight_batch.to(device).float()

            mean_batch = torch.full(np.shape(airlight_batch), AirlightDataset.atmosphere_mean()).to(device)

            airlight_shape = np.shape(airlight_batch.cpu().numpy())[0]
            for j in range(airlight_shape):
                mean_input = torch.unsqueeze(mean_batch[j], 0)
                styled_input = torch.unsqueeze(styled_batch[j], 0)
                airlight_input = torch.unsqueeze(airlight_batch[j], 0)
                albedo_input = torch.unsqueeze(albedo_batch[j], 0)

                mean_error = mse(mean_input, airlight_input).item()
                A1_error = mse(A1(styled_input), airlight_input).item()
                A2_error = mse(A2(torch.cat([styled_input, albedo_input], 1)), airlight_input).item()

                average_MSE[0] += mean_error
                average_MSE[1] += A1_error
                average_MSE[2] += A2_error

                count = count + 1
                #print("Errors: ", mean_error, A1_error, A2_error, file = f)
                print("Errors: ", mean_error, A1_error, A2_error)

        average_MSE[0] = np.round(average_MSE[0] / count * 1.0, 5)
        average_MSE[1] = np.round(average_MSE[1] / count * 1.0, 5)
        average_MSE[2] = np.round(average_MSE[2] / count * 1.0, 5)
        print("Overall MSE for Mean: " + str(average_MSE[0]), file=f)
        print("Overall MSE for A1: " + str(average_MSE[1]), file=f)
        print("Overall MSE for A2: " + str(average_MSE[2]), file=f)

def perform_transmission_map_estimation(model_checkpt_name, is_unet):
    ABS_PATH_RESULTS = "D:/Users/delgallegon/Documents/GithubProjects/NeuralNets-GenerativeExperiment/results/"
    ABS_PATH_CHECKPOINT = "D:/Users/delgallegon/Documents/GithubProjects/NeuralNets-GenerativeExperiment/checkpoint/"
    PATH_TO_FILE = ABS_PATH_RESULTS + str(model_checkpt_name) + ".txt"

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    if(is_unet == True):
        G_A = un.UnetGenerator(input_nc=3, output_nc=1, num_downs=8).to(device)
    else:
        G_A = cycle_gan.Generator(input_nc=3, output_nc=1, n_residual_blocks=8).to(device)

    checkpoint = torch.load(ABS_PATH_CHECKPOINT + model_checkpt_name + '.pt')
    G_A.load_state_dict(checkpoint[constants.GENERATOR_KEY + "A"])
    G_A.eval()

    print("G transmission network loaded")
    print("===================================================")

    test_loader = dataset_loader.load_transmission_albedo_dataset(constants.DATASET_ALBEDO_PATH_COMPLETE_3, constants.DATASET_ALBEDO_PATH_PSEUDO_3, constants.DATASET_DEPTH_PATH_COMPLETE_3, False, 128, 50000)

    ave_losses = [0.0, 0.0, 0.0, 0.0]
    count = 0
    with open(PATH_TO_FILE, "w") as f, torch.no_grad():
        for i, (test_data) in enumerate(test_loader, 0):
            _, rgb_batch, transmission_batch = test_data
            rgb_tensor = rgb_batch.to(device).float()
            transmission_batch = transmission_batch.to(device).float()
            transmission_like = G_A(rgb_tensor)

            transmission_batch = (transmission_batch * 0.5) + 0.5 #remove tanh normalization
            transmission_like = (transmission_like * 0.5) + 0.5

            #use common losses in torch and kornia
            l1_loss = nn.L1Loss()
            l2_loss = nn.MSELoss()
            ssim = ssim_loss.SSIM()

            mae = l1_loss(transmission_like, transmission_batch).cpu().item()
            mse = l2_loss(transmission_like, transmission_batch).cpu().item()
            psnr = kornia.losses.psnr(transmission_batch, transmission_like, torch.max(transmission_batch).item()).cpu().item()
            ssim_val = ssim(transmission_batch, transmission_like).cpu().item()

            ave_losses[0] += mae
            ave_losses[1] += mse
            ave_losses[2] += psnr
            ave_losses[3] += ssim_val

            count = count + 1
            print("MAE: ", np.round(mae, 5))
            print("MSE: ", np.round(mse, 5))
            print("PSNR: ", np.round(psnr, 5))
            print("SSIM: ", np.round(ssim_val, 5))
            print("MAE: ", np.round(mae, 5), file = f)
            print("MSE: ", np.round(mse, 5), file = f)
            print("PSNR: ", np.round(psnr, 5), file = f)
            print("SSIM: ", np.round(ssim_val, 5), file = f)

        ave_losses[0] = np.round(ave_losses[0] / count * 1.0, 5)
        ave_losses[1] = np.round(ave_losses[1] / count * 1.0, 5)
        ave_losses[2] = np.round(ave_losses[2] / count * 1.0, 5)
        ave_losses[3] = np.round(ave_losses[3] / count * 1.0, 5)

        print("Overall MAE: " + str(ave_losses[0]), file=f)
        print("Overall MSE: " + str(ave_losses[1]), file=f)
        print("Overall PSNR: " + str(ave_losses[2]), file=f)
        print("Overall SSIM: " + str(ave_losses[3]), file=f)

def perform_albedo_reconstruction(model_checkpt_name, num_blocks):
    ABS_PATH_RESULTS = "D:/Users/delgallegon/Documents/GithubProjects/NeuralNets-GenerativeExperiment/results/"
    ABS_PATH_CHECKPOINT = "D:/Users/delgallegon/Documents/GithubProjects/NeuralNets-GenerativeExperiment/checkpoint/"
    PATH_TO_FILE = ABS_PATH_RESULTS + str(model_checkpt_name) + ".txt"

    print("Loading: ", ABS_PATH_CHECKPOINT + model_checkpt_name)
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    #G_A = cycle_gan.Generator(n_residual_blocks=16).to(device)
    G_A = ffa_gan.FFA(gps=3, blocks=num_blocks).to(device)
    checkpoint = torch.load(ABS_PATH_CHECKPOINT + model_checkpt_name + '.pt')
    G_A.load_state_dict(checkpoint[constants.GENERATOR_KEY + "A"])
    G_A.eval()

    print("G transmission network loaded")
    print("===================================================")

    test_loader = dataset_loader.load_color_albedo_test_dataset(constants.DATASET_CLEAN_PATH_COMPLETE_STYLED_3, constants.DATASET_ALBEDO_PATH_COMPLETE_3, constants.DATASET_DEPTH_PATH_COMPLETE_3, constants.infer_size, 100000)
    count = 0
    ave_losses = [0.0, 0.0, 0.0, 0.0]
    with open(PATH_TO_FILE, "w") as f, torch.no_grad():
        for i, (test_data) in enumerate(test_loader, 0):
            _, hazy_batch, albedo_batch = test_data
            hazy_batch = hazy_batch.to(device).float()
            albedo_batch = albedo_batch.to(device).float()

            albedo_like = G_A(hazy_batch)
            # inferred albedo image appears darker --> adjust brightness and contrast of albedo image
            #albedo_like = kornia.adjust_brightness(albedo_like, 0.6)

            albedo_batch = (albedo_batch * 0.5) + 0.5  # remove tanh normalization
            albedo_like = (albedo_like * 0.5) + 0.5

            #show_images(albedo_batch, "Albedo GT")
            #show_images(albedo_like, "Albedo Like")

            # use common losses in torch and kornia
            l1_loss = nn.L1Loss()
            l2_loss = nn.MSELoss()
            ssim = ssim_loss.SSIM()

            mae = l1_loss(albedo_like, albedo_batch).cpu().item()
            mse = l2_loss(albedo_like, albedo_batch).cpu().item()
            psnr = kornia.losses.psnr(albedo_like, albedo_batch, torch.max(albedo_batch).item()).cpu().item()
            ssim_val = ssim(albedo_like, albedo_batch).cpu().item()

            ave_losses[0] += mae
            ave_losses[1] += mse
            ave_losses[2] += psnr
            ave_losses[3] += ssim_val

            count = count + 1
            print("MAE: ", np.round(mae, 5))
            print("MSE: ", np.round(mse, 5))
            print("PSNR: ", np.round(psnr, 5))
            print("SSIM: ", np.round(ssim_val, 5))
            # print("MAE: ", np.round(mae, 5), file=f)
            # print("MSE: ", np.round(mse, 5), file=f)
            # print("PSNR: ", np.round(psnr, 5), file=f)
            # print("SSIM: ", np.round(ssim_val, 5), file=f)

        ave_losses[0] = np.round(ave_losses[0] / count * 1.0, 5)
        ave_losses[1] = np.round(ave_losses[1] / count * 1.0, 5)
        ave_losses[2] = np.round(ave_losses[2] / count * 1.0, 5)
        ave_losses[3] = np.round(ave_losses[3] / count * 1.0, 5)

        print("Overall MAE: " + str(ave_losses[0]), file=f)
        print("Overall MSE: " + str(ave_losses[1]), file=f)
        print("Overall PSNR: " + str(ave_losses[2]), file=f)
        print("Overall SSIM: " + str(ave_losses[3]), file=f)

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
            # img_b = cv2.normalize(img_b, dst=None, alpha=0
            # , beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
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
    perform_airlight_predictions("airlight_estimator_v1.05_2", "albedo_transfer_v1.04_1", 18)
    #perform_airlight_predictions("airlight_estimator_v1.04_2", "albedo_transfer_v1.04_1", 18)
    #perform_airlight_predictions("airlight_estimator_v1.04_3", "albedo_transfer_v1.04_1", 18)
    #perform_transmission_map_estimation("transmission_albedo_estimator_v1.04_2")
    #perform_transmission_map_estimation("transmission_albedo_estimator_v1.04_3")
    # perform_transmission_map_estimation("transmission_albedo_estimator_v1.04_4")
    # perform_transmission_map_estimation("transmission_albedo_estimator_v1.04_5")
    # perform_transmission_map_estimation("transmission_albedo_estimator_v1.04_6")
    # perform_transmission_map_estimation("transmission_albedo_estimator_v1.04_7")
    #perform_transmission_map_estimation("transmission_albedo_estimator_v1.06_1", False)

    #perform_albedo_reconstruction("albedo_transfer_v1.04_4", 16)
    # perform_albedo_reconstruction("albedo_transfer_v1.04_5", 20)
    # perform_albedo_reconstruction("albedo_transfer_v1.04_6", 24)
    # perform_albedo_reconstruction("albedo_transfer_v1.04_7", 28)


if __name__=="__main__": 
    main()   