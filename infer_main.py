# -*- coding: utf-8 -*-
"""
Class for producing figures
Created on Sat May  2 09:09:21 2020

@author: delgallegon
"""

import torch.utils.data
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from trainers import depth_trainer
from loaders import dataset_loader
from model import vanilla_cycle_gan as cycle_gan
from model import ffa_net as ffa
import constants
from torchvision import transforms
import cv2
from utils import tensor_utils
from utils import dark_channel_prior
import os
import glob
from skimage.metrics import peak_signal_noise_ratio


def show_images(img_tensor, caption):
    plt.figure(figsize=(16, 4))
    plt.axis("off")
    plt.title(caption)
    plt.imshow(np.transpose(
        vutils.make_grid(img_tensor[:constants.batch_size], nrow=8, padding=2, normalize=True).cpu(),
        (1, 2, 0)))
    plt.show()

def get_transform_ops(output_size):
    dark_transform_op = transforms.Compose([transforms.ToPILImage(), 
                                            transforms.Resize(output_size),
                                            transforms.CenterCrop(output_size),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5), (0.5))])
    
    rgb_transform_op = transforms.Compose([transforms.ToPILImage(),
                                           transforms.Resize(output_size),
                                           transforms.CenterCrop(output_size),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    return dark_transform_op, rgb_transform_op

def plot_and_save(item_number, tensor_a, tensor_b):
    LOCATION = os.getcwd() + "/figures/"
    fig, ax = plt.subplots(2, 1)
    fig.set_size_inches((32, 16))
    fig.tight_layout()

    ims = np.transpose(vutils.make_grid(tensor_a, nrow=8, padding=2, normalize = True).cpu(), (1, 2, 0))
    ax[0].set_axis_off()
    ax[0].imshow(ims)

    ims = np.transpose(vutils.make_grid(tensor_b, nrow=8, padding=2, normalize = False).cpu(), (1, 2, 0))
    ax[1].set_axis_off()
    ax[1].imshow(ims)

    plt.subplots_adjust(left=0.06, wspace=0.0, hspace=0.15)
    plt.savefig(LOCATION + "result_" + str(item_number) + ".png")
    plt.show()


def save_img(img_numpy, item_number):
    LOCATION = os.getcwd() + "/figures/"
    im = Image.fromarray(img_numpy)
    im.save(LOCATION + "image_" + str(item_number) + ".png")


def produce_ffa_video(video_path):
    DEHAZER_CHECKPATH = "checkpoint/dehazer_v1.10_2 - stable@12.pt"
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    dehazer = ffa.FFA(gps=3, blocks=19).to(device)
    dehazer_checkpt = torch.load(DEHAZER_CHECKPATH)
    dehazer.load_state_dict(dehazer_checkpt[constants.GENERATOR_KEY])
    OUTPUT_SIZE = (480, 704)

    r_transform_op, rgb_transform_op = get_transform_ops(OUTPUT_SIZE)
    video_name = video_path.split("/")[3].split(".")[0]
    SAVE_PATH = "E:/VEMON Dataset/vemon enhanced/" + video_name + "dehazer_v1.10_2_enhanced.avi"

    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    success = True
    video_out = cv2.VideoWriter(SAVE_PATH, cv2.VideoWriter_fourcc(*"MJPG"), 8.0, (OUTPUT_SIZE[1], OUTPUT_SIZE[0]))
    with torch.no_grad():
        while success:
            success, img = vidcap.read()
            if (success):
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                rgb_tensor = rgb_transform_op(img)
                rgb_tensor = torch.unsqueeze(rgb_tensor, 0).to(device)
                rgb_tensor_clean = dehazer(rgb_tensor)

                result_img = tensor_utils.normalize_to_matplotimg(rgb_tensor_clean.cpu(), 0, 0.5, 0.5)
                # result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                result_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
                video_out.write(result_img)
        video_out.release()


def produce_video(video_path):
    DEHAZER_CHECKPATH = "checkpoint/dehazer_v1.09_2.pt"
    COLORIZER_CHECKPATH = "checkpoint/colorizer_v1.07_2.pt"

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    dehazer = cycle_gan.Generator(input_nc=1, output_nc=1, n_residual_blocks=5).to(device)
    dehazer_checkpt = torch.load(DEHAZER_CHECKPATH)
    dehazer.load_state_dict(dehazer_checkpt[constants.GENERATOR_KEY + "A"])

    colorizer = cycle_gan.Generator(input_nc=3, output_nc=3).to(device)
    colorizer_checkpt = torch.load(COLORIZER_CHECKPATH)
    colorizer.load_state_dict(colorizer_checkpt[constants.GENERATOR_KEY + "A"])

    vidcap = cv2.VideoCapture(video_path)
    success,image = vidcap.read()
    success = True
    
    #OUTPUT_SIZE = (480,704)
    OUTPUT_SIZE = (960, 1408)
    y_transform_op, yuv_transform_op = get_transform_ops(OUTPUT_SIZE)
    
    video_name = video_path.split("/")[3].split(".")[0]
    SAVE_PATH = "E:/VEMON Dataset/vemon enhanced/" + video_name + "dehazer_v1.09_2_enhanced.avi"
    
    video_out = cv2.VideoWriter(SAVE_PATH, cv2.VideoWriter_fourcc(*"MJPG"), 8.0, (OUTPUT_SIZE[1],OUTPUT_SIZE[0]))
    with torch.no_grad():
        while success:
            success,yuv_img = vidcap.read()
            if(success):
                yuv_img = cv2.cvtColor(yuv_img, cv2.COLOR_BGR2YUV)
                y_img = tensor_utils.get_y_channel(yuv_img)
                yuv_tensor = yuv_transform_op(yuv_img)
                yuv_tensor = torch.unsqueeze(yuv_tensor, 0).to(device)
                y_tensor = y_transform_op(y_img)
                y_tensor = torch.unsqueeze(y_tensor, 0).to(device)
                y_tensor_clean = dehazer(y_tensor)

                (y, u, v) = torch.chunk(yuv_tensor.transpose(0, 1), 3)
                input_tensor = torch.cat((y_tensor_clean.transpose(0, 1), u, v)).transpose(0, 1)
                input_tensor_clean = colorizer(input_tensor)

                input_tensor_clean = tensor_utils.yuv_to_rgb(input_tensor_clean).cpu()
                result_img = tensor_utils.normalize_to_matplotimg(input_tensor_clean, 0, 0.5, 0.5)
                #result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                result_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
                video_out.write(result_img)
        video_out.release()


def color_transfer():
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    # load color transfer
    color_transfer_checkpt = torch.load('checkpoint/color_transfer_v1.11_1 - stable.pt')
    color_transfer_gan = cycle_gan.Generator(n_residual_blocks=10).to(device)
    color_transfer_gan.load_state_dict(color_transfer_checkpt[constants.GENERATOR_KEY + "A"])
    print("Color transfer GAN model loaded.")
    print("===================================================")
    
    dataloader = dataset_loader.load_test_dataset(constants.DATASET_HAZY_PATH_COMPLETE, constants.DATASET_PLACES_PATH, constants.infer_size, -1)
    
    # Plot some training images
    name_batch, dirty_batch, clean_batch = next(iter(dataloader))
    plt.figure(figsize=constants.FIG_SIZE)
    plt.axis("off")
    plt.title("Training - Old Images")
    plt.imshow(np.transpose(vutils.make_grid(dirty_batch.to(device)[:constants.infer_size], nrow = 8, padding=2, normalize=True).cpu(),(1,2,0)))
    plt.show()
    
    plt.figure(figsize=constants.FIG_SIZE)
    plt.axis("off")
    plt.title("Training - New Images")
    plt.imshow(np.transpose(vutils.make_grid(clean_batch.to(device)[:constants.infer_size], nrow = 8, padding=2, normalize=True).cpu(),(1,2,0)))
    plt.show()
    
    item_number = 0
    for i, (name, dirty_batch, clean_batch) in enumerate(dataloader, 0):
        with torch.no_grad():
            input_tensor = dirty_batch.to(device)
            item_number = item_number + 1
            result = color_transfer_gan(input_tensor)
            show_images(input_tensor, "Input images: " +str(item_number))
            show_images(result, "Color transfer: " + str(item_number))

def produce_video_batch():
    VIDEO_FOLDER_PATH = "E:/VEMON Dataset/vemon videos/"
    #VIDEO_FOLDER_PATH = "E:/VEMON Dataset/mmda videos/"
    video_list = os.listdir(VIDEO_FOLDER_PATH)
    for i in range(len(video_list)):
        video_path = VIDEO_FOLDER_PATH + video_list[i]
        print(video_path)
        produce_ffa_video(video_path)

def benchmark_reside():
    HAZY_PATH = "E:/Hazy Dataset Benchmark/RESIDE-Unannotated/"
    SAVE_PATH = "results/"
    MODEL_CHECKPOINT = "transmission_estimator_v1.01_1"
    BENCHMARK_PATH = "results/metrics - " + str(MODEL_CHECKPOINT) + ".txt"

    FFA_RESULTS_PATH = "results/FFA Net - Results - RESIDE-3/"
    GRID_DEHAZE_RESULTS_PATH = "results/GridDehazeNet - Results - RESIDE-3/"
    CYCLE_DEHAZE_PATH = "results/CycleDehaze - Results - RESIDE-3/"

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    hazy_list = glob.glob(HAZY_PATH + "*.jpeg")
    ffa_list = glob.glob(FFA_RESULTS_PATH + "*.jpg")
    grid_list = glob.glob(GRID_DEHAZE_RESULTS_PATH + "*.jpeg")
    cycle_dh_list = glob.glob(CYCLE_DEHAZE_PATH + "*.jpeg")

    print("Found images: ", len(hazy_list), len(ffa_list), len(grid_list), len(cycle_dh_list))

    gray_img_op = transforms.Compose([transforms.ToPILImage(),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5), (0.5))])

    rgb_img_op = transforms.Compose([transforms.ToPILImage(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    transmission_G = cycle_gan.Generator(input_nc=3, output_nc=1, n_residual_blocks=6).to(device)
    checkpt = torch.load('checkpoint/' + MODEL_CHECKPOINT + ".pt")
    transmission_G.load_state_dict(checkpt[constants.GENERATOR_KEY + "A"])
    print("Transmission GAN model loaded.")

    FIG_ROWS = 6;
    FIG_COLS = 7
    FIG_WIDTH = 20
    FIG_HEIGHT = 20
    fig, ax = plt.subplots(ncols=FIG_COLS, nrows=FIG_ROWS, constrained_layout=True, sharex=True)
    fig.set_size_inches(FIG_WIDTH, FIG_HEIGHT)
    column = 0
    fig_num = 0
    count = 0

    for i, (hazy_path, ffa_path, grid_path, cycle_dh_path) in enumerate(zip(hazy_list, ffa_list, grid_list, cycle_dh_list)):
        with torch.no_grad():
            count = count + 1
            img_name = hazy_path.split("\\")[1]
            hazy_img = cv2.imread(hazy_path)
            #hazy_img = cv2.resize(hazy_img, (int(np.shape(hazy_img)[1] / 2), int(np.shape(hazy_img)[0] / 2)))
            hazy_img = cv2.resize(hazy_img, (512, 512))

            input_tensor = rgb_img_op(cv2.cvtColor(hazy_img, cv2.COLOR_BGR2RGB)).to(device)
            transmission_img = transmission_G(torch.unsqueeze(input_tensor, 0))
            transmission_img = torch.squeeze(transmission_img).cpu().numpy()

            # remove 0.5 normalization for dehazing equation
            transmission_img = ((transmission_img * 0.5) + 0.5)
            #transmission_img = transmission_img + 0.7
            hazy_img = ((hazy_img * 0.5) + 0.5)

            dark_channel = dark_channel_prior.get_dark_channel(hazy_img, 15)
            dcp_transmission = dark_channel_prior.estimate_transmission(hazy_img, dark_channel_prior.estimate_atmosphere(hazy_img, dark_channel),
                                                                        dark_channel)
            # DCP is not 0-1 range
            dcp_transmission = cv2.normalize(dcp_transmission, dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,
                                             dtype=cv2.CV_32F)
            transmission_blend = dcp_transmission * 0.5 + transmission_img * 0.5

            dcp_clear_img = dark_channel_prior.perform_dcp_dehaze(hazy_img)
            clear_img = tensor_utils.perform_dehazing_equation_with_transmission(hazy_img, transmission_blend, 0.4)
            # clear_img = tensor_utils.perform_dehazing_equation_with_transmission(hazy_img, transmission_img, 0.3)
            #clear_img = tensor_utils.refine_dehaze_img(hazy_img, clear_img, transmission_blend)

            ffa_img = cv2.imread(ffa_path)
            ffa_img = cv2.resize(ffa_img, (int(np.shape(hazy_img)[1]), int(np.shape(hazy_img)[0])))

            grid_img = cv2.imread(grid_path)
            grid_img = cv2.resize(grid_img, (int(np.shape(hazy_img)[1]), int(np.shape(hazy_img)[0])))

            cycle_dehaze_img = cv2.imread(cycle_dh_path)
            cycle_dehaze_img = cv2.resize(cycle_dehaze_img, (int(np.shape(hazy_img)[1]), int(np.shape(hazy_img)[0])))

            # normalize images
            hazy_img = cv2.normalize(hazy_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            clear_img = cv2.normalize(clear_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            dcp_clear_img = cv2.normalize(dcp_clear_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            ffa_img = cv2.normalize(ffa_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            grid_img = cv2.normalize(grid_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            cycle_dehaze_img = cv2.normalize(cycle_dehaze_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

            # make images compatible with matplotlib
            hazy_img = cv2.cvtColor(hazy_img, cv2.COLOR_BGR2RGB)
            clear_img = cv2.cvtColor(clear_img, cv2.COLOR_BGR2RGB)
            dcp_clear_img = cv2.cvtColor(dcp_clear_img, cv2.COLOR_BGR2RGB)
            ffa_img = cv2.cvtColor(ffa_img, cv2.COLOR_BGR2RGB)
            grid_img = cv2.cvtColor(grid_img, cv2.COLOR_BGR2RGB)
            cycle_dehaze_img = cv2.cvtColor(cycle_dehaze_img, cv2.COLOR_BGR2RGB)

            ax[0, column].imshow(hazy_img)
            ax[0, column].axis('off')
            ax[1, column].imshow(dcp_clear_img)
            ax[1, column].axis('off')
            ax[2, column].imshow(cycle_dehaze_img)
            ax[2, column].axis('off')
            ax[3, column].imshow(ffa_img)
            ax[3, column].axis('off')
            ax[4, column].imshow(grid_img)
            ax[4, column].axis('off')
            ax[5, column].imshow(clear_img)
            ax[5, column].axis('off')
            column = column + 1

            if (column == FIG_COLS):
                fig_num = fig_num + 1
                file_name = SAVE_PATH + "fig_" + str(fig_num) + "_" + MODEL_CHECKPOINT + ".jpg"
                plt.savefig(file_name)
                #plt.show()

                # create new figure
                fig, ax = plt.subplots(ncols=FIG_COLS, nrows=FIG_ROWS, constrained_layout=True, sharex=True)
                fig.set_size_inches(FIG_WIDTH, FIG_HEIGHT)
                column = 0

def visdom_preview():
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    gt = depth_trainer.DepthTrainer(constants.TRANSMISSION_VERSION, constants.ITERATION, device, 0.0002, 0.0002)
    checkpoint = torch.load(constants.TRANSMISSION_ESTIMATOR_CHECKPATH)
    start_epoch = checkpoint['epoch'] + 1
    iteration = checkpoint['iteration'] + 1
    gt.load_saved_state(iteration, checkpoint, constants.GENERATOR_KEY, constants.DISCRIMINATOR_KEY,
                        constants.OPTIMIZER_KEY)

    print("Loaded checkpt: %s Current epoch: %d" % (constants.TRANSMISSION_ESTIMATOR_CHECKPATH, start_epoch))
    print("===================================================")

    ohaze_loader = dataset_loader.load_transmision_test_dataset(constants.DATASET_OHAZE_HAZY_PATH_COMPLETE, constants.display_size, 500)
    reside_loader = dataset_loader.load_transmision_test_dataset(constants.DATASET_RESIDE_TEST_PATH_COMPLETE, constants.display_size, 500)

    __, view_rgb_batch_1, view_gray_batch_1 = next(iter(ohaze_loader))
    ___, view_rgb_batch_2, view_gray_batch_2 = next(iter(reside_loader))

    view_rgb_batch_1 = view_rgb_batch_1.to(device).float()
    view_gray_batch_1 = view_gray_batch_1.to(device).float()
    gt.visdom_plot_test_image(view_rgb_batch_1, view_gray_batch_1, 2)

    view_rgb_batch_2 = view_rgb_batch_2.to(device).float()
    view_gray_batch_2 = view_gray_batch_2.to(device).float()
    gt.visdom_plot_test_image(view_rgb_batch_2, view_gray_batch_2, 3)

def main():
    VERSION = "dehazer_v1.08"
    ITERATION = "1"
    CHECKPATH = 'checkpoint/' + VERSION + "_" + ITERATION +'.pt'
    
    #produce_video_batch()
    #benchmark_ohaze()
    #benchmark_reside()
    #color_transfer()
    #remove_haze_by_transmission(constants.DATASET_IHAZE_HAZY_PATH_COMPLETE)
    visdom_preview()

#FIX for broken pipe num_workers issue.
if __name__=="__main__": 
    main()        
        
        