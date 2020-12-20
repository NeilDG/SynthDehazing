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
from torch import nn
from PIL import Image

from loaders import dataset_loader
from trainers import denoise_net_trainer
from trainers import div2k_trainer
from trainers import dehaze_trainer
from model import vanilla_cycle_gan as cycle_gan
from model import style_transfer_gan as color_gan
from model import ffa_net as ffa
import constants
from torchvision import transforms
import cv2
from utils import tensor_utils
import os
import glob
from skimage.measure import compare_ssim
from skimage.measure import compare_mse
from skimage.measure import compare_nrmse

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

def benchmark():
    HAZY_PATH = "E:/Hazy Dataset Benchmark/O-HAZE/hazy/"
    GT_PATH = "E:/Hazy Dataset Benchmark/O-HAZE/GT/"
    FFA_RESULTS_PATH = "results/FFA Net - Results - OHaze/"
    SAVE_PATH = "results/"
    BENCHMARK_PATH = "results/metrics.txt"
    MODEL_CHECKPOINT = "depth_estimator_v1.00_2"

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    hazy_list = glob.glob(HAZY_PATH + "*.jpg")
    gt_list = glob.glob(GT_PATH + "*.jpg")
    ffa_list = glob.glob(FFA_RESULTS_PATH + "*.png")
    print(ffa_list)

    gray_img_op = transforms.Compose([transforms.ToPILImage(),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5), (0.5))])

    rgb_image_op = transforms.Compose([transforms.ToPILImage(),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    transmission_G = cycle_gan.Generator(input_nc=1, output_nc=1, n_residual_blocks=6).to(device)
    checkpt = torch.load('checkpoint/' + MODEL_CHECKPOINT + ".pt")
    transmission_G.load_state_dict(checkpt[constants.GENERATOR_KEY + "A"])
    print("Transmission GAN model loaded.")

    FIG_ROWS = 4;
    FIG_COLS = 6
    fig, ax = plt.subplots(ncols=FIG_COLS, nrows=FIG_ROWS, constrained_layout=True, sharex=True)
    fig.set_size_inches(30, 10)
    column = 0
    fig_num = 0
    average_SSIM = [0.0, 0.0]
    count = 0

    with open(BENCHMARK_PATH, "w") as f:
        for i, (hazy_path, gt_path, ffa_path) in enumerate(zip(hazy_list, gt_list, ffa_list)):
            with torch.no_grad():
                count = count + 1
                img_name = hazy_path.split("\\")[1]
                hazy_img = cv2.imread(hazy_path)
                hazy_img = cv2.resize(hazy_img, (int(np.shape(hazy_img)[1] / 4), int(np.shape(hazy_img)[0] / 4)))
                gt_img = cv2.imread(gt_path)
                gt_img = cv2.resize(gt_img, (int(np.shape(gt_img)[1] / 4), int(np.shape(gt_img)[0] / 4)))
                ffa_img = cv2.imread(ffa_path)

                input_tensor = gray_img_op(cv2.cvtColor(hazy_img, cv2.COLOR_BGR2GRAY)).to(device)
                transmission_img = transmission_G(torch.unsqueeze(input_tensor, 0))
                transmission_img = torch.squeeze(transmission_img).cpu().numpy()

                # remove 0.5 normalization for dehazing equation
                transmission_img = ((transmission_img * 0.5) + 0.5)
                hazy_img = ((hazy_img * 0.5) + 0.5)

                clear_img = tensor_utils.perform_dehazing_equation_with_transmission(hazy_img, transmission_img, 0.8)

                #normalize images
                hazy_img = cv2.normalize(hazy_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                clear_img = cv2.normalize(clear_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8U)
                ffa_img = cv2.normalize(ffa_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                gt_img = cv2.normalize(gt_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

                #make images compatible with matplotlib
                hazy_img = cv2.cvtColor(hazy_img, cv2.COLOR_BGR2RGB)
                clear_img = cv2.cvtColor(clear_img, cv2.COLOR_BGR2RGB)
                ffa_img = cv2.cvtColor(ffa_img, cv2.COLOR_BGR2RGB)
                gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)

                # measure SSIM
                SSIM = np.round(compare_ssim(ffa_img, gt_img, multichannel=True), 4)
                print("[FFA-Net] SSIM of ", img_name, " : ", SSIM, file=f)
                average_SSIM[0] += SSIM

                print(file = f)

                SSIM = np.round(compare_ssim(clear_img, gt_img, multichannel=True), 4)
                print("[Ours] SSIM of " ,img_name," : ", SSIM, file = f)
                average_SSIM[1] += SSIM

                ax[0, column].imshow(hazy_img);
                ax[0, column].axis('off')
                ax[1, column].imshow(ffa_img);
                ax[1, column].axis('off')
                ax[2, column].imshow(clear_img);
                ax[2, column].axis('off')
                ax[3, column].imshow(gt_img);
                ax[3, column].axis('off')
                column = column + 1

                if (column == FIG_COLS):
                    fig_num = fig_num + 1
                    file_name = SAVE_PATH + "fig_" + str(fig_num) + "_" + MODEL_CHECKPOINT + ".jpg"
                    plt.savefig(file_name)
                    plt.show()

                    # create new figure
                    fig, ax = plt.subplots(ncols=FIG_COLS, nrows=FIG_ROWS, constrained_layout=True, sharex=True)
                    fig.set_size_inches(24, 7)
                    column = 0

        average_SSIM[0] = average_SSIM[0] / count * 1.0
        average_SSIM[1] = average_SSIM[1] / count * 1.0
        print(file = f)
        print("[FFA-Net] Average SSIM: ", np.round(average_SSIM[0], 5), file = f)
        print("[Ours] Average SSIM: ", np.round(average_SSIM[1], 5), file=f)


def create_figures(checkpath, version, iteration):
    HAZY_PATH = "E:/Hazy Dataset Benchmark/Unannotated"
    HAZY_PATH = "E:/Hazy Dataset Benchmark/I-HAZE/hazy"
    GT_PATH = "E:/Hazy Dataset Benchmark/I-HAZE/GT"
    SAVE_PATH = "D:/Users/delgallegon/Documents/GithubProjects/NeuralNets-GenerativeExperiment/results/"
    #SAVE_PATH = "E:/Hazy Dataset Benchmark/Unannotated Results/"

    OUTPUT_SIZE = (708, 1164)
    alpha = 1.0
    beta = 1.0
    
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    gt = dehaze_trainer.DehazeTrainer(version, iteration, device, gen_blocks = 5)
    checkpoint = torch.load(checkpath)
    gt.load_saved_state(0, checkpoint, constants.GENERATOR_KEY, constants.DISCRIMINATOR_KEY, constants.OPTIMIZER_KEY)
    
    denoiser = cycle_gan.Generator(n_residual_blocks=3).to(device)
    denoise_checkpt = torch.load(constants.DENOISE_CHECKPATH)
    denoiser.load_state_dict(denoise_checkpt[constants.GENERATOR_KEY + "A"])


    hazy_list = []; gt_list = []
    for (root, dirs, files) in os.walk(HAZY_PATH):
        for f in files:
                file_name = os.path.join(root, f)
                hazy_list.append(file_name)
    
    for (root, dirs, files) in os.walk(GT_PATH):
        for f in files:
                file_name = os.path.join(root, f)
                gt_list.append(file_name)
    
    dark_transform_op, rgb_transform_op = get_transform_ops(OUTPUT_SIZE)
    
    FIG_ROWS = 3; FIG_COLS = 6
    fig, ax = plt.subplots(ncols=FIG_COLS, nrows=FIG_ROWS, constrained_layout=True, sharex=True)
    fig.set_size_inches(24,7)
    column = 0
    fig_num = 0
    with torch.no_grad():
        for i, (hazy_path, gt_path) in enumerate(zip(hazy_list, gt_list)):
            rgb_img = cv2.imread(hazy_path)
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
            dark_img = tensor_utils.get_dark_channel(rgb_img)
            gt_img = cv2.imread(gt_path)
            gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
            
            dark_img_tensor = dark_transform_op(dark_img)
            dark_img_tensor = torch.unsqueeze(dark_img_tensor, 0).to(device)
            rgb_img_tensor = rgb_transform_op(rgb_img).unsqueeze(0).to(device)
            gt_img_tensor = rgb_transform_op(gt_img).unsqueeze(0)
            
            result_tensor = gt.infer_single(dark_img_tensor, rgb_img_tensor, alpha, beta)
            #result_tensor = denoiser(result_tensor).cpu()
            
            rgb_img_tensor = rgb_img_tensor.cpu()
            rgb_img = tensor_utils.normalize_to_matplotimg(rgb_img_tensor, 0, 0.5, 0.5)
            #result_img = tensor_utils.normalize_to_matplotimg(result_tensor, 0, 0.5, 0.5)
            result_img = result_tensor
            gt_img = tensor_utils.normalize_to_matplotimg(gt_img_tensor, 0, 0.5, 0.5)
            
            # result_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
            # file_name = SAVE_PATH + "img_"+str(i)+ "_" + version + "_" +str(iteration)+".jpg"
            # cv2.imwrite(file_name, result_img)
            
            ax[0,column].imshow(rgb_img); ax[0,column].axis('off')
            ax[1,column].imshow(result_img); ax[1,column].axis('off')
            ax[2,column].imshow(gt_img); ax[2,column].axis('off')
            column = column + 1
            
            if(column == FIG_COLS):
                fig_num = fig_num + 1
                file_name = SAVE_PATH + "fig_"+str(fig_num)+ "_" + version + "_" +str(iteration)+".jpg"
                plt.savefig(file_name)
                plt.show()
                
                #create new figure
                fig, ax = plt.subplots(ncols=FIG_COLS, nrows=FIG_ROWS, constrained_layout=True, sharex=True)
                fig.set_size_inches(24,7)
                column = 0
        
        fig_num = fig_num + 1
        file_name = SAVE_PATH + "fig_"+str(fig_num)+ "_" + version + "_" +str(iteration)+".jpg"
        plt.savefig(file_name)
        plt.show()


def dehaze_infer(checkpath, version, iteration):
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    # load color transfer
    color_transfer_checkpt = torch.load('checkpoint/dehaze_colortransfer_v1.06_10.pt')
    color_transfer_gan = color_gan.Generator().to(device)
    color_transfer_gan.load_state_dict(color_transfer_checkpt[constants.GENERATOR_KEY + "A"])
    print("Color transfer GAN model loaded.")

    dehaze_transfer_checkpt = torch.load(checkpath)
    dehazer_gan = cycle_gan.Generator(n_residual_blocks=8).to(device)
    dehazer_gan.load_state_dict(dehaze_transfer_checkpt[constants.GENERATOR_KEY + "A"])
    print("Dehazer loaded. ", checkpath)
    print("===================================================")

    dataloader = dataset_loader.load_test_dataset(constants.DATASET_VEMON_PATH_COMPLETE,
                                                  constants.DATASET_CLEAN_PATH_COMPLETE, constants.infer_size, -1)

    # Plot some training images
    name_batch, dirty_batch, clean_batch = next(iter(dataloader))
    plt.figure(figsize=constants.FIG_SIZE)
    plt.axis("off")
    plt.title("Training - Dirty Images")
    plt.imshow(np.transpose(
        vutils.make_grid(dirty_batch.to(device)[:constants.infer_size], nrow=8, padding=2, normalize=True).cpu(),
        (1, 2, 0)))
    plt.show()

    plt.figure(figsize=constants.FIG_SIZE)
    plt.axis("off")
    plt.title("Training - Clean Images")
    plt.imshow(np.transpose(
        vutils.make_grid(clean_batch.to(device)[:constants.infer_size], nrow=8, padding=2, normalize=True).cpu(),
        (1, 2, 0)))
    plt.show()

    item_number = 0
    for i, (name, vemon_batch, synth_batch) in enumerate(dataloader, 0):
        with torch.no_grad():
            item_number = item_number + 1

            vemon_tensor = vemon_batch.to(device)
            vemon_dehazed = dehazer_gan(vemon_tensor)

            # resize tensors for better viewing
            resized_vemon = nn.functional.interpolate(vemon_tensor, scale_factor=4.0, mode="bilinear",
                                                      recompute_scale_factor=True)
            resized_vemon_dehazed = nn.functional.interpolate(vemon_dehazed, scale_factor=4.0, mode="bilinear",
                                                              recompute_scale_factor=True)

            print("New shapes: %s %s" % (np.shape(resized_vemon), np.shape(resized_vemon_dehazed)))

            plot_and_save(item_number, resized_vemon, resized_vemon_dehazed)
    

def color_transfer(checkpath, version, iteration):
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    
    colorizer = div2k_trainer.Div2kTrainer(version, iteration, device, g_lr = 0.0002, d_lr = 0.0002)
    
    checkpoint = torch.load(checkpath)
    colorizer.load_saved_state(0, checkpoint, constants.GENERATOR_KEY, constants.DISCRIMINATOR_KEY, constants.OPTIMIZER_KEY)
 
    print("Loaded results checkpt ",checkpath)
    print("===================================================")
    
    dataloader = dataset_loader.load_test_dataset(constants.DATASET_CLEAN_PATH_COMPLETE, constants.DATASET_VEMON_PATH_COMPLETE, constants.infer_size, -1)
    
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
        input_tensor = dirty_batch.to(device)
        item_number = item_number + 1
        colorizer.infer(input_tensor, item_number)

def produce_video_batch():
    VIDEO_FOLDER_PATH = "E:/VEMON Dataset/vemon videos/"
    #VIDEO_FOLDER_PATH = "E:/VEMON Dataset/mmda videos/"
    video_list = os.listdir(VIDEO_FOLDER_PATH)
    for i in range(len(video_list)):
        video_path = VIDEO_FOLDER_PATH + video_list[i]
        print(video_path)
        produce_ffa_video(video_path)


def dark_channel_test():
    HAZY_PATH = "E:/Hazy Dataset Benchmark/Unannotated"
    #HAZY_PATH = "E:/Hazy Dataset Benchmark/I-HAZE/hazy"
    
    GT_PATH = "E:/Hazy Dataset Benchmark/I-HAZE/GT"
    
    hazy_list = []; gt_list = []
    for (root, dirs, files) in os.walk(HAZY_PATH):
        for f in files:
                file_name = os.path.join(root, f)
                hazy_list.append(file_name)
    
    for (root, dirs, files) in os.walk(GT_PATH):
        for f in files:
                file_name = os.path.join(root, f)
                gt_list.append(file_name)

    for i, (hazy_path, gt_path) in enumerate(zip(hazy_list, gt_list)):
        rgb_img = cv2.imread(hazy_path)
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        dark_img = tensor_utils.get_dark_channel_and_mask(rgb_img)
        break

def remove_haze_by_transmission(path_a):
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    data_loader = dataset_loader.load_transmision_test_dataset(path_a, batch_size=1, num_image_to_load=-1)
    #depth_loader = dataset_loader.load_transmission_dataset(constants.DATASET_HAZY_PATH_COMPLETE, constants.DATASET_DEPTH_PATH_COMPLETE, batch_size = 16, num_image_to_load = -1)

    transmission_G = cycle_gan.Generator(input_nc = 1, output_nc = 1, n_residual_blocks=6).to(device)
    checkpt = torch.load('checkpoint/depth_estimator_v1.00_2.pt')
    transmission_G.load_state_dict(checkpt[constants.GENERATOR_KEY + "A"])
    print("Transmission GAN model loaded.")

    count = 0
    for i, (A) in enumerate(data_loader, 0):
        _, img_batch, gray_batch = A
        #_, _, transmission_batch = B

        with torch.no_grad():
            gray_batch = gray_batch.to(device)
            for k in range(np.shape(gray_batch)[0]):

                transmission_img = transmission_G(torch.unsqueeze(gray_batch[k], 0))
                transmission_img = torch.squeeze(transmission_img).cpu().numpy()
                hazy_img = np.transpose(img_batch[k].numpy(), (-2, -1, 0))

                transmission_img = ((transmission_img * 0.5) + 0.5)
                hazy_img = ((hazy_img * 0.5) + 0.5)

                # beta = 0.0
                # for z in range(0, 5):
                #     beta = beta + 0.1
                #     clear_img = tensor_utils.perform_dehazing_equation_with_transmission(hazy_img, transmission_img, beta)
                #     clear_img = cv2.normalize(clear_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                #     #plt.imshow(clear_img)
                #     #plt.show()
                #
                #     count = count + 1
                #     save_img(clear_img, count)

                clear_img = tensor_utils.perform_dehazing_equation_with_transmission(hazy_img, transmission_img, 0.7)
                clear_img = cv2.normalize(clear_img, dst = None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                count = count + 1
                save_img(clear_img, count)

            # count = count + 1
            # result = tensor_utils.perform_dehazing_equation_batch(img_batch.numpy(), transmission_G(gray_batch).cpu().numpy(), 0.85)
            # result_tensor = torch.from_numpy(result)
            # plot_and_save(count, img_batch, result_tensor)



    
def main():
    VERSION = "dehazer_v1.08"
    ITERATION = "1"
    CHECKPATH = 'checkpoint/' + VERSION + "_" + ITERATION +'.pt'
    
    #produce_video_batch()
    benchmark()
    #color_transfer(CHECKPATH, VERSION, ITERATION)
    #remove_haze_by_transmission(constants.DATASET_IHAZE_HAZY_PATH_COMPLETE)

#FIX for broken pipe num_workers issue.
if __name__=="__main__": 
    main()        
        
        