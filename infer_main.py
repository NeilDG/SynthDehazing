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
from loaders import dataset_loader
from trainers import denoise_net_trainer
from trainers import div2k_trainer
from trainers import dehaze_trainer
from model import vanilla_cycle_gan as denoise_gan
import constants
from torchvision import transforms
import cv2
from utils import tensor_utils
import os

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

def produce_video(video_path, checkpath, version, iteration):
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    gt = dehaze_trainer.DehazeTrainer(version, iteration, device, gen_blocks = 5)
    checkpoint = torch.load(checkpath)
    gt.load_saved_state(0, checkpoint, constants.GENERATOR_KEY, constants.DISCRIMINATOR_KEY, constants.OPTIMIZER_KEY)
    
    denoiser = denoise_gan.Generator(n_residual_blocks=3).to(device)
    denoise_checkpt = torch.load(constants.DENOISE_CHECKPATH)
    denoiser.load_state_dict(denoise_checkpt[constants.GENERATOR_KEY + "A"])
    
    vidcap = cv2.VideoCapture(video_path)
    success,image = vidcap.read()
    success = True
    
    OUTPUT_SIZE = (480,704)
    dark_transform_op, rgb_transform_op = get_transform_ops(OUTPUT_SIZE)
    
    video_name = video_path.split("/")[3].split(".")[0]
    SAVE_PATH = "E:/VEMON Dataset/vemon enhanced/" + video_name + "_" + version + "_" +str(iteration)+"_enhanced.avi"
    
    video_out = cv2.VideoWriter(SAVE_PATH, cv2.VideoWriter_fourcc(*"MJPG"), 8.0, (OUTPUT_SIZE[1],OUTPUT_SIZE[0]))
    
    alpha = 0.7
    beta = 0.7
    with torch.no_grad():
        while success:
            success,rgb_img = vidcap.read()
            if(success):
                rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
                dark_img = tensor_utils.get_dark_channel(rgb_img)
                
                dark_img_tensor = dark_transform_op(dark_img)
                dark_img_tensor = torch.unsqueeze(dark_img_tensor, 0).to(device)
                rgb_img_tensor = rgb_transform_op(rgb_img).unsqueeze(0).to(device)
                
                result_tensor = gt.infer_single(dark_img_tensor, rgb_img_tensor, alpha, beta)
                result_tensor = denoiser(result_tensor).cpu()
                
                result_img = tensor_utils.normalize_to_matplotimg(result_tensor, 0, 0.5, 0.5)
                
                #plt.imshow(result_img)
                #plt.show()
                result_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
                video_out.write(result_img)
        video_out.release()
                
def benchmark(checkpath, version, iteration):
    HAZY_PATH = "E:/Hazy Dataset Benchmark/Unannotated"
    HAZY_PATH = "E:/Hazy Dataset Benchmark/I-HAZE/hazy"
    
    GT_PATH = "E:/Hazy Dataset Benchmark/I-HAZE/GT"
    
    SAVE_PATH = "E:/Hazy Dataset Benchmark/I-HAZE/results/"
    #SAVE_PATH = "E:/Hazy Dataset Benchmark/Unannotated Results/"
    
    OUTPUT_SIZE = (708, 1164)
    alpha = 1.0
    beta = 1.0
    
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    gt = dehaze_trainer.DehazeTrainer(version, iteration, device, gen_blocks = 5)
    checkpoint = torch.load(checkpath)
    gt.load_saved_state(0, checkpoint, constants.GENERATOR_KEY, constants.DISCRIMINATOR_KEY, constants.OPTIMIZER_KEY)
    
    denoiser = denoise_gan.Generator(n_residual_blocks=3).to(device)
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
        
def dehaze_infer(batch_size, checkpath, version, iteration):
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    
    denoiser = denoise_gan.Generator(n_residual_blocks=3).to(device)
    denoise_checkpt = torch.load(constants.DENOISE_CHECKPATH)
    denoiser.load_state_dict(denoise_checkpt[constants.GENERATOR_KEY + "A"])
    dehazer = denoise_net_trainer.DenoiseTrainer(version, iteration, device, gen_blocks=3)
    
    checkpoint = torch.load(checkpath)
    dehazer.load_saved_state(0, checkpoint, constants.GENERATOR_KEY, constants.DISCRIMINATOR_KEY, constants.OPTIMIZER_KEY)
 
    print("Loaded results checkpt ",checkpath)
    print("===================================================")
    
    dataloader = dataset_loader.load_test_dataset(constants.DATASET_VEMON_PATH, constants.DATASET_CLEAN_PATH, batch_size, -1)
    
    # Plot some training images
    name_batch, dirty_batch, clean_batch = next(iter(dataloader))
    plt.figure(figsize=constants.FIG_SIZE)
    plt.axis("off")
    plt.title("Training - Dirty Images")
    plt.imshow(np.transpose(vutils.make_grid(dirty_batch.to(device)[:constants.infer_size], nrow = 8, padding=2, normalize=True).cpu(),(1,2,0)))
    plt.show()
    
    plt.figure(figsize=constants.FIG_SIZE)
    plt.axis("off")
    plt.title("Training - Clean Images")
    plt.imshow(np.transpose(vutils.make_grid(clean_batch.to(device)[:constants.infer_size], nrow = 8, padding=2, normalize=True).cpu(),(1,2,0)))
    plt.show()
    
    item_number = 0
    for i, (name, vemon_batch, gta_batch) in enumerate(dataloader, 0):
        vemon_tensor = vemon_batch.to(device)
        item_number = item_number + 1
        dehazer.dehaze_infer(denoiser, vemon_tensor, item_number)
    

def color_transfer(checkpath, version, iteration):
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    
    colorizer = div2k_trainer.Div2kTrainer(version, iteration, device, g_lr = 0.0002, d_lr = 0.0002)
    
    checkpoint = torch.load(checkpath)
    colorizer.load_saved_state(0, checkpoint, constants.GENERATOR_KEY, constants.DISCRIMINATOR_KEY, constants.OPTIMIZER_KEY)
 
    print("Loaded results checkpt ",checkpath)
    print("===================================================")
    
    dataloader = dataset_loader.load_test_dataset(constants.DATASET_HAZY_PATH, constants.DATASET_VEMON_PATH, constants.infer_size, -1)
    
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

def produce_video_batch(CHECKPATH, VERSION, ITERATION):
    VIDEO_FOLDER_PATH = "E:/VEMON Dataset/vemon videos/"
    #VIDEO_FOLDER_PATH = "E:/VEMON Dataset/mmda videos/"
    video_list = os.listdir(VIDEO_FOLDER_PATH)
    for i in range(len(video_list)):
        video_path = VIDEO_FOLDER_PATH + video_list[i]
        print(video_path)
        produce_video(video_path, CHECKPATH, VERSION, ITERATION)


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
        
    
def main():
    VERSION = "dehaze_colortransfer_v1.06"
    ITERATION = "8"
    CHECKPATH = 'checkpoint/' + VERSION + "_" + ITERATION +'.pt'
    
    #produce_video_batch(CHECKPATH, VERSION, ITERATION)
    #benchmark(CHECKPATH, VERSION, ITERATION)
    color_transfer(CHECKPATH, VERSION, ITERATION)

#FIX for broken pipe num_workers issue.
if __name__=="__main__": 
    main()        
        
        