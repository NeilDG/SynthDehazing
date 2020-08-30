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
    
    OUTPUT_SIZE = (400,800)
    dark_transform_op = transforms.Compose([transforms.ToPILImage(), 
                                            transforms.CenterCrop(OUTPUT_SIZE),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5), (0.5))])
    
    rgb_transform_op = transforms.Compose([transforms.ToPILImage(),
                                           transforms.CenterCrop(OUTPUT_SIZE),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    video_name = video_path.split("/")[3].split(".")[0]
    SAVE_PATH = "E:/VEMON Dataset/mmda videos enhanced/" + video_name + "_" + version + "_" +str(iteration)+"_enhanced.avi"
    
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
                
      
def infer(batch_size, checkpath, version, iteration):
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    
    #gt = denoise_net_trainer.DenoiseTrainer(version, iteration, device, gen_blocks=3)
    gt = div2k_trainer.Div2kTrainer(version, iteration, device)
    checkpoint = torch.load(checkpath)
    gt.load_saved_state(0, checkpoint, constants.GENERATOR_KEY, constants.DISCRIMINATOR_KEY, constants.OPTIMIZER_KEY)
 
    print("Loaded results checkpt ",checkpath)
    print("===================================================")
    
    dataloader = dataset_loader.load_test_dataset(constants.DATASET_VEMON_PATH, constants.DATASET_DIV2K_PATH, batch_size, 72296)
    
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
        gt.infer(vemon_tensor, item_number)

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

def produce_video_batch(CHECKPATH, VERSION, ITERATION):
    #VIDEO_FOLDER_PATH = "E:/VEMON Dataset/vemon videos/"
    VIDEO_FOLDER_PATH = "E:/VEMON Dataset/mmda videos/"
    video_list = os.listdir(VIDEO_FOLDER_PATH)
    for i in range(len(video_list)):
        video_path = VIDEO_FOLDER_PATH + video_list[i]
        print(video_path)
        produce_video(video_path, CHECKPATH, VERSION, ITERATION)


def main():
    VERSION = "dehaze_v1.02"
    ITERATION = "3"
    CHECKPATH = 'checkpoint/' + VERSION + "_" + ITERATION +'.pt'
    
    #dehaze_infer(constants.infer_size, CHECKPATH, VERSION, ITERATION)
    #infer(constants.infer_size, CHECKPATH, VERSION, ITERATION)
    produce_video_batch(CHECKPATH, VERSION, ITERATION)

#FIX for broken pipe num_workers issue.
if __name__=="__main__": 
    main()        
        
        