# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 20:11:02 2020

@author: delgallegon
"""
import glob
import os
import torch

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import tensor_utils
import torchvision.utils as torchutils
from loaders import dataset_loader
import torchvision.transforms as transforms
import constants
from model import vanilla_cycle_gan as cycle_gan
from model import ffa_net as ffa_gan
from model import unet_gan
import kornia

DATASET_DIV2K_PATH = "E:/DIV2K_train_HR/"
SAVE_PATH = "E:/VEMON_Transfer/train/C/"

class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
    
def assemble_img_list(path, num_image_to_load = -1):
    img_list = []
    
    for (root, dirs, files) in os.walk(path):
        for f in files:
            file_name = os.path.join(root, f)
            img_list.append(file_name)
    
    return img_list


def unsharp_mask(div2k_img):
    gaussian_3 = cv2.GaussianBlur(div2k_img, (9,9), 10.0)
    unsharp_image = cv2.addWeighted(div2k_img, 1.5, gaussian_3, -0.5, 0)
    
    return unsharp_image

def create_data_from_video(video_path, save_path, filename_format, img_size, patch_size, offset, repeats):
    vidcap = cv2.VideoCapture(video_path)
    count = offset
    success = True

    final_op = transforms.Compose([transforms.ToPILImage(),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.Resize(img_size),
                                   transforms.RandomCrop(patch_size),
                                   transforms.ToTensor()])

    while success:
        success, image = vidcap.read()
        if (success):
            w, h, c = np.shape(image)
            image = cv2.resize(image, (int(h / 4), int(w / 4)), interpolation=cv2.INTER_CUBIC)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            for i in range(repeats):
                file_name = save_path + filename_format % count

                new_img = final_op(image).numpy()
                new_img = np.moveaxis(new_img, -1, 0)
                new_img = np.moveaxis(new_img, -1, 0)

                #plt.imshow(new_img)
                #plt.show()

                cv2.imwrite(file_name, cv2.cvtColor(cv2.convertScaleAbs(new_img, alpha=255.0), cv2.COLOR_BGR2RGB))
                print("Saved: ", file_name)
                count = count + 1

def create_img_data(dataset_path, save_path, filename_format, img_size, patch_size, repeats):
    img_list = assemble_img_list(dataset_path)
    count = 0
    for k in range(len(img_list)):
        normal_img = cv2.imread(img_list[k])
        normal_img = cv2.cvtColor(normal_img, cv2.COLOR_BGR2RGB)
        
        final_op = transforms.Compose([transforms.ToPILImage(), 
                                       transforms.RandomHorizontalFlip(),
                                       transforms.Resize(img_size),
                                       transforms.RandomCrop(patch_size),
                                       transforms.ToTensor()])
        
        for i in range(repeats):
            file_name = save_path + filename_format % count
            
            new_img = final_op(normal_img).numpy()
            final_img = unsharp_mask(new_img)
            
            new_img = np.moveaxis(new_img, -1, 0)
            new_img = np.moveaxis(new_img, -1, 0)
            final_img = np.moveaxis(final_img, -1, 0)
            final_img = np.moveaxis(final_img, -1, 0)
            
            # plt.imshow(new_img)
            # plt.show()
            # plt.imshow(final_img)
            # plt.show()

            cv2.imwrite(file_name, cv2.cvtColor(cv2.convertScaleAbs(final_img, alpha=255.0), cv2.COLOR_BGR2RGB))
            print("Saved: ", file_name)
            count = count + 1


#creates img patches if sufficient connditions are met
def create_filtered_img_data(dataset_path, save_path, filename_format, img_size, patch_size, threshold, repeats, offset = 0):
    img_list = assemble_img_list(dataset_path)
    count = offset
    for k in range(len(img_list)):
        normal_img = cv2.imread(img_list[k])
        normal_img = cv2.cvtColor(normal_img, cv2.COLOR_BGR2RGB)

        final_op = transforms.Compose([transforms.ToPILImage(),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.Resize(img_size),
                                       transforms.RandomCrop(patch_size),
                                       transforms.ToTensor()])

        for i in range(repeats):
            file_name = save_path + filename_format % count

            new_img = final_op(normal_img).numpy()
            final_img = unsharp_mask(new_img)

            new_img = np.moveaxis(new_img, -1, 0)
            new_img = np.moveaxis(new_img, -1, 0)
            final_img = np.moveaxis(final_img, -1, 0)
            final_img = np.moveaxis(final_img, -1, 0)

            #final_img_dc = tensor_utils.get_dark_channel(final_img, 3)
            if(np.linalg.norm(final_img) > 10.0): #filter out very hazy images by dark channel prior
                sobel_x = cv2.Sobel(final_img, cv2.CV_64F, 1, 0, ksize=5)
                sobel_y = cv2.Sobel(final_img, cv2.CV_64F, 0, 1, ksize=5)
                sobel_img = sobel_x + sobel_y
                sobel_quality = np.linalg.norm(sobel_img)
                if(sobel_quality > threshold): #only consider images with good edges
                    cv2.imwrite(file_name, cv2.cvtColor(cv2.convertScaleAbs(final_img, alpha=255.0), cv2.COLOR_BGR2RGB))
                    print("Norm value: ", sobel_quality, " Saved: ", file_name)
            count = count + 1

def create_filtered_paired_img_data(dataset_path_a, dataset_path_b, save_path_a, save_path_b, filename_format, img_size, patch_size, threshold, repeats, offset = 0):
    img_list_a = assemble_img_list(dataset_path_a)
    img_list_b = assemble_img_list(dataset_path_b)

    count = offset
    for k in range(len(img_list_a)):
        img_a = cv2.imread(img_list_a[k])
        img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB)
        img_b = cv2.imread(img_list_b[k])
        img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB)

        initial_transform_op = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(img_size),
        ])

        for i in range(repeats):
            file_name_a = save_path_a + filename_format % count
            file_name_b = save_path_b + filename_format % count

            img_a_patch = initial_transform_op(img_a)
            img_b_patch = initial_transform_op(img_b)

            crop_indices = transforms.RandomCrop.get_params(img_a_patch, output_size=patch_size)
            i, j, h, w = crop_indices

            img_a_patch = transforms.functional.crop(img_a_patch, i, j, h, w)
            img_b_patch = transforms.functional.crop(img_b_patch, i, j, h, w)

            img_b_dc = tensor_utils.get_dark_channel(np.asarray(img_b_patch), 15)
            if (np.linalg.norm(img_b_dc) > 10.0):  # filter out very hazy images by dark channel prior
                sobel_x = cv2.Sobel(img_b_dc, cv2.CV_64F, 1, 0, ksize=5)
                sobel_y = cv2.Sobel(img_b_dc, cv2.CV_64F, 0, 1, ksize=5)
                sobel_img = sobel_x + sobel_y
                sobel_quality = np.linalg.norm(sobel_img)
                if (sobel_quality > threshold):  # only consider images with good edges
                    img_a_patch.save(file_name_a)
                    img_b_patch.save(file_name_b)
                    #plt.imshow(img_b_patch)
                    #plt.show()
                    print("Norm value: ", sobel_quality, " Saved: ", file_name_a, file_name_b)
            count = count + 1

def create_paired_img_data(dataset_path_a, dataset_path_b, save_path_a, save_path_b, filename_format, img_size, patch_size, repeats):
    img_list_a = assemble_img_list(dataset_path_a)
    img_list_b = assemble_img_list(dataset_path_b)

    count = 0
    for k in range(len(img_list_a)):
        img_a = cv2.imread(img_list_a[k])
        img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB)
        img_b = cv2.imread(img_list_b[k])
        img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB)

        initial_transform_op = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(img_size),
        ])

        for i in range(repeats):
            file_name_a = save_path_a + filename_format % count
            file_name_b = save_path_b + filename_format % count

            img_a_patch = initial_transform_op(img_a)
            img_b_patch = initial_transform_op(img_b)

            crop_indices = transforms.RandomCrop.get_params(img_a_patch, output_size=patch_size)
            i, j, h, w = crop_indices

            img_a_patch = transforms.functional.crop(img_a_patch, i, j, h, w)
            img_b_patch = transforms.functional.crop(img_b_patch, i, j, h, w)

            img_a_patch.save(file_name_a)
            img_b_patch.save(file_name_b)
            print("Saved: ", file_name_a, file_name_b)
            count = count + 1

def create_tri_img_data(dataset_path_a, dataset_path_b, dataset_path_c, save_path_a, save_path_b, save_path_c,
                        filename_format, img_size, patch_size, repeats, offset = 0):
    img_list_a = assemble_img_list(dataset_path_a)
    img_list_b = assemble_img_list(dataset_path_b)
    img_list_c = assemble_img_list(dataset_path_c)

    count = offset
    for k in range(len(img_list_a)):
        img_a = cv2.imread(img_list_a[k])
        img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB)
        img_b = cv2.imread(img_list_b[k])
        img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB)
        img_c = cv2.imread(img_list_c[k])
        img_c = cv2.cvtColor(img_c, cv2.COLOR_BGR2RGB)

        initial_transform_op = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(img_size),
        ])

        for i in range(repeats):
            file_name_a = save_path_a + filename_format % count
            file_name_b = save_path_b + filename_format % count
            file_name_c = save_path_c + filename_format % count

            img_a_patch = initial_transform_op(img_a)
            img_b_patch = initial_transform_op(img_b)
            img_c_patch = initial_transform_op(img_c)

            crop_indices = transforms.RandomCrop.get_params(img_a_patch, output_size=patch_size)
            i, j, h, w = crop_indices

            img_a_patch = transforms.functional.crop(img_a_patch, i, j, h, w)
            img_b_patch = transforms.functional.crop(img_b_patch, i, j, h, w)
            img_c_patch = transforms.functional.crop(img_c_patch, i, j, h, w)

            img_a_patch.save(file_name_a)
            img_b_patch.save(file_name_b)
            img_c_patch.save(file_name_c)

            print("Saved: ", file_name_a, file_name_b, file_name_c)
            count = count + 1

def create_hazy_data(offset):
    clean_video_path = "D:/Users/delgallegon/Documents/GithubProjects/NeuralNets-SynthWorkplace/Recordings/synth_12_clean.mp4"
    #hazy_video_path = "D:/Users/delgallegon/Documents/GithubProjects/NeuralNets-SynthWorkplace/Recordings/synth_11_haze.mp4"
    albedo_video_path = "D:/Users/delgallegon/Documents/GithubProjects/NeuralNets-SynthWorkplace/Recordings/synth_12_albedo.mp4"
    depth_video_path = "D:/Users/delgallegon/Documents/GithubProjects/NeuralNets-SynthWorkplace/Recordings/synth_12_depth.mp4"

    CLEAN_SAVE_PATH = "E:/Synth Hazy 3/clean/"
    #HAZY_SAVE_PATH = "E:/Synth Hazy 3/hazy/"
    DEPTH_SAVE_PATH = "E:/Synth Hazy 3/depth/"
    ALBEDO_SAVE_PATH = "E:/Synth Hazy 3/albedo/"
    
    vidcap = cv2.VideoCapture(clean_video_path)
    count = offset

    success = True
    while success:
        success,image = vidcap.read()
        if(success):
            w,h,c = np.shape(image)
            image = cv2.resize(image, (int(h / 4), int(w / 4)), interpolation = cv2.INTER_CUBIC)
            cv2.imwrite(CLEAN_SAVE_PATH + "synth_%d.png" % count, image)
            print("Saved clean: synth_%d.png" % count)
            count += 1
    
    #for noisy
    # vidcap = cv2.VideoCapture(hazy_video_path)
    # count = offset
    #
    # success = True
    # while success:
    #     success,image = vidcap.read()
    #     if(success):
    #         w,h,c = np.shape(image)
    #         image = cv2.resize(image, (int(h / 4), int(w / 4)), interpolation = cv2.INTER_CUBIC)
    #         cv2.imwrite(HAZY_SAVE_PATH + "synth_%d.png" % count, image)
    #         print("Saved hazy: synth_%d.png" % count)
    #         count += 1

    #for depth
    vidcap = cv2.VideoCapture(depth_video_path)
    count = offset
    success = True
    while success:
        success, image = vidcap.read()
        if (success):
            w, h, c = np.shape(image)
            image = cv2.resize(image, (int(h / 4), int(w / 4)), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(DEPTH_SAVE_PATH + "synth_%d.png" % count, image)
            print("Saved depth: synth_%d.png" % count)
            count += 1

    #for albedo
    vidcap = cv2.VideoCapture(albedo_video_path)
    count = offset
    success = True
    while success:
        success, image = vidcap.read()
        if (success):
            w, h, c = np.shape(image)
            image = cv2.resize(image, (int(h / 4), int(w / 4)), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(ALBEDO_SAVE_PATH + "synth_%d.png" % count, image)
            print("Saved albedo: synth_%d.png" % count)
            count += 1


def create_img_from_video_data(VIDEO_PATH, SAVE_PATH, offset):
    vidcap = cv2.VideoCapture(VIDEO_PATH)
    count = offset

    success = True
    while success:
        success, image = vidcap.read()
        if (success):
            w, h, c = np.shape(image)
            image = cv2.resize(image, (int(h / 4), int(w / 4)), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(SAVE_PATH + "synth_%d.png" % count, image)
            print("Saved: synth_%d.png" % count)
            count += 1

def produce_color_images(INPUT_PATH, SAVE_PATH, CHECKPT_NAME, net_config):
    CHECKPT_ROOT = "D:/Documents/GithubProjects/SynthDehazing/checkpoint/"

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    try:
        os.mkdir(SAVE_PATH)
    except OSError as error:
        print("Save path already exists. Skipping.", error)

    # load color transfer
    if (net_config == 1):
        print("Using vanilla cycle GAN")
        color_transfer_gan = cycle_gan.Generator(n_residual_blocks=6, has_dropout=False).to(device)
    elif (net_config == 3):
        print("Using U-Net GAN")
        color_transfer_gan = unet_gan.UnetGenerator(input_nc=3, output_nc=3, num_downs=4).to(device)
    else:
        print("Using stable CycleGAN")
        color_transfer_gan = cycle_gan.Generator(downsampling_blocks=2, n_residual_blocks=10, has_dropout=False).to(device)

    color_transfer_checkpt = torch.load(CHECKPT_ROOT + CHECKPT_NAME, map_location=device)
    color_transfer_gan.load_state_dict(color_transfer_checkpt[constants.GENERATOR_KEY + "A"])
    color_transfer_gan.eval()
    print("Color transfer GAN model loaded.")
    print("===================================================")

    dataloader = dataset_loader.load_test_dataset(INPUT_PATH, constants.DATASET_PLACES_PATH, constants.infer_size, -1)

    # Plot some training images
    name_batch, dirty_batch, clean_batch = next(iter(dataloader))
    plt.axis("off")
    plt.title("Training - Old Images")
    plt.imshow(np.transpose(torchutils.make_grid(dirty_batch.to(device)[:constants.infer_size], nrow=8, padding=2, normalize=True).cpu(), (1, 2, 0)))
    plt.show()

    plt.figure(figsize=constants.FIG_SIZE)
    plt.axis("off")
    plt.title("Training - New Images")
    plt.imshow(np.transpose(torchutils.make_grid(clean_batch.to(device)[:constants.infer_size], nrow=8, padding=2, normalize=True).cpu(), (1, 2, 0)))
    plt.show()

    for i, (name, dirty_batch, clean_batch) in enumerate(dataloader, 0):
        with torch.no_grad():
            input_tensor = dirty_batch.to(device)
            result = color_transfer_gan(input_tensor)

            for i in range(0, len(result)):
                img_name = name[i].split(".")[0]
                style_img = result[i].cpu().numpy()
                style_img = ((style_img * 0.5) + 0.5) #remove normalization
                style_img = np.rollaxis(style_img, 0, 3)
                style_img = cv2.normalize(style_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                style_img = cv2.cvtColor(style_img, cv2.COLOR_BGR2RGB)

                cv2.imwrite(SAVE_PATH + img_name + ".png", style_img)
                print("Saved styled image: ", img_name)

def produce_single_color_img(IMG_PATH, CHECKPT_NAME, net_config):
    SAVE_PATH = "D:/Documents/GithubProjects/SynthDehazing/results/Single/"
    CHECKPT_ROOT = "D:/Documents/GithubProjects/SynthDehazing/checkpoint/"

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    # load color transfer
    if (net_config == 1):
        print("Using vanilla cycle GAN")
        color_transfer_gan = cycle_gan.Generator(n_residual_blocks=6, has_dropout=False).to(device)
    elif(net_config == 3):
        print("Using U-Net GAN")
        color_transfer_gan = unet_gan.UnetGenerator(input_nc=3, output_nc=3, num_downs=4).to(device)
    else:
        print("Using stable CycleGAN")
        color_transfer_gan = cycle_gan.Generator(downsampling_blocks=2, n_residual_blocks=10, has_dropout=False).to(device)

    color_transfer_checkpt = torch.load(CHECKPT_ROOT + CHECKPT_NAME, map_location=device)
    color_transfer_gan.load_state_dict(color_transfer_checkpt[constants.GENERATOR_KEY + "A"])
    color_transfer_gan.eval()
    print("Color transfer GAN model loaded.")
    print("===================================================")

    img_name = IMG_PATH.split("/")[-1].split(".")[0]
    img_a = cv2.imread(IMG_PATH)
    img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB)

    transform_op = transforms.Compose([transforms.ToPILImage(),
                                       transforms.Resize(constants.TEST_IMAGE_SIZE),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    img_tensor = transform_op(img_a).to(device)
    img_tensor = torch.unsqueeze(img_tensor, 0)

    with torch.no_grad():
        styled_tensor = color_transfer_gan(img_tensor)
        styled_tensor = torch.squeeze(styled_tensor)
        styled_tensor = (styled_tensor * 0.5) + 0.5 #remove normalization
        torchutils.save_image(styled_tensor, SAVE_PATH + img_name + "_" + CHECKPT_NAME + ".png")

def produce_pseudo_albedo_images():
    SAVE_PATH = "E:/Synth Hazy - Low/albedo - pseudo/"
    CHECKPT_ROOT = "D:/Documents/GithubProjects/SynthDehazing/checkpoint/"

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    # load color transfer
    color_transfer_checkpt = torch.load(CHECKPT_ROOT + "albedo_transfer_v1.04_1.pt")
    #albedo_transfer_gan = cycle_gan.Generator(n_residual_blocks=8).to(device)
    albedo_transfer_gan = ffa_gan.FFA(gps=3, blocks=18).to(device)
    albedo_transfer_gan.load_state_dict(color_transfer_checkpt[constants.GENERATOR_KEY + "A"])
    albedo_transfer_gan.eval()

    albedo_discriminator = cycle_gan.Discriminator().to(device)
    albedo_discriminator.load_state_dict(color_transfer_checkpt[constants.DISCRIMINATOR_KEY + "A"])
    albedo_discriminator.eval()

    print("Color transfer GAN model loaded.")
    print("===================================================")

    dataloader = dataset_loader.load_test_dataset("E:/Synth Hazy - Low/clean - styled/", constants.DATASET_ALBEDO_PATH_COMPLETE_3, constants.infer_size, -1)

    # Plot some training images
    name_batch, dirty_batch, clean_batch = next(iter(dataloader))
    plt.figure(figsize=constants.FIG_SIZE)
    plt.axis("off")
    plt.title("Test - Colored Images")
    plt.imshow(np.transpose(torchutils.make_grid(dirty_batch.to(device)[:constants.infer_size], nrow=8, padding=2, normalize=True).cpu(), (1, 2, 0)))
    plt.show()

    plt.figure(figsize=constants.FIG_SIZE)
    plt.axis("off")
    plt.title("Test - Albedo Images")
    plt.imshow(np.transpose(torchutils.make_grid(clean_batch.to(device)[:constants.infer_size], nrow=8, padding=2, normalize=True).cpu(), (1, 2, 0)))
    plt.show()

    for i, (name, dirty_batch, _) in enumerate(dataloader, 0):
        with torch.no_grad():
            input_tensor = dirty_batch.to(device)
            result = albedo_transfer_gan(input_tensor)
            result = kornia.adjust_brightness(result, 0.6)
            prediction = albedo_discriminator(result)
            for i in range(0, len(result)):
                img_name = name[i].split(".")[0]
                #first check with discrminator score. If it's good, save image

                # if(prediction[i].item() > 3.55):
                style_img = result[i].cpu().numpy()
                style_img = ((style_img * 0.5) + 0.5) #remove normalization
                style_img = np.rollaxis(style_img, 0, 3)
                style_img = cv2.normalize(style_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                style_img = cv2.cvtColor(style_img, cv2.COLOR_BGR2RGB)

                cv2.imwrite(SAVE_PATH + img_name + ".png", style_img)
                print("Prediction of %s from discriminator: %f. Saved styled image: %s" % (img_name, prediction[i].item(), img_name))


def process_gta_images():

    INPUT_PATH = "E:/GTAV_540/"
    OUTPUT_PATH = "E:/GTAV_Processed/"

    rgb_list = glob.glob(INPUT_PATH + "*/images/*.png", recursive=True)
    depth_list = glob.glob(INPUT_PATH + "*/depths/*.exr", recursive=True)
    print("Images found: ", len(rgb_list), len(depth_list))

    for i, (rgb_path, depth_path) in enumerate(zip(rgb_list, depth_list)):
        file_name = rgb_path.split("\\")[-1]
        print(file_name)
        rgb_img = tensor_utils.load_metrics_compatible_img(rgb_path, (810, 540))
        depth_img = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        depth_img = np.reciprocal(depth_img)

        depth_img = cv2.normalize(depth_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # plt.imshow(depth_img, cmap = 'gray')
        # plt.show()
        # plt.imshow(rgb_img)
        # plt.show()
        # break

        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(OUTPUT_PATH + "/images/synth_" + str(i) + ".png", rgb_img)
        cv2.imwrite(OUTPUT_PATH + "/depth/synth_" + str(i) + ".png", depth_img)



def main():
    # process_gta_images()
    # VIDEO_PATH = "D:/Users/delgallegon/Documents/GithubProjects/NeuralNets-SynthWorkplace/Recordings/directionality_1.mp4"
    # SAVE_PATH = "E:/Synth Hazy 2/directionality/"
    # create_data_from_video(VIDEO_PATH, SAVE_PATH, "lightdir_%d.png", (512, 512), (256, 256), offset = 0, repeats = 7)
    #
    # VIDEO_PATH = "D:/Users/delgallegon/Documents/GithubProjects/NeuralNets-SynthWorkplace/Recordings/directionality_2.mp4"
    # create_data_from_video(VIDEO_PATH, SAVE_PATH, "lightdir_%d.png", (512, 512), (256, 256), offset=0, repeats=7)

    #PATH_A = constants.DATASET_ALBEDO_PATH_PSEUDO_3
    #SAVE_PATH_A = constants.DATASET_ALBEDO_PATH_PSEUDO_PATCH_3
    #create_filtered_img_data(PATH_A, SAVE_PATH_A, "frame_%d.png", (256, 256), (32, 32), 25, 16, offset = 0)

    # PATH_A = "D:/Documents/GithubProjects/NeuralNets-SynthWorkplace/Recordings/hazy_001.mp4"
    # SAVE_PATH_A = constants.DATASET_HAZY_END_TO_END_PATH
    #
    # PATH_B = "D:/Documents/GithubProjects/NeuralNets-SynthWorkplace/Recordings/clean_001.mp4"
    # SAVE_PATH_B = constants.DATASET_CLEAN_END_TO_END_PATH
    #
    # create_img_from_video_data(PATH_A, SAVE_PATH_A, 0)
    # create_img_from_video_data(PATH_B, SAVE_PATH_B, 0)

    # PATH_A = "D:/Documents/GithubProjects/NeuralNets-SynthWorkplace/Recordings/clean_003.mp4"
    # SAVE_PATH_A = constants.DATASET_CLEAN_LOW_PATH
    #
    # PATH_B = "D:/Documents/GithubProjects/NeuralNets-SynthWorkplace/Recordings/depth_003.mp4"
    # SAVE_PATH_B = constants.DATASET_DEPTH_LOW_PATH
    #
    # create_img_from_video_data(PATH_A, SAVE_PATH_A, 0)
    # create_img_from_video_data(PATH_B, SAVE_PATH_B, 0)

    # PATH_A = "D:/Documents/GithubProjects/NeuralNets-SynthWorkplace/Recordings/hazy_002.mp4"
    # SAVE_PATH_A = constants.DATASET_HAZY_END_TO_END_PATH_TEST
    #
    # PATH_B = "D:/Documents/GithubProjects/NeuralNets-SynthWorkplace/Recordings/clean_002.mp4"
    # SAVE_PATH_B = constants.DATASET_CLEAN_END_TO_END_PATH_TEST
    #
    # create_img_from_video_data(PATH_A, SAVE_PATH_A, 0)
    # create_img_from_video_data(PATH_B, SAVE_PATH_B, 0)

    #create_hazy_data(0)
    # produce_color_images("E:/Synth Hazy 4/clean/", "E:/Synth Hazy 4/clean - styled/",  "synth2places_v1.15_1.pt", net_config = 3)
    # produce_color_images(constants.DATASET_CLEAN_PATH_COMPLETE_GTA, constants.DATASET_CLEAN_PATH_COMPLETE_STYLED_GTA, "color_transfer_v1.11_1 - stable.pt", net_config=2)

    produce_color_images("E:/SynthWeather Dataset 6/azimuth/0deg/rgb/", "E:/SynthWeather Dataset 6/azimuth/0deg/rgb - styled/", "color_transfer_v1.11_1 - stable.pt", net_config=2)


    #produce_pseudo_albedo_images()
    # produce_single_color_img("E:/Synth Hazy - End-to-End - Test/clean/synth_4918.png", "synth2places_v1.15_1.pt", net_config = 3)
    # produce_single_color_img("E:/Synth Hazy - End-to-End - Test/clean/synth_4918.png", "synth2places_v1.15_2.pt", net_config=3)
    # produce_single_color_img("E:/Synth Hazy - End-to-End - Test/clean/synth_4918.png", "synth2places_v1.15_3.pt", net_config=3)
    # produce_single_color_img("E:/Synth Hazy - End-to-End - Test/clean/synth_4918.png", "synth2places_v1.15_4.pt", net_config=3)
    # produce_single_color_img("E:/Synth Hazy - End-to-End - Test/clean/synth_4918.png", "synth2places_v1.16_1.pt", net_config=1)
    # produce_single_color_img("E:/Synth Hazy - End-to-End - Test/clean/synth_4918.png", "synth2places_v1.16_2.pt", net_config=1)
    # produce_single_color_img("E:/Synth Hazy - End-to-End - Test/clean/synth_4918.png", "synth2places_v1.16_3.pt", net_config=1)
    # produce_single_color_img("E:/Synth Hazy - End-to-End - Test/clean/synth_4918.png", "synth2places_v1.16_4.pt", net_config=1)
    # produce_single_color_img("E:/Synth Hazy - End-to-End - Test/clean/synth_4918.png", "color_transfer_v1.11_2.pt", net_config=4)

if __name__=="__main__": 
    main()   