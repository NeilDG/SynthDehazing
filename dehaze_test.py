import torch.utils.data
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from PIL import Image

from loaders import dataset_loader
from trainers import denoise_net_trainer
from trainers import albedo_trainer
from trainers import dehaze_trainer
from model import vanilla_cycle_gan as cycle_gan
from model import style_transfer_gan as color_gan
from model import ffa_net as ffa
import constants
from torchvision import transforms
import cv2
from utils import tensor_utils
from utils import dark_channel_prior
import os
import glob
from skimage.metrics import peak_signal_noise_ratio
from skimage.measure import compare_ssim

def save_img(img_numpy, item_number):
    LOCATION = os.getcwd() + "/figures/"
    im = Image.fromarray(img_numpy)
    im.save(LOCATION + "image_" + str(item_number) + ".png")

def show_images(img_tensor, caption):
    plt.figure(figsize=(16, 4))
    plt.axis("off")
    plt.title(caption)
    plt.imshow(np.transpose(
        vutils.make_grid(img_tensor[:constants.batch_size], nrow=8, padding=2, normalize=True).cpu(),
        (1, 2, 0)))
    plt.show()


def dark_channel_test():
    HAZY_PATH = "E:/Hazy Dataset Benchmark/Unannotated"
    # HAZY_PATH = "E:/Hazy Dataset Benchmark/I-HAZE/hazy"

    GT_PATH = "E:/Hazy Dataset Benchmark/I-HAZE/GT"

    hazy_list = [];
    gt_list = []
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
    # depth_loader = dataset_loader.load_transmission_dataset(constants.DATASET_HAZY_PATH_COMPLETE, constants.DATASET_DEPTH_PATH_COMPLETE, batch_size = 16, num_image_to_load = -1)

    transmission_G = cycle_gan.Generator(input_nc=1, output_nc=1, n_residual_blocks=6).to(device)
    checkpt = torch.load('checkpoint/depth_estimator_v1.00_2.pt')
    transmission_G.load_state_dict(checkpt[constants.GENERATOR_KEY + "A"])
    print("Transmission GAN model loaded.")

    count = 0
    for i, (A) in enumerate(data_loader, 0):
        _, img_batch, gray_batch = A
        # _, _, transmission_batch = B

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
                clear_img = cv2.normalize(clear_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                          dtype=cv2.CV_8U)
                count = count + 1
                save_img(clear_img, count)

            # count = count + 1
            # result = tensor_utils.perform_dehazing_equation_batch(img_batch.numpy(), transmission_G(gray_batch).cpu().numpy(), 0.85)
            # result_tensor = torch.from_numpy(result)
            # plot_and_save(count, img_batch, result_tensor)

def transmission_intersection():
    HAZY_PATH = "E:/Hazy Dataset Benchmark/O-HAZE/hazy/"
    GT_PATH = "E:/Hazy Dataset Benchmark/O-HAZE/GT/"
    SAVE_PATH = "results/"
    MODEL_CHECKPOINT = "transmission_estimator_v1.01_1"
    BENCHMARK_PATH = "results/metrics - " + str(MODEL_CHECKPOINT) + ".txt"

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    hazy_list = glob.glob(HAZY_PATH + "*.jpg")

    print(hazy_list)

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

    count = 0
    for i, (hazy_path) in enumerate(hazy_list):
        with torch.no_grad():
            print(hazy_path)
            count = count + 1
            img_name = hazy_path.split("\\")[1]
            hazy_img = cv2.imread(hazy_path)
            hazy_img = cv2.resize(hazy_img, (512, 512))

            input_tensor = rgb_img_op(cv2.cvtColor(hazy_img, cv2.COLOR_BGR2RGB)).to(device)
            transmission_infer = transmission_G(torch.unsqueeze(input_tensor, 0))
            transmission_infer = torch.squeeze(transmission_infer).cpu().numpy()


            # remove 0.5 normalization for dehazing equation
            transmission_infer = 1 - ((transmission_infer * 0.5) + 0.5)
            hazy_img = ((hazy_img * 0.5) + 0.5)

            dark_channel = dark_channel_prior.get_dark_channel(hazy_img, 15)
            dcp_transmission = dark_channel_prior.estimate_transmission(hazy_img, dark_channel_prior.estimate_atmosphere(hazy_img, dark_channel),
                                                                        dark_channel)
            #DCP is not 0-1 range
            dcp_transmission = cv2.normalize(dcp_transmission, dst = None, alpha = 0, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)

            # normalize images
            hazy_img = cv2.normalize(hazy_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            dcp_transmission_img = cv2.normalize(dcp_transmission, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            transmission_img = cv2.normalize(transmission_infer, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

            dcp_mask = np.greater(dcp_transmission, 0.1).astype(int)
            transmission_mask = np.greater(transmission_infer, 0.1).astype(int)
            union = np.logical_or(dcp_mask, transmission_mask)
            blend = dcp_transmission * 0.5 + transmission_infer * 0.5
            blend_img = cv2.normalize(blend, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

            fig, ax = plt.subplots(ncols=3, nrows=2, constrained_layout=True, sharex=True)
            ax[0, 0].imshow(hazy_img)
            ax[0, 1].imshow(dcp_mask)
            ax[0, 2].imshow(transmission_mask)
            ax[1, 0].imshow(dcp_transmission_img)
            ax[1, 1].imshow(transmission_img)
            ax[1, 2].imshow(blend_img)
            plt.show()


def main():
    transmission_intersection()

if __name__=="__main__":
    main()