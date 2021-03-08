import torch.utils.data
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from trainers import transmission_trainer
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
from skimage.metrics import structural_similarity
from processing import gist
from custom_losses import vgg_loss_model

def monet_perceptual_loss():
    ALL_MONET_PATH = "G:/My Drive/CONFERENCES/WSCG 2021/Experiments/Monet Gallery numbered/"
    THESIS_PATH = "G:/My Drive/CONFERENCES/WSCG 2021/Experiments/FINAL PUSH/thesis output/"
    PREDECESSOR_PATH = "G:/My Drive/CONFERENCES/WSCG 2021/Experiments/FINAL PUSH/predecessor output/"
    CYCLEGAN_PATH = "G:/My Drive/CONFERENCES/WSCG 2021/Experiments/FINAL PUSH/PyTorch GAN output/"
    ECCV_PATH = "G:/My Drive/CONFERENCES/WSCG 2021/Experiments/FINAL PUSH/AdaptiveStyleTransfer-ECCV output/"
    BENCHMARK_PATH = "G:/My Drive/CONFERENCES/WSCG 2021/Experiments/perceptual_losses_256.txt"

    monet_list = glob.glob(ALL_MONET_PATH + "*.jpg")
    thesis_list = glob.glob(THESIS_PATH + "*.jpg")

    print(monet_list)
    print(thesis_list)

    rgb_img_op = transforms.Compose([transforms.ToPILImage(),
                                     transforms.ToTensor()])

    IMG_SIZE = (256, 256)

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    vgg_loss = vgg_loss_model.VGGPerceptualLoss().to(device)

    with open(BENCHMARK_PATH, "w") as f:
        with torch.no_grad():
            print(" , Baseline, Ours, CycleGAN, AdaptiveStyleNet", file=f)
            for i, (thesis_path) in enumerate(thesis_list):
                img_name = thesis_path.split("\\")[1]
                pred_path = PREDECESSOR_PATH + img_name
                cyclegan_path = CYCLEGAN_PATH + img_name
                eccv_path = ECCV_PATH + img_name

                pred_img = cv2.imread(pred_path)
                thesis_img = cv2.imread(thesis_path)
                cyclegan_img = cv2.imread(cyclegan_path)
                eccv_img = cv2.imread(eccv_path)

                pred_img = torch.unsqueeze(rgb_img_op(cv2.resize(pred_img, IMG_SIZE)), 0).to(device)
                thesis_img = torch.unsqueeze(rgb_img_op(cv2.resize(thesis_img, IMG_SIZE)), 0).to(device)
                cyclegan_img = torch.unsqueeze(rgb_img_op(cv2.resize(cyclegan_img, IMG_SIZE)), 0).to(device)
                eccv_img = torch.unsqueeze(rgb_img_op(cv2.resize(eccv_img, IMG_SIZE)), 0).to(device)

                m_perceptual_losses = [0.0, 0.0, 0.0, 0.0]
                for monet_path in monet_list:
                    #monet_img_name = monet_path.split("\\")[1]
                    monet_img = cv2.imread(monet_path)
                    monet_img = torch.unsqueeze(rgb_img_op(cv2.resize(monet_img, IMG_SIZE)), 0).to(device)

                    m_perceptual_losses[0] += np.round(vgg_loss(monet_img, pred_img).item(), 5)
                    m_perceptual_losses[1] += np.round(vgg_loss(monet_img, thesis_img).item(), 5)
                    m_perceptual_losses[2] += np.round(vgg_loss(monet_img, cyclegan_img).item(), 5)
                    m_perceptual_losses[3] += np.round(vgg_loss(monet_img, eccv_img).item(), 5)

                for i in range(0,len(m_perceptual_losses)):
                    m_perceptual_losses[i] = str(m_perceptual_losses[i] / len(monet_list) * 1.0)

                print(img_name + "," + m_perceptual_losses[0] + "," + m_perceptual_losses[1] + "," + m_perceptual_losses[2] + "," + m_perceptual_losses[3])
                print(img_name + "," + m_perceptual_losses[0] + "," + m_perceptual_losses[1] + "," + m_perceptual_losses[2] + "," + m_perceptual_losses[3], file = f)

def monet_psnr_ssim_loss():
    ALL_MONET_PATH = "G:/My Drive/CONFERENCES/WSCG 2021/Experiments/Monet Gallery numbered/"
    THESIS_PATH = "G:/My Drive/CONFERENCES/WSCG 2021/Experiments/FINAL PUSH/thesis output/"
    PREDECESSOR_PATH = "G:/My Drive/CONFERENCES/WSCG 2021/Experiments/FINAL PUSH/predecessor output/"
    CYCLEGAN_PATH = "G:/My Drive/CONFERENCES/WSCG 2021/Experiments/FINAL PUSH/PyTorch GAN output/"
    ECCV_PATH = "G:/My Drive/CONFERENCES/WSCG 2021/Experiments/FINAL PUSH/AdaptiveStyleTransfer-ECCV output/"
    BENCHMARK_PATH = "G:/My Drive/CONFERENCES/WSCG 2021/Experiments/perceptual_losses.txt"

    BENCHMARK_PATH = "G:/My Drive/CONFERENCES/WSCG 2021/Experiments/psnr_ssim_losses.txt"

    monet_list = glob.glob(ALL_MONET_PATH + "*.jpg")
    thesis_list = glob.glob(THESIS_PATH + "*.jpg")

    IMG_SIZE = (64, 64)

    with open(BENCHMARK_PATH, "w") as f:
        for i, (thesis_path) in enumerate(thesis_list):
            img_name = thesis_path.split("\\")[1]
            pred_path = PREDECESSOR_PATH + img_name
            cyclegan_path = CYCLEGAN_PATH + img_name
            eccv_path = ECCV_PATH + img_name

            pred_img = cv2.imread(pred_path)
            thesis_img = cv2.imread(thesis_path)
            cyclegan_img = cv2.imread(cyclegan_path)
            eccv_img = cv2.imread(eccv_path)

            pred_img = cv2.resize(pred_img, IMG_SIZE)
            thesis_img = cv2.resize(thesis_img, IMG_SIZE)
            cyclegan_img = cv2.resize(cyclegan_img, IMG_SIZE)
            eccv_img = cv2.resize(eccv_img, IMG_SIZE)

            psnr_list = [0.0, 0.0, 0.0, 0.0]
            ssim_list = [0.0, 0.0, 0.0, 0.0]

            for monet_path in monet_list:
                #monet_img_name = monet_path.split("\\")[1]
                monet_img = cv2.resize(cv2.imread(monet_path), IMG_SIZE)

                psnr_list[0] += np.round(peak_signal_noise_ratio(monet_img, pred_img), 4)
                psnr_list[1] += np.round(peak_signal_noise_ratio(monet_img, thesis_img), 4)
                psnr_list[2] += np.round(peak_signal_noise_ratio(monet_img, cyclegan_img), 4)
                psnr_list[3] += np.round(peak_signal_noise_ratio(monet_img, eccv_img), 4)

                ssim_list[0] += np.round(tensor_utils.measure_ssim(monet_img, pred_img), 4)
                ssim_list[1] += np.round(tensor_utils.measure_ssim(monet_img, thesis_img), 4)
                ssim_list[2] += np.round(tensor_utils.measure_ssim(monet_img, cyclegan_img), 4)
                ssim_list[3] += np.round(tensor_utils.measure_ssim(monet_img, eccv_img), 4)

            for i in range(0, len(psnr_list)):
                psnr_list[i] = str(psnr_list[i] / len(monet_list) * 1.0)
                ssim_list[i] = str(ssim_list[i] / len(monet_list) * 1.0)
            print("PSNR, " + img_name + " - Predecessor," + psnr_list[0])
            print("PSNR, " + img_name + " - Ours," + psnr_list[1])
            print("PSNR, " + img_name + " - CycleGAN," + psnr_list[2])
            print("PSNR, " + img_name + " - AdaptiveStyleTransfer," + psnr_list[3])

            print("SSIM, " + img_name + " - Predecessor," + ssim_list[0])
            print("SSIM, " + img_name + " - Ours," + ssim_list[1])
            print("SSIM, " + img_name + " - CycleGAN," + ssim_list[2])
            print("SSIM, " + img_name + " - AdaptiveStyleTransfer," + ssim_list[3])

            print("PSNR, " + img_name + " - Predecessor," + psnr_list[0], file = f)
            print("PSNR, " + img_name + " - Ours," + psnr_list[1], file = f)
            print("PSNR, " + img_name + " - CycleGAN," + psnr_list[2], file = f)
            print("PSNR, " + img_name + " - AdaptiveStyleTransfer," + psnr_list[3], file = f)

            print("SSIM, " + img_name + " - Predecessor," + ssim_list[0], file = f)
            print("SSIM, " + img_name + " - Ours," + ssim_list[1], file = f)
            print("SSIM, " + img_name + " - CycleGAN," + ssim_list[2], file = f)
            print("SSIM, " + img_name + " - AdaptiveStyleTransfer," + ssim_list[3], file = f)
def main():
    #monet_psnr_ssim_loss()
    monet_perceptual_loss()

#FIX for broken pipe num_workers issue.
if __name__=="__main__":
    main()