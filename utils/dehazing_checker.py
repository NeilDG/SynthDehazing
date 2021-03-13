import constants
import torch
import numpy as np
from utils import plot_utils
from model import vanilla_cycle_gan as cycle_gan
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
import glob
from torchvision import transforms
import cv2
from utils import tensor_utils

class DehazingChecker:

    def __init__(self):
        self.gpu_device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        self.visdom_reporter = plot_utils.VisdomReporter()

        MODEL_CHECKPOINT = "transmission_estimator_v1.02_7"
        self.transmission_G = cycle_gan.Generator(input_nc=3, output_nc=1, n_residual_blocks=8).to(self.gpu_device)
        checkpt = torch.load('checkpoint/' + MODEL_CHECKPOINT + ".pt")
        self.transmission_G.load_state_dict(checkpt[constants.GENERATOR_KEY + "A"])
        print("Transmission GAN model loaded. ", MODEL_CHECKPOINT)

        self.batch_num = 0
        self.PSNR_KEY = "PSNR_KEY"
        self.SSIM_KEY = "SSIM_KEY"

        self.initialize_dict()

    def initialize_dict(self):
        # what to store in visdom?
        self.losses_dict = {}
        self.losses_dict[self.PSNR_KEY] = []
        self.losses_dict[self.SSIM_KEY] = []

        self.caption_dict = {}
        self.caption_dict[self.PSNR_KEY] = "PSNR per iteration"
        self.caption_dict[self.SSIM_KEY] = "SSIM per iteration"


    def check_performance(self):
        OHAZE_HAZY_PATH = "D:/Datasets/OTS_BETA/haze/"
        OHAZE_CLEAR_PATH = "D:/Datasets/OTS_BETA/clear/"
        IMAGE_LIMIT = 5

        hazy_list = glob.glob(OHAZE_HAZY_PATH + "*.jpg")
        gt_list = glob.glob(OHAZE_CLEAR_PATH + "*.jpg")

        rgb_img_op = transforms.Compose([transforms.ToPILImage(),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        average_PSNR = 0.0
        average_SSIM = 0.0

        count = 0
        for i, (hazy_path, gt_path) in enumerate(zip(hazy_list, gt_list)):
            with torch.no_grad():
                img_name = hazy_path.split("\\")[1]
                hazy_img = cv2.imread(hazy_path)
                # hazy_img = cv2.resize(hazy_img, (int(np.shape(hazy_img)[1] / 4), int(np.shape(hazy_img)[0] / 4)))
                hazy_img = cv2.resize(hazy_img, (512, 512))
                gt_img = cv2.imread(gt_path)
                # gt_img = cv2.resize(gt_img, (int(np.shape(gt_img)[1] / 4), int(np.shape(gt_img)[0] / 4)))
                gt_img = cv2.resize(gt_img, (512, 512))

                input_tensor = rgb_img_op(cv2.cvtColor(hazy_img, cv2.COLOR_BGR2RGB)).to(self.gpu_device)
                transmission_img = self.transmission_G(torch.unsqueeze(input_tensor, 0))
                transmission_img = torch.squeeze(transmission_img).cpu().numpy()

                # remove 0.5 normalization for dehazing equation
                transmission_img = ((transmission_img * 0.5) + 0.5)

                hazy_img = ((hazy_img * 0.5) + 0.5)
                clear_img = tensor_utils.perform_dehazing_equation_with_transmission(hazy_img, transmission_img,
                                                                                     tensor_utils.AtmosphereMethod.NETWORK_ESTIMATOR,
                                                                                     0.8)

                # normalize images
                hazy_img = cv2.normalize(hazy_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8U)
                clear_img = cv2.normalize(clear_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8U)

                # make images compatible with matplotlib
                hazy_img = cv2.cvtColor(hazy_img, cv2.COLOR_BGR2RGB)
                clear_img = cv2.cvtColor(clear_img, cv2.COLOR_BGR2RGB)
                gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)

                PSNR = np.round(peak_signal_noise_ratio(gt_img, clear_img), 4)
                SSIM = np.round(tensor_utils.measure_ssim(gt_img, clear_img), 4)

                average_PSNR += PSNR
                average_SSIM += SSIM

                count += 1
                if(count == IMAGE_LIMIT):
                    break

        average_PSNR = average_PSNR / count * 1.0
        average_SSIM = average_SSIM / count * 1.0

        self.losses_dict[self.PSNR_KEY].append(average_PSNR)
        self.losses_dict[self.SSIM_KEY].append(average_SSIM)


    def visdom_report(self, iteration):
        self.visdom_reporter.plot_psnr_ssim_loss("O-Haze Performance", iteration, self.losses_dict, self.caption_dict, self.PSNR_KEY)


