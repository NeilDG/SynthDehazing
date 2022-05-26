import os
import sys
from optparse import OptionParser
import glob
from utils import dehazing_proper, tensor_utils
import torch
from torchvision import utils as vutils
import cv2
import constants
import matplotlib.pyplot as plt
import numpy as np
from model import vanilla_cycle_gan as cycle_gan
from model import unet_gan
from loaders import dataset_loader
import torchvision.transforms as transforms

parser = OptionParser()
parser.add_option('--cuda_device', type=str, help="CUDA Device?", default="cuda:0")
parser.add_option('--path', type=str, default="./input_images/*.jpg")
parser.add_option('--output', type=str, default="./output/")
parser.add_option('--checkpt_name', type=str, default="color_transfer_v1.11_2.pth")

def show_images(img_tensor, caption):
    plt.figure(figsize=(16, 4))
    plt.axis("off")
    plt.title(caption)
    plt.imshow(np.transpose(
        vutils.make_grid(img_tensor[:16], nrow=8, padding=2, normalize=True).cpu(),
        (1, 2, 0)))
    plt.show()

def save_img(img, path):
    img = cv2.normalize(img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)

def color_transfer(opts):
    os.makedirs(opts.output, exist_ok=True)

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    # load color transfer
    color_transfer_checkpt = torch.load('checkpoint/' + opts.checkpt_name)
    # color_transfer_gan = unet_gan.UnetGenerator(3, 3, num_downs=10).to(device)
    # Slight correction from paper: Our latest CycleGAN model, seems to produce acceptable results too.
    color_transfer_gan = cycle_gan.Generator(n_residual_blocks = 10, has_dropout=False).to(device)
    color_transfer_gan.load_state_dict(color_transfer_checkpt[constants.GENERATOR_KEY + "A"])
    print("Color transfer GAN model loaded.")
    print("===================================================")

    rgb_transform_op = transforms.Compose([transforms.ToPILImage(),
                                           transforms.Resize((256, 256)),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    img_list = glob.glob(opts.path)  # specify atmosphere intensity
    print(img_list)

    for i, (img_path) in enumerate(img_list):
        with torch.no_grad():
            img_name = img_path.split("\\")[1].split(".")[0]  # save new image as PNG
            input_tensor = rgb_transform_op(tensor_utils.load_true_img(img_path)).to(device)
            input_tensor = torch.unsqueeze(input_tensor, 0)

            styled_tensor = color_transfer_gan(input_tensor)
            styled_tensor = torch.squeeze(styled_tensor)
            styled_tensor = (styled_tensor * 0.5) + 0.5 #move to 0-1 normalization

            print("Processed: ", (opts.output + img_name + ".png"))
            vutils.save_image(styled_tensor, (opts.output + img_name + ".png"))

def main(argv):
    (opts, args) = parser.parse_args(argv)
    print(opts)

    color_transfer(opts)


if __name__ == "__main__":
    main(sys.argv)