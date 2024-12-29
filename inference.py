import os
import sys
from optparse import OptionParser
import glob
from utils import dehazing_proper, tensor_utils
import torch
from torchvision import utils as torchutils
import cv2

parser = OptionParser()
parser.add_option('--cuda_device', type=str, help="CUDA Device?", default="cuda:0")
parser.add_option('--path', type=str, default="./input_images/*.jpg")
parser.add_option('--output', type=str, default="./output/")
parser.add_option('--albedo_checkpt_name', type=str, default="albedo_transfer_v1.04_1")
parser.add_option('--t_checkpt_name', type=str, default="transmission_albedo_estimator_v1.16_6") #place in checkpoint
parser.add_option('--a_checkpt_name', type=str, default="airlight_estimator_v1.16_6")
parser.add_option('--repeats', type=int, default=1)

def save_img(img, path):
    img = cv2.normalize(img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)

def main(argv):
    (opts, args) = parser.parse_args(argv)
    print(opts)

    os.makedirs(opts.output, exist_ok=True)

    hazy_copy_dir = "./output/hazy/I-Haze/"
    os.makedirs(hazy_copy_dir, exist_ok=True)

    model_dehazer = dehazing_proper.ModelDehazer()
    model_dehazer.set_models(opts.albedo_checkpt_name, opts.t_checkpt_name, opts.a_checkpt_name)

    hazy_list = glob.glob(opts.path)  # specify atmosphere intensity
    print(hazy_list)

    for i, (hazy_path) in enumerate(hazy_list):
        with torch.no_grad():
            img_name = hazy_path.split("\\")[1].split(".")[0]  # save new image as PNG
            hazy_img = tensor_utils.load_true_img(hazy_path)

            #to avoid banding issues. Input images must be square
            input_size = (1024, 1024)
            im_size = (hazy_img.shape[1], hazy_img.shape[0])
            hazy_img = cv2.resize(hazy_img, input_size, cv2.INTER_CUBIC)

            clear_img, T_tensor, A_tensor = model_dehazer.perform_dehazing_direct_v4(hazy_img, 0.0, True)
            for j in range(1, opts.repeats): #iterative dehazing
                clear_img, T_tensor, A_tensor = model_dehazer.perform_dehazing_direct_v4(clear_img, 0.0, True)

            hazy_img = cv2.resize(hazy_img, im_size, cv2.INTER_CUBIC)
            clear_img = cv2.resize(clear_img, im_size, cv2.INTER_CUBIC)

            print("Processed: ", (opts.output + img_name + ".png"))
            save_img(hazy_img, hazy_copy_dir + img_name + ".png")
            save_img(clear_img, opts.output + img_name + ".png")


if __name__ == "__main__":
    main(sys.argv)