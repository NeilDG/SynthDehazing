import sys
from optparse import OptionParser
import glob
from utils import dehazing_proper, tensor_utils
import torch
from torchvision import utils as torchutils
import cv2

parser = OptionParser()
parser.add_option('--cuda_device', type=str, help="CUDA Device?", default="cuda:0")
parser.add_option('--directory', type=str, default="./input_images/")
parser.add_option('--image_type', type=str, default=".jpg")
parser.add_option('--output', type=str, default="./output/")
parser.add_option('--albedo_checkpt_name', type=str, default="albedo_transfer_v1.04_1")
parser.add_option('--t_checkpt_name', type=str, default="transmission_albedo_estimator_v1.16_6") #place in checkpoint
parser.add_option('--a_checkpt_name', type=str, default="airlight_estimator_v1.16_6")

def save_img(img, path):
    img = cv2.normalize(img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)

def main(argv):
    (opts, args) = parser.parse_args(argv)
    print(opts)

    model_dehazer = dehazing_proper.ModelDehazer()
    model_dehazer.set_models_v2(opts.albedo_checkpt_name, opts.t_checkpt_name, opts.a_checkpt_name)

    # print(opts.directory + "*." + opts.image_type)
    hazy_list = glob.glob(opts.directory + "*0.95_0.2.jpg")  # specify atmosphere intensity
    # hazy_list = glob.glob(opts.directory + "*" + opts.image_type)
    print(hazy_list)

    for i, (hazy_path) in enumerate(hazy_list):
        with torch.no_grad():
            img_name = hazy_path.split("\\")[1].split(".")[0]  # save new image as PNG
            hazy_img = tensor_utils.load_true_img(hazy_path)

            #to avoid banding issues. Input images must be square
            input_size = (1024, 1024)
            # im_size = (int(hazy_img.shape[1] / 3), int(hazy_img.shape[0] / 3))
            im_size = (hazy_img.shape[1], hazy_img.shape[0])
            hazy_img = cv2.resize(hazy_img, input_size, cv2.INTER_CUBIC)

            clear_img, T_tensor, A_tensor = model_dehazer.perform_dehazing_direct_v4(hazy_img, 0.0, True)

            clear_img = cv2.resize(clear_img, im_size, cv2.INTER_CUBIC)
            save_img(clear_img, opts.output + img_name + ".jpg")


if __name__ == "__main__":
    main(sys.argv)