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


def color_transfer():
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    # load color transfer
    color_transfer_checkpt = torch.load('checkpoint/color_transfer_v1.11_2.pt')
    color_transfer_gan = cycle_gan.Generator(n_residual_blocks=10).to(device)
    color_transfer_gan.load_state_dict(color_transfer_checkpt[constants.GENERATOR_KEY + "A"])
    print("Color transfer GAN model loaded.")
    print("===================================================")

    dataloader = dataset_loader.load_test_dataset(constants.DATASET_CLEAN_PATH_COMPLETE, constants.DATASET_PLACES_PATH, constants.infer_size, -1)

    # Plot some training images
    name_batch, dirty_batch, clean_batch = next(iter(dataloader))
    plt.figure(figsize=constants.FIG_SIZE)
    plt.axis("off")
    plt.title("Training - Old Images")
    plt.imshow(np.transpose(vutils.make_grid(dirty_batch.to(device)[:constants.infer_size], nrow=8, padding=2, normalize=True).cpu(), (1, 2, 0)))
    plt.show()

    plt.figure(figsize=constants.FIG_SIZE)
    plt.axis("off")
    plt.title("Training - New Images")
    plt.imshow(np.transpose(vutils.make_grid(clean_batch.to(device)[:constants.infer_size], nrow=8, padding=2, normalize=True).cpu(), (1, 2, 0)))
    plt.show()

    item_number = 0
    for i, (name, dirty_batch, clean_batch) in enumerate(dataloader, 0):
        with torch.no_grad():
            input_tensor = dirty_batch.to(device)
            item_number = item_number + 1
            result = color_transfer_gan(input_tensor)
            show_images(input_tensor, "Input images: " + str(item_number))
            show_images(result, "Color transfer: " + str(item_number))

def save_img(img, path):
    img = cv2.normalize(img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)
    # cv2.imwrite(path, img, [cv2.IMWRITE_PNG_COMPRESSION, 9])

def main(argv):
    (opts, args) = parser.parse_args(argv)
    print(opts)

    model_dehazer = dehazing_proper.ModelDehazer()
    model_dehazer.set_models_v2(opts.albedo_checkpt_name, opts.t_checkpt_name, opts.a_checkpt_name)

    print(opts.directory)
    hazy_list = glob.glob(opts.directory)  # specify atmosphere intensity
    # hazy_list = glob.glob(opts.directory + "*" + opts.image_type)
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
            clear_img = cv2.resize(clear_img, im_size, cv2.INTER_CUBIC)

            print("Processed: ", (opts.output + img_name + ".png"))
            save_img(clear_img, opts.output + img_name + ".png")


if __name__ == "__main__":
    main(sys.argv)