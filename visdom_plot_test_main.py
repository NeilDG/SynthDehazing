from optparse import OptionParser
import random
import sys
import constants
import torch
from utils import plot_utils
from loaders import dataset_loader

parser = OptionParser()
parser.add_option('--coare', type=int, help="Is running on COARE?", default=0)
parser.add_option('--img_to_load', type=int, help="Image to load?", default=-1)
parser.add_option('--image_size', type=int, help="Weight", default="256")
parser.add_option('--batch_size', type=int, help="Weight", default="64")

def update_config(opts):
    constants.is_coare = opts.coare

    if (constants.is_coare == 1):
        print("Using COARE configuration.")
        constants.LIGHTCOORDS_ESTIMATOR_CHECKPATH = 'checkpoint/' + constants.LIGHTS_ESTIMATOR_VERSION + "_" + constants.ITERATION + '.pt'
        constants.DATASET_HAZY_PATH_COMPLETE = "/scratch1/scratch2/neil.delgallego/Synth Hazy 2/hazy/"
        constants.DATASET_DEPTH_PATH_COMPLETE = "/scratch1/scratch2/neil.delgallego/Synth Hazy 2/depth/"
        constants.DATASET_CLEAN_PATH_COMPLETE = "/scratch1/scratch2/neil.delgallego/Synth Hazy 2/clean/"
        constants.DATASET_CLEAN_PATH_COMPLETE_STYLED = "/scratch1/scratch2/neil.delgallego/Synth Hazy 2/clean - styled/"
        constants.DATASET_LIGHTCOORDS_PATH_COMPLETE = "/scratch1/scratch2/neil.delgallego/Synth Hazy 2/light/"
        constants.DATASET_VEMON_PATH_COMPLETE = "/scratch1/scratch2/neil.delgallego/VEMON Dataset/frames/"
        constants.DATASET_PLACES_PATH = "/scratch1/scratch2/neil.delgallego/Places Dataset/"
        constants.DATASET_CLEAN_PATH_COMPLETE_STYLED_3 = "/scratch1/scratch2/neil.delgallego/Synth Hazy 3/clean/"
        constants.DATASET_ALBEDO_PATH_COMPLETE_3 = "/scratch1/scratch2/neil.delgallego/Synth Hazy 3/albedo/"

        constants.num_workers = 4

def main(argv):
    (opts, args) = parser.parse_args(argv)
    update_config(opts)
    print("=====================BEGIN============================")
    print("Is Coare? %d Has GPU available? %d Count: %d Torch CUDA version: %s"
          % (constants.is_coare, torch.cuda.is_available(), torch.cuda.device_count(), torch.version.cuda))

    manualSeed = 1  # set this for experiments and promoting fixed results
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    print("Device: %s" % device)

    train_loader = dataset_loader.load_model_based_transmission_dataset(constants.DATASET_CLEAN_PATH_COMPLETE_STYLED, constants.DATASET_DEPTH_PATH_COMPLETE, constants.DATASET_LIGHTCOORDS_PATH_COMPLETE,
                                                                        constants.TEST_IMAGE_SIZE, constants.batch_size, opts.img_to_load)
    visdom_reporter = plot_utils.VisdomReporter()

    for i, train_data in enumerate(train_loader, 0):
        _, rgb_batch, _, _, _ = train_data
        rgb_tensor = rgb_batch.to(device).float()
        visdom_reporter.plot_image((rgb_tensor), "Training RGB images - " +str(i % 10))

    print("Successfully plotted visdom image")

# FIX for broken pipe num_workers issue.
if __name__ == "__main__":
    main(sys.argv)