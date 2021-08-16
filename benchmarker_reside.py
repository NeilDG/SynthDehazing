import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt
from model import vanilla_cycle_gan as cycle_gan
from model import unet_gan as un
import constants
from torchvision import transforms
from torchvision import utils as torchutils
import cv2
from utils import tensor_utils
from utils import dark_channel_prior
from utils import dehazing_proper
import glob
from skimage.metrics import peak_signal_noise_ratio

def produce_reside(T_CHECKPT_NAME, A_ESTIMATOR_NAME):
    HAZY_PATH = "E:/Hazy Dataset Benchmark/RESIDE-Unannotated/"
    SAVE_PATH = "results/Ours - Results - RESIDE-3/"
    SAVE_TRANSMISSION_PATH = "results/Ours - Results - RESIDE-3/Transmission/"
    SAVE_ATMOSPHERE_PATH = "results/Ours - Results - RESIDE-3/Atmosphere/"

    hazy_list = glob.glob(HAZY_PATH + "*.jpeg")

    ALBEDO_CHECKPT = "albedo_transfer_v1.04_1"
    TRANSMISSION_CHECKPT = T_CHECKPT_NAME
    AIRLIGHT_ESTIMATOR_CHECKPT = A_ESTIMATOR_NAME

    model_dehazer = dehazing_proper.ModelDehazer()
    model_dehazer.set_models_v2(ALBEDO_CHECKPT, TRANSMISSION_CHECKPT, AIRLIGHT_ESTIMATOR_CHECKPT)

    for i, (hazy_path) in enumerate(hazy_list):
        with torch.no_grad():
            img_name = hazy_path.split("\\")[1]
            hazy_img = cv2.imread(hazy_path)
            hazy_img = cv2.resize(hazy_img, (512, 512))

            clear_img, T_tensor, A_tensor = model_dehazer.perform_dehazing_direct_v4(hazy_img, 0.0)
            clear_img = cv2.normalize(clear_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            cv2.imwrite(SAVE_PATH + img_name, clear_img)

            torchutils.save_image(T_tensor, SAVE_TRANSMISSION_PATH + img_name)
            torchutils.save_image(A_tensor, SAVE_ATMOSPHERE_PATH + img_name)

            print("Saved: " + SAVE_PATH + img_name)

def benchmark_reside():
    HAZY_PATH = "E:/Hazy Dataset Benchmark/RESIDE-Unannotated/"

    AOD_RESULTS_PATH = "results/AODNet- Results - RESIDE-3/"
    FFA_RESULTS_PATH = "results/FFA Net - Results - RESIDE-3/"
    GRID_DEHAZE_RESULTS_PATH = "results/GridDehazeNet - Results - RESIDE-3/"
    CYCLE_DEHAZE_PATH = "results/CycleDehaze - Results - RESIDE-3/"
    EDPN_DEHAZE_PATH = "results/EDPN - Results - RESIDE-3/"

    EXPERIMENT_NAME = "metrics - 1"
    TRANSMISSION_CHECKPT = "checkpoint/transmission_albedo_estimator_v1.06_1.pt"
    AIRLIGHT_CHECKPT = "checkpoint/airlight_estimator_v1.05_1.pt"

    SAVE_PATH = "results/RESIDE-3/"
    BENCHMARK_PATH = SAVE_PATH + EXPERIMENT_NAME + ".txt"

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    hazy_list = glob.glob(HAZY_PATH + "*.jpeg") #specify atmosphere intensity
    aod_list = glob.glob(AOD_RESULTS_PATH + "*.jpeg")
    ffa_list = glob.glob(FFA_RESULTS_PATH + "*.jpg")
    grid_list = glob.glob(GRID_DEHAZE_RESULTS_PATH + "*.jpeg")
    cycle_dh_list = glob.glob(CYCLE_DEHAZE_PATH + "*.jpeg")
    edpn_list = glob.glob(EDPN_DEHAZE_PATH + "*.png")

    print("Found images: ", len(hazy_list), len(aod_list), len(ffa_list), len(grid_list), len(cycle_dh_list), len(edpn_list))

    gray_img_op = transforms.Compose([transforms.ToPILImage(),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5), (0.5))])

    rgb_img_op = transforms.Compose([transforms.ToPILImage(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    transmission_G = cycle_gan.Generator(input_nc=3, output_nc=1, n_residual_blocks=8).to(device)
    #transmission_G = un.UnetGenerator(input_nc=3, output_nc=1, num_downs=8).to(device)
    checkpt = torch.load(TRANSMISSION_CHECKPT)
    transmission_G.load_state_dict(checkpt[constants.GENERATOR_KEY + "A"])
    print("Transmission GAN model loaded.")

    FIG_ROWS = 8
    FIG_COLS = 4
    FIG_WIDTH = 10
    FIG_HEIGHT = 20
    fig, ax = plt.subplots(ncols=FIG_COLS, nrows=FIG_ROWS, constrained_layout=True, sharex=True)
    fig.set_size_inches(FIG_WIDTH, FIG_HEIGHT)
    column = 0
    fig_num = 0
    average_SSIM = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    average_PSNR = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    count = 0

    with open(BENCHMARK_PATH, "w") as f:
        for i, (hazy_path, ffa_path, grid_path, cycle_dh_path, aod_path, edpn_path) in \
                enumerate(zip(hazy_list, ffa_list, grid_list, cycle_dh_list, aod_list, edpn_list)):
            with torch.no_grad():
                count = count + 1
                img_name = hazy_path.split("\\")[1]

                hazy_img = cv2.imread(hazy_path)
                hazy_img = cv2.resize(hazy_img, (512, 512))

                aod_img = cv2.imread(aod_path)
                aod_img = cv2.resize(aod_img, (int(np.shape(hazy_img)[1]), int(np.shape(hazy_img)[0])))

                edpn_img = cv2.imread(edpn_path)
                edpn_img = cv2.resize(edpn_img, (int(np.shape(hazy_img)[1]), int(np.shape(hazy_img)[0])))

                ffa_img = cv2.imread(ffa_path)
                ffa_img = cv2.resize(ffa_img, (int(np.shape(hazy_img)[1]), int(np.shape(hazy_img)[0])))

                grid_img = cv2.imread(grid_path)
                grid_img = cv2.resize(grid_img, (int(np.shape(hazy_img)[1]), int(np.shape(hazy_img)[0])))

                cycle_dehaze_img = cv2.imread(cycle_dh_path)
                cycle_dehaze_img = cv2.resize(cycle_dehaze_img, (int(np.shape(hazy_img)[1]), int(np.shape(hazy_img)[0])))

                # input_tensor = gray_img_op(cv2.cvtColor(hazy_img, cv2.COLOR_BGR2GRAY)).to(device)
                input_tensor = rgb_img_op(cv2.cvtColor(hazy_img, cv2.COLOR_BGR2RGB)).to(device)
                transmission_img = transmission_G(torch.unsqueeze(input_tensor, 0))
                transmission_img = torch.squeeze(transmission_img).cpu().numpy()

                # remove 0.5 normalization for dehazing equation
                transmission_img = 1 - ((transmission_img * 0.5) + 0.5)

                hazy_img = ((hazy_img * 0.5) + 0.5)
                dark_channel = dark_channel_prior.get_dark_channel(hazy_img, 15)
                dcp_transmission = dark_channel_prior.estimate_transmission(hazy_img, dark_channel_prior.estimate_atmosphere(hazy_img, dark_channel),
                                                                            dark_channel)

                # DCP is not 0-1 range
                dcp_transmission = cv2.normalize(dcp_transmission, dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,
                                                 dtype=cv2.CV_32F)
                transmission_blend = dcp_transmission * 0.0 + transmission_img * 1.0

                dcp_clear_img = dark_channel_prior.perform_dcp_dehaze(hazy_img, True)
                clear_img = dehazing_proper.perform_dehazing_equation_with_transmission(hazy_img, transmission_blend, dehazing_proper.AtmosphereMethod.NETWORK_ESTIMATOR_V1, AIRLIGHT_CHECKPT, 0.8)

                # normalize images
                hazy_img = cv2.normalize(hazy_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                clear_img = cv2.normalize(clear_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                dcp_clear_img = cv2.normalize(dcp_clear_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                ffa_img = cv2.normalize(ffa_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                grid_img = cv2.normalize(grid_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                cycle_dehaze_img = cv2.normalize(cycle_dehaze_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                aod_img = cv2.normalize(aod_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8U)
                edpn_img = cv2.normalize(edpn_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

                # make images compatible with matplotlib
                hazy_img = cv2.cvtColor(hazy_img, cv2.COLOR_BGR2RGB)
                clear_img = cv2.cvtColor(clear_img, cv2.COLOR_BGR2RGB)
                dcp_clear_img = cv2.cvtColor(dcp_clear_img, cv2.COLOR_BGR2RGB)
                ffa_img = cv2.cvtColor(ffa_img, cv2.COLOR_BGR2RGB)
                grid_img = cv2.cvtColor(grid_img, cv2.COLOR_BGR2RGB)
                cycle_dehaze_img = cv2.cvtColor(cycle_dehaze_img, cv2.COLOR_BGR2RGB)
                aod_img = cv2.cvtColor(aod_img, cv2.COLOR_BGR2RGB)
                edpn_img = cv2.cvtColor(edpn_img, cv2.COLOR_BGR2RGB)


                ax[0, column].imshow(hazy_img)
                ax[0, column].axis('off')
                ax[1, column].imshow(dcp_clear_img)
                ax[1, column].axis('off')
                ax[2, column].imshow(aod_img)
                ax[2, column].axis('off')
                ax[3, column].imshow(cycle_dehaze_img)
                ax[3, column].axis('off')
                ax[4, column].imshow(edpn_img)
                ax[4, column].axis('off')
                ax[5, column].imshow(ffa_img)
                ax[5, column].axis('off')
                ax[6, column].imshow(grid_img)
                ax[6, column].axis('off')
                ax[7, column].imshow(clear_img)
                ax[7, column].axis('off')

                column = column + 1

                if (column == FIG_COLS):
                    fig_num = fig_num + 1
                    file_name = SAVE_PATH + "fig_" + str(fig_num) + "_" + EXPERIMENT_NAME + ".jpg"
                    plt.savefig(file_name)
                    #plt.show()

                    # create new figure
                    fig, ax = plt.subplots(ncols=FIG_COLS, nrows=FIG_ROWS, constrained_layout=True, sharex=True)
                    fig.set_size_inches(FIG_WIDTH, FIG_HEIGHT)
                    column = 0

def main():
    CHECKPT_NAME = "dehazer_v2.07_3"

    produce_reside(CHECKPT_NAME, "airlight_estimator_v1.08_1")
    #benchmark_reside()

# FIX for broken pipe num_workers issue.
if __name__ == "__main__":
    main()