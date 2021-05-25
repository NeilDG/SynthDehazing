import kornia
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt
from model import vanilla_cycle_gan as cycle_gan
from model import unet_gan as un
import constants
from torchvision import transforms
import cv2
from utils import tensor_utils
from utils import dark_channel_prior
from utils import dehazing_proper
import glob
from skimage.metrics import peak_signal_noise_ratio
from custom_losses import ssim_loss

def benchmark_ohaze():
    HAZY_PATH = "E:/Hazy Dataset Benchmark/O-HAZE/hazy/"
    GT_PATH = "E:/Hazy Dataset Benchmark/O-HAZE/GT/"
    #HAZY_PATH = constants.DATASET_HAZY_PATH_COMPLETE
    #GT_PATH = constants.DATASET_CLEAN_PATH_COMPLETE

    AOD_RESULTS_PATH = "results/AODNet- Results - OHaze/"
    FFA_RESULTS_PATH = "results/FFA Net - Results - OHaze/"
    GRID_DEHAZE_RESULTS_PATH = "results/GridDehazeNet - Results - OHaze/"
    CYCLE_DEHAZE_PATH = "results/CycleDehaze - Results - OHaze/"
    EDPN_DEHAZE_PATH = "results/EDPN - Results - OHaze/"

    EXPERIMENT_NAME = "metrics - 1"
    TRANSMISSION_CHECKPT = "checkpoint/transmission_albedo_estimator_v1.04_2.pt"
    AIRLIGHT_CHECKPT = "checkpoint/airlight_estimator_v1.04_1.pt"

    SAVE_PATH = "results/O-HAZE/"
    BENCHMARK_PATH = SAVE_PATH + EXPERIMENT_NAME + ".txt"

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    hazy_list = glob.glob(HAZY_PATH + "*.jpg")
    gt_list = glob.glob(GT_PATH + "*.jpg")
    aod_list = glob.glob(AOD_RESULTS_PATH + "*.jpg")
    ffa_list = glob.glob(FFA_RESULTS_PATH + "*.png")
    grid_list = glob.glob(GRID_DEHAZE_RESULTS_PATH + "*.jpg")
    cycle_dh_list = glob.glob(CYCLE_DEHAZE_PATH + "*.jpg")
    edpn_list = glob.glob(EDPN_DEHAZE_PATH + "*.png")

    print(hazy_list)
    print(gt_list)

    gray_img_op = transforms.Compose([transforms.ToPILImage(),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5), (0.5))])

    rgb_img_op = transforms.Compose([transforms.ToPILImage(),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    ssim_op = transforms.Compose([transforms.ToPILImage(),
                                   transforms.ToTensor()])

    transmission_G = cycle_gan.Generator(input_nc=3, output_nc=1, n_residual_blocks=8).to(device)
    #transmission_G = un.UnetGenerator(input_nc= 3, output_nc = 1, num_downs = 8).to(self.gpu_device)
    checkpt = torch.load(TRANSMISSION_CHECKPT)
    transmission_G.load_state_dict(checkpt[constants.GENERATOR_KEY + "A"])
    print("Transmission GAN model loaded.")

    FIG_ROWS = 9
    FIG_COLS = 4
    FIG_WIDTH = 10
    FIG_HEIGHT = 25
    fig, ax = plt.subplots(ncols=FIG_COLS, nrows=FIG_ROWS, constrained_layout=True, sharex=True)
    fig.set_size_inches(FIG_WIDTH, FIG_HEIGHT)
    column = 0
    fig_num = 0
    average_SSIM = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    average_PSNR = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    count = 0

    with open(BENCHMARK_PATH, "w") as f:
        for i, (hazy_path, gt_path, ffa_path, grid_path, cycle_dh_path, aod_path, edpn_path) in \
                enumerate(zip(hazy_list, gt_list, ffa_list, grid_list, cycle_dh_list, aod_list, edpn_list)):
            with torch.no_grad():
                count = count + 1
                img_name = hazy_path.split("\\")[1]
                hazy_img = cv2.imread(hazy_path)
                hazy_img = cv2.resize(hazy_img, (512, 512))

                gt_img = cv2.imread(gt_path)
                gt_img = cv2.resize(gt_img, (512, 512))

                aod_img = cv2.imread(aod_path)
                aod_img = cv2.resize(aod_img, (int(np.shape(gt_img)[1]), int(np.shape(gt_img)[0])))

                edpn_img = cv2.imread(edpn_path)
                edpn_img = cv2.resize(edpn_img, (int(np.shape(gt_img)[1]), int(np.shape(gt_img)[0])))

                ffa_img = cv2.imread(ffa_path)
                ffa_img = cv2.resize(ffa_img, (int(np.shape(gt_img)[1]), int(np.shape(gt_img)[0])))

                grid_img = cv2.imread(grid_path)
                grid_img = cv2.resize(grid_img, (int(np.shape(gt_img)[1]), int(np.shape(gt_img)[0])))

                cycle_dehaze_img = cv2.imread(cycle_dh_path)
                cycle_dehaze_img = cv2.resize(cycle_dehaze_img, (int(np.shape(gt_img)[1]), int(np.shape(gt_img)[0])))

                input_tensor = rgb_img_op(cv2.cvtColor(hazy_img, cv2.COLOR_BGR2RGB)).to(device)
                transmission_img = transmission_G(torch.unsqueeze(input_tensor, 0))
                transmission_img = torch.squeeze(transmission_img).cpu().numpy()

                # remove 0.5 normalization for dehazing equation
                transmission_img = ((transmission_img * 0.5) + 0.5)


                dark_channel = dark_channel_prior.get_dark_channel(hazy_img, 15)
                dcp_transmission = dark_channel_prior.estimate_transmission(hazy_img, dark_channel_prior.estimate_atmosphere(hazy_img, dark_channel),
                                                                            dark_channel)
                # DCP is not 0-1 range
                dcp_transmission = cv2.normalize(dcp_transmission, dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                transmission_blend = dcp_transmission * 0.0 + transmission_img * 1.0

                dcp_clear_img = dark_channel_prior.perform_dcp_dehaze(hazy_img, True)
                clear_img = dehazing_proper.perform_dehazing_equation_with_transmission(hazy_img, transmission_blend, dehazing_proper.AtmosphereMethod.NETWORK_ESTIMATOR_V1, AIRLIGHT_CHECKPT, 0.8)

                #normalize images
                hazy_img = cv2.normalize(hazy_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                clear_img = cv2.normalize(clear_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8U)
                dcp_clear_img = cv2.normalize(dcp_clear_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                ffa_img = cv2.normalize(ffa_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                grid_img = cv2.normalize(grid_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                cycle_dehaze_img = cv2.normalize(cycle_dehaze_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                aod_img = cv2.normalize(aod_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                edpn_img = cv2.normalize(edpn_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                gt_img = cv2.normalize(gt_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

                #make images compatible with matplotlib
                hazy_img = cv2.cvtColor(hazy_img, cv2.COLOR_BGR2RGB)
                clear_img = cv2.cvtColor(clear_img, cv2.COLOR_BGR2RGB)
                dcp_clear_img = cv2.cvtColor(dcp_clear_img, cv2.COLOR_BGR2RGB)
                ffa_img = cv2.cvtColor(ffa_img, cv2.COLOR_BGR2RGB)
                grid_img = cv2.cvtColor(grid_img, cv2.COLOR_BGR2RGB)
                cycle_dehaze_img = cv2.cvtColor(cycle_dehaze_img, cv2.COLOR_BGR2RGB)
                aod_img = cv2.cvtColor(aod_img, cv2.COLOR_BGR2RGB)
                edpn_img = cv2.cvtColor(edpn_img, cv2.COLOR_BGR2RGB)
                gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)

                # measure PSNR
                PSNR = np.round(peak_signal_noise_ratio(gt_img, dcp_clear_img), 4)
                print("[DCP] PSNR of ", img_name, " : ", PSNR, file=f)
                average_PSNR[0] += PSNR

                PSNR = np.round(peak_signal_noise_ratio(gt_img, aod_img), 4)
                print("[AOD-Net] PSNR of ", img_name, " : ", PSNR, file=f)
                average_PSNR[1] += PSNR

                PSNR = np.round(peak_signal_noise_ratio(gt_img, cycle_dehaze_img), 4)
                print("[CycleDehaze] PSNR of ", img_name, " : ", PSNR, file=f)
                average_PSNR[2] += PSNR

                PSNR = np.round(peak_signal_noise_ratio(gt_img, ffa_img), 4)
                print("[FFA-Net] PSNR of ", img_name, " : ", PSNR, file=f)
                average_PSNR[3] += PSNR

                PSNR = np.round(peak_signal_noise_ratio(gt_img, grid_img), 4)
                print("[GridDehazeNet] PSNR of ", img_name, " : ", PSNR, file=f)
                average_PSNR[4] += PSNR

                PSNR = np.round(peak_signal_noise_ratio(gt_img, edpn_img), 4)
                print("[EDPN] PSNR of ", img_name, " : ", PSNR, file=f)
                average_PSNR[5] += PSNR

                PSNR = np.round(peak_signal_noise_ratio(gt_img, clear_img), 4)
                print("[Ours] PSNR of ", img_name, " : ", PSNR, file=f)
                average_PSNR[6] += PSNR

                # measure SSIM
                SSIM = np.round(tensor_utils.measure_ssim(gt_img, dcp_clear_img), 4)
                print("[DCP] SSIM of ", img_name, " : ", SSIM, file=f)
                average_SSIM[0] += SSIM

                SSIM = np.round(tensor_utils.measure_ssim(gt_img, aod_img), 4)
                print("[AOD-Net] SSIM of ", img_name, " : ", SSIM, file=f)
                average_SSIM[1] += SSIM

                SSIM = np.round(tensor_utils.measure_ssim(gt_img, cycle_dehaze_img), 4)
                print("[CycleDehaze] SSIM of ", img_name, " : ", SSIM, file=f)
                average_SSIM[2] += SSIM

                SSIM = np.round(tensor_utils.measure_ssim(gt_img, ffa_img), 4)
                print("[FFA-Net] SSIM of ", img_name, " : ", SSIM, file=f)
                average_SSIM[3] += SSIM

                SSIM = np.round(tensor_utils.measure_ssim(gt_img, grid_img), 4)
                print("[GridDehazeNet] SSIM of ", img_name, " : ", SSIM, file=f)
                average_SSIM[4] += SSIM

                SSIM = np.round(tensor_utils.measure_ssim(gt_img, edpn_img), 4)
                print("[EDPN] SSIM of ", img_name, " : ", SSIM, file=f)
                average_SSIM[5] += SSIM

                SSIM = np.round(tensor_utils.measure_ssim(gt_img, clear_img), 4)
                print("[Ours] SSIM of " ,img_name," : ", SSIM, file = f)
                average_SSIM[6] += SSIM

                print(file = f)

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
                ax[8, column].imshow(gt_img)
                ax[8, column].axis('off')

                column = column + 1

                if (column == FIG_COLS):
                    fig_num = fig_num + 1
                    file_name = SAVE_PATH + "fig_" + str(fig_num) + "_" + EXPERIMENT_NAME + ".jpg"
                    plt.savefig(file_name)
                    plt.show()

                    # create new figure
                    fig, ax = plt.subplots(ncols=FIG_COLS, nrows=FIG_ROWS, constrained_layout=True, sharex=True)
                    fig.set_size_inches(FIG_WIDTH, FIG_HEIGHT)
                    column = 0

        for i in range(len(average_SSIM)):
            average_SSIM[i] = average_SSIM[i] / count * 1.0
            average_PSNR[i] = average_PSNR[i] / count * 1.0

        print(file = f)
        print("[DCP] Average PSNR: ", np.round(average_PSNR[0], 5), file=f)
        print("[AOD-Net] Average PSNR: ", np.round(average_PSNR[1], 5), file=f)
        print("[CycleDehaze] Average PSNR: ", np.round(average_PSNR[2], 5), file=f)
        print("[FFA-Net] Average PSNR: ", np.round(average_PSNR[3], 5), file=f)
        print("[GridDehazeNet] Average PSNR: ", np.round(average_PSNR[4], 5), file=f)
        print("[EDPN] Average PSNR: ", np.round(average_PSNR[5], 5), file=f)
        print("[Ours] Average PSNR: ", np.round(average_PSNR[6], 5), file=f)
        print(file = f)
        print("[DCP] Average SSIM: ", np.round(average_SSIM[0], 5), file=f)
        print("[AOD-Net] Average SSIM: ", np.round(average_SSIM[1], 5), file=f)
        print("[CycleDehaze] Average SSIM: ", np.round(average_SSIM[2], 5), file=f)
        print("[FFA-Net] Average SSIM: ", np.round(average_SSIM[3], 5), file = f)
        print("[GridDehazeNet] Average SSIM: ", np.round(average_SSIM[4], 5), file=f)
        print("[EDPN] Average SSIM: ", np.round(average_SSIM[5], 5), file=f)
        print("[Ours] Average SSIM: ", np.round(average_SSIM[6], 5), file=f)

def benchmark_ohaze_inmodels():
    HAZY_PATH = "E:/Hazy Dataset Benchmark/O-HAZE/hazy/"
    GT_PATH = "E:/Hazy Dataset Benchmark/O-HAZE/GT/"
    MODEL_CHECKPOINT = "transmission_estimator_v1.02_2"

    SAVE_PATH = "results/O-HAZE/"
    BENCHMARK_PATH = SAVE_PATH + "metrics - " + str(MODEL_CHECKPOINT) + "_inmodels.txt"

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    hazy_list = glob.glob(HAZY_PATH + "*.jpg")
    gt_list = glob.glob(GT_PATH + "*.jpg")

    print(hazy_list)
    print(gt_list)

    rgb_img_op = transforms.Compose([transforms.ToPILImage(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    transmission_G = cycle_gan.Generator(input_nc=3, output_nc=1, n_residual_blocks=8).to(device)
    checkpt = torch.load('checkpoint/' + MODEL_CHECKPOINT + ".pt")
    transmission_G.load_state_dict(checkpt[constants.GENERATOR_KEY + "A"])
    print("Transmission GAN model loaded.")

    FIG_ROWS = 6
    FIG_COLS = 4
    FIG_WIDTH = 10
    FIG_HEIGHT = 20
    fig, ax = plt.subplots(ncols=FIG_COLS, nrows=FIG_ROWS, constrained_layout=True, sharex=True)
    fig.set_size_inches(FIG_WIDTH, FIG_HEIGHT)
    column = 0
    fig_num = 0
    average_SSIM = [0.0, 0.0, 0.0, 0.0]
    average_PSNR = [0.0, 0.0, 0.0, 0.0]
    count = 0

    with open(BENCHMARK_PATH, "w") as f:
        for i, (hazy_path, gt_path) in enumerate(zip(hazy_list, gt_list)):
            with torch.no_grad():
                count = count + 1
                img_name = hazy_path.split("\\")[1]
                hazy_img = cv2.imread(hazy_path)
                hazy_img = cv2.resize(hazy_img, (512, 512))
                gt_img = cv2.imread(gt_path)
                gt_img = cv2.resize(gt_img, (512, 512))

                input_tensor = rgb_img_op(cv2.cvtColor(hazy_img, cv2.COLOR_BGR2RGB)).to(device)
                transmission_img = transmission_G(torch.unsqueeze(input_tensor, 0))
                transmission_img = torch.squeeze(transmission_img).cpu().numpy()

                # remove 0.5 normalization for dehazing equation
                transmission_img = ((transmission_img * 0.5) + 0.5)

                hazy_img = ((hazy_img * 0.5) + 0.5)
                dark_channel = dark_channel_prior.get_dark_channel(hazy_img, 15)
                dcp_transmission = dark_channel_prior.estimate_transmission(hazy_img,
                                                                            dark_channel_prior.estimate_atmosphere(
                                                                                hazy_img, dark_channel),
                                                                            dark_channel)
                # DCP is not 0-1 range
                dcp_transmission = cv2.normalize(dcp_transmission, dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,
                                                 dtype=cv2.CV_32F)
                transmission_blend = dcp_transmission * 0.0 + transmission_img * 1.0

                dcp_clear_img = dark_channel_prior.perform_dcp_dehaze(hazy_img, True)
                clear_img_v1 = dehazing_proper.perform_dehazing_equation_with_transmission(hazy_img, transmission_blend, dehazing_proper.AtmosphereMethod.SCENE_RADIANCE, 0.8)
                clear_img_v2 = dehazing_proper.perform_dehazing_equation_with_transmission(hazy_img, transmission_blend, dehazing_proper.AtmosphereMethod.NETWORK_ESTIMATOR_V1, 0.8)
                clear_img_v3 = dehazing_proper.perform_dehazing_equation_with_transmission(hazy_img, transmission_blend, dehazing_proper.AtmosphereMethod.NETWORK_ESTIMATOR_V2, 0.8)

                # normalize images
                hazy_img = cv2.normalize(hazy_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                clear_img_v1 = cv2.normalize(clear_img_v1, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                clear_img_v2 = cv2.normalize(clear_img_v2, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                clear_img_v3 = cv2.normalize(clear_img_v3, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                dcp_clear_img = cv2.normalize(dcp_clear_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                gt_img = cv2.normalize(gt_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

                # make images compatible with matplotlib
                hazy_img = cv2.cvtColor(hazy_img, cv2.COLOR_BGR2RGB)
                clear_img_v1 = cv2.cvtColor(clear_img_v1, cv2.COLOR_BGR2RGB)
                clear_img_v2 = cv2.cvtColor(clear_img_v2, cv2.COLOR_BGR2RGB)
                clear_img_v3 = cv2.cvtColor(clear_img_v3, cv2.COLOR_BGR2RGB)
                dcp_clear_img = cv2.cvtColor(dcp_clear_img, cv2.COLOR_BGR2RGB)
                gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)

                # measure PSNR
                PSNR = np.round(peak_signal_noise_ratio(gt_img, dcp_clear_img), 4)
                print("[DCP] PSNR of ", img_name, " : ", PSNR, file=f)
                average_PSNR[0] += PSNR

                PSNR = np.round(peak_signal_noise_ratio(gt_img, clear_img_v1), 4)
                print("[Ours] PSNR of ", img_name, " : ", PSNR, file=f)
                average_PSNR[1] += PSNR

                PSNR = np.round(peak_signal_noise_ratio(gt_img, clear_img_v2), 4)
                print("[Ours] PSNR of ", img_name, " : ", PSNR, file=f)
                average_PSNR[2] += PSNR

                PSNR = np.round(peak_signal_noise_ratio(gt_img, clear_img_v3), 4)
                print("[Ours] PSNR of ", img_name, " : ", PSNR, file=f)
                average_PSNR[3] += PSNR

                # measure SSIM
                SSIM = np.round(tensor_utils.measure_ssim(gt_img, dcp_clear_img), 4)
                print("[DCP] SSIM of ", img_name, " : ", SSIM, file=f)
                average_SSIM[0] += SSIM

                SSIM = np.round(tensor_utils.measure_ssim(gt_img, clear_img_v1), 4)
                print("[Ours] SSIM of ", img_name, " : ", SSIM, file=f)
                average_SSIM[1] += SSIM

                SSIM = np.round(tensor_utils.measure_ssim(gt_img, clear_img_v2), 4)
                print("[Ours] SSIM of ", img_name, " : ", SSIM, file=f)
                average_SSIM[2] += SSIM

                SSIM = np.round(tensor_utils.measure_ssim(gt_img, clear_img_v3), 4)
                print("[Ours] SSIM of ", img_name, " : ", SSIM, file=f)
                average_SSIM[3] += SSIM

                print(file=f)

                ax[0, column].imshow(hazy_img)
                ax[0, column].axis('off')
                ax[1, column].imshow(dcp_clear_img)
                ax[1, column].axis('off')
                ax[2, column].imshow(clear_img_v1)
                ax[2, column].axis('off')
                ax[3, column].imshow(clear_img_v2)
                ax[3, column].axis('off')
                ax[4, column].imshow(clear_img_v3)
                ax[4, column].axis('off')
                ax[5, column].imshow(gt_img)
                ax[5, column].axis('off')

                column = column + 1

                if (column == FIG_COLS):
                    fig_num = fig_num + 1
                    file_name = SAVE_PATH + "fig_" + str(fig_num) + "_" + MODEL_CHECKPOINT + ".jpg"
                    plt.savefig(file_name)
                    plt.show()

                    # create new figure
                    fig, ax = plt.subplots(ncols=FIG_COLS, nrows=FIG_ROWS, constrained_layout=True, sharex=True)
                    fig.set_size_inches(FIG_WIDTH, FIG_HEIGHT)
                    column = 0

        for i in range(len(average_SSIM)):
            average_SSIM[i] = average_SSIM[i] / count * 1.0
            average_PSNR[i] = average_PSNR[i] / count * 1.0

        print(file=f)
        print("[DCP] Average PSNR: ", np.round(average_PSNR[0], 5), file=f)
        print("[Ours-Scene Radiance] Average PSNR: ", np.round(average_PSNR[1], 5), file=f)
        print("[Ours-Network Estimator V1] Average PSNR: ", np.round(average_PSNR[2], 5), file=f)
        print("[Ours-Network Estimator V2] Average PSNR: ", np.round(average_PSNR[3], 5), file=f)
        print(file=f)
        print("[DCP] Average SSIM: ", np.round(average_SSIM[0], 5), file=f)
        print("[Ours-Scene Radiance] Average SSIM: ", np.round(average_SSIM[1], 5), file=f)
        print("[Ours-Network Estimator V1] Average SSIM: ", np.round(average_SSIM[2], 5), file=f)
        print("[Ours-Network Estimator V2] Average SSIM: ", np.round(average_SSIM[3], 5), file=f)
def main():
    benchmark_ohaze()
    #benchmark_ohaze_inmodels()


# FIX for broken pipe num_workers issue.
if __name__ == "__main__":
    main()