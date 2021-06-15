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


def produce_ots():
    HAZY_PATH = "E:Hazy Dataset Benchmark/OTS_BETA/haze/"
    SAVE_PATH = "results/Ours - Results - OTS-Beta/"
    SAVE_TRANSMISSION_PATH = "results/Ours - Results - OTS-Beta/Transmission/"
    SAVE_ATMOSPHERE_PATH = "results/Ours - Results - OTS-Beta/Atmosphere/"

    hazy_list = glob.glob(HAZY_PATH + "*0.95_0.2.jpg") #specify atmosphere intensity

    TRANSMISSION_CHECKPT = "transmission_albedo_estimator_v1.07_3"
    AIRLIGHT_CHECKPT = "airlight_estimator_v1.05_1"

    model_dehazer = dehazing_proper.ModelDehazer()
    model_dehazer.set_models(TRANSMISSION_CHECKPT, AIRLIGHT_CHECKPT, dehazing_proper.AtmosphereMethod.NETWORK_ESTIMATOR_V1)

    for i, (hazy_path) in enumerate(hazy_list):
        with torch.no_grad():
            img_name = hazy_path.split("\\")[1]
            hazy_img = cv2.imread(hazy_path)
            hazy_img = cv2.resize(hazy_img, (512, 512))

            clear_img = model_dehazer.perform_dehazing(hazy_img, 0.8, 0.3)
            clear_img = cv2.normalize(clear_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            cv2.imwrite(SAVE_PATH + img_name, clear_img)

            T_tensor, A_tensor = model_dehazer.derive_T_and_A(hazy_img)
            torchutils.save_image(T_tensor, SAVE_TRANSMISSION_PATH + img_name + ".png")
            torchutils.save_image(A_tensor, SAVE_ATMOSPHERE_PATH + img_name + ".png")

            print("Saved: " + SAVE_PATH + img_name)

def benchmark_ots():
    HAZY_PATH = "E:/Hazy Dataset Benchmark/OTS_BETA/haze/"
    GT_PATH = "E:/Hazy Dataset Benchmark/OTS_BETA/clear/"

    AOD_RESULTS_PATH = "results/AODNet- Results - OTS-Beta/"
    FFA_RESULTS_PATH = "results/FFA Net - Results - OTS-Beta/"
    GRID_DEHAZE_RESULTS_PATH = "results/GridDehazeNet - Results - OTS-Beta/"
    CYCLE_DEHAZE_PATH = "results/CycleDehaze - Results - OTS-Beta/"
    EDPN_DEHAZE_PATH = "results/EDPN - Results - OTS-Beta/"
    OUR_PATH = "results/Ours - Results - OTS-Beta/"

    EXPERIMENT_NAME = "metrics - 1"
    SAVE_PATH = "results/RESIDE-OTS Beta/"
    BENCHMARK_PATH = SAVE_PATH + EXPERIMENT_NAME + ".txt"

    hazy_list = glob.glob(HAZY_PATH + "*0.95_0.2.jpg") #specify atmosphere intensity
    aod_list = glob.glob(AOD_RESULTS_PATH + "*.jpg")
    ffa_list = glob.glob(FFA_RESULTS_PATH + "*.jpg")
    grid_list = glob.glob(GRID_DEHAZE_RESULTS_PATH + "*.jpg")
    cycle_dh_list = glob.glob(CYCLE_DEHAZE_PATH + "*.jpg")
    edpn_list = glob.glob(EDPN_DEHAZE_PATH + "*.png")
    our_list = glob.glob(OUR_PATH + "*.jpg")

    print(len(hazy_list))
    print(len(aod_list))
    print(len(ffa_list))
    print(len(grid_list))
    print(len(cycle_dh_list))
    print(len(edpn_list))
    print(len(our_list))

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
        for i, (hazy_path, ffa_path, grid_path, cycle_dh_path, aod_path, edpn_path, our_path) in \
                enumerate(zip(hazy_list, ffa_list, grid_list, cycle_dh_list, aod_list, edpn_list, our_list)):
            with torch.no_grad():
                count = count + 1
                img_name = hazy_path.split("\\")[1]
                gt_prefix = img_name.split("_")[0]

                hazy_img = cv2.imread(hazy_path)
                hazy_img = cv2.resize(hazy_img, (512, 512))

                gt_path = GT_PATH + gt_prefix + ".jpg"
                gt_img = cv2.imread(gt_path)
                gt_img = cv2.resize(gt_img, (int(np.shape(hazy_img)[1]), int(np.shape(hazy_img)[0])))

                aod_img = cv2.imread(aod_path)
                aod_img = cv2.resize(aod_img, (int(np.shape(hazy_img)[1]), int(np.shape(hazy_img)[0])))

                edpn_img = cv2.imread(edpn_path)
                edpn_img = cv2.resize(edpn_img, (int(np.shape(gt_img)[1]), int(np.shape(gt_img)[0])))

                ffa_img = cv2.imread(ffa_path)
                ffa_img = cv2.resize(ffa_img, (int(np.shape(hazy_img)[1]), int(np.shape(hazy_img)[0])))

                grid_img = cv2.imread(grid_path)
                grid_img = cv2.resize(grid_img, (int(np.shape(hazy_img)[1]), int(np.shape(hazy_img)[0])))

                cycle_dehaze_img = cv2.imread(cycle_dh_path)
                cycle_dehaze_img = cv2.resize(cycle_dehaze_img, (int(np.shape(gt_img)[1]), int(np.shape(gt_img)[0])))

                dcp_clear_img = dark_channel_prior.perform_dcp_dehaze(hazy_img, True)

                clear_img = cv2.imread(our_path)
                clear_img = cv2.resize(clear_img, (int(np.shape(gt_img)[1]), int(np.shape(gt_img)[0])))

                # normalize images
                hazy_img = cv2.normalize(hazy_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                clear_img = cv2.normalize(clear_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                dcp_clear_img = cv2.normalize(dcp_clear_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                ffa_img = cv2.normalize(ffa_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                grid_img = cv2.normalize(grid_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                cycle_dehaze_img = cv2.normalize(cycle_dehaze_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                aod_img = cv2.normalize(aod_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8U)
                edpn_img = cv2.normalize(edpn_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                gt_img = cv2.normalize(gt_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

                # make images compatible with matplotlib
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
                print("[Ours] SSIM of ", img_name, " : ", SSIM, file=f)
                print("[Ours] SSIM of ", img_name, " : ", SSIM)
                average_SSIM[6] += SSIM

                print(file=f)

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
                    #plt.show()

                    # create new figure
                    fig, ax = plt.subplots(ncols=FIG_COLS, nrows=FIG_ROWS, constrained_layout=True, sharex=True)
                    fig.set_size_inches(FIG_WIDTH, FIG_HEIGHT)
                    column = 0

        for i in range(len(average_SSIM)):
            average_SSIM[i] = average_SSIM[i] / count * 1.0
            average_PSNR[i] = average_PSNR[i] / count * 1.0

        print(file=f)
        print("[DCP] Average PSNR: ", np.round(average_PSNR[0], 5), file=f)
        print("[AOD-Net] Average PSNR: ", np.round(average_PSNR[1], 5), file=f)
        print("[CycleDehaze] Average PSNR: ", np.round(average_PSNR[2], 5), file=f)
        print("[FFA-Net] Average PSNR: ", np.round(average_PSNR[3], 5), file=f)
        print("[GridDehazeNet] Average PSNR: ", np.round(average_PSNR[4], 5), file=f)
        print("[EDPN] Average PSNR: ", np.round(average_PSNR[5], 5), file=f)
        print("[Ours] Average PSNR: ", np.round(average_PSNR[6], 5), file=f)
        print(file=f)
        print("[DCP] Average SSIM: ", np.round(average_SSIM[0], 5), file=f)
        print("[AOD-Net] Average SSIM: ", np.round(average_SSIM[1], 5), file=f)
        print("[CycleDehaze] Average SSIM: ", np.round(average_SSIM[2], 5), file=f)
        print("[FFA-Net] Average SSIM: ", np.round(average_SSIM[3], 5), file=f)
        print("[GridDehazeNet] Average SSIM: ", np.round(average_SSIM[4], 5), file=f)
        print("[EDPN] Average SSIM: ", np.round(average_SSIM[5], 5), file=f)
        print("[Ours] Average SSIM: ", np.round(average_SSIM[6], 5), file=f)


def main():
    produce_ots()
    benchmark_ots()

# FIX for broken pipe num_workers issue.
if __name__ == "__main__":
    main()