import kornia
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
from skimage.metrics import mean_squared_error
from custom_losses import ssim_loss


def atmosphere_benchmark():
    HAZY_PATH = "E:/Hazy Dataset Benchmark/O-HAZE/hazy/"
    hazy_list = glob.glob(HAZY_PATH + "*.jpg")

    atmosphere_pred = []

    for i, (hazy_path) in enumerate(hazy_list):
        hazy_img = cv2.imread(hazy_path)
        hazy_img = cv2.resize(hazy_img, (512, 512))

        A = dehazing_proper.estimate_atmosphere(hazy_img, dehazing_proper.get_dark_channel(hazy_img, 4))
        print("Estimated DCP atmosphere: ", A / 255.0)
        atmosphere_pred.append(A)

    print("Average DCP A: ", np.round(np.average(atmosphere_pred), 5) / 255.0)



def produce_dcp():
    HAZY_PATH = "E:/Hazy Dataset Benchmark/O-HAZE/hazy/"
    SAVE_PATH = "results/DCP - Results - O-Haze/"

    hazy_list = glob.glob(HAZY_PATH + "*.jpg")
    for i, (hazy_path) in enumerate(hazy_list):
        img_name = hazy_path.split("\\")[1].split(".")[0]  # save new image as PNG
        hazy_img = cv2.imread(hazy_path)
        hazy_img = cv2.resize(hazy_img, (512, 512))
        dcp_clear_img = dark_channel_prior.perform_dcp_dehaze(hazy_img, True)
        dcp_clear_img = cv2.normalize(dcp_clear_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        cv2.imwrite(SAVE_PATH + img_name + ".png", dcp_clear_img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        print("Saved DCP: ", img_name)

#simply produces results without benchmarking
def dehaze_single(T_CHECKPT_NAME, A_ESTIMATOR_NAME, INPUT_PATH, use_unlit):
    SAVE_PATH = "results/Single/"
    SAVE_TRANSMISSION_PATH = "results/Single/Transmission/"
    SAVE_ATMOSPHERE_PATH = "results/Single/Atmosphere/"

    ALBEDO_CHECKPT = "albedo_transfer_v1.04_1"
    TRANSMISSION_CHECKPT = T_CHECKPT_NAME
    AIRLIGHT_ESTIMATOR_CHECKPT = A_ESTIMATOR_NAME

    model_dehazer = dehazing_proper.ModelDehazer()
    model_dehazer.set_models_v2(ALBEDO_CHECKPT, TRANSMISSION_CHECKPT, AIRLIGHT_ESTIMATOR_CHECKPT)

    with torch.no_grad():
        print("INPUT PATH: ", INPUT_PATH)
        img_name = INPUT_PATH.split("/")[-1].split(".")[0]  # save new image as PNG
        hazy_img = cv2.imread(INPUT_PATH)
        hazy_img = cv2.resize(hazy_img, (512, 512))

        clear_img, T_tensor, A_tensor = model_dehazer.perform_dehazing_direct_v4(hazy_img, 0.0, use_unlit)
        clear_img = cv2.normalize(clear_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        cv2.imwrite(SAVE_PATH + img_name + ".png", clear_img, [cv2.IMWRITE_PNG_COMPRESSION, 9])

        torchutils.save_image(T_tensor, SAVE_TRANSMISSION_PATH + img_name + ".png")
        torchutils.save_image(A_tensor, SAVE_ATMOSPHERE_PATH + img_name + ".png")

        print("Saved: " + SAVE_PATH + img_name)

def dehaze_single_end_to_end(T_CHECKPT_NAME, INPUT_PATH):
    SAVE_PATH = "results/Single/"

    model_dehazer = dehazing_proper.ModelDehazer()

    with torch.no_grad():
        print("INPUT PATH: ", INPUT_PATH)
        img_name = INPUT_PATH.split("/")[-1].split(".")[0]  # save new image as PNG
        hazy_img = cv2.imread(INPUT_PATH)
        hazy_img = cv2.resize(hazy_img, (512, 512))

        clear_img = model_dehazer.perform_dehazing_end_to_end(hazy_img, T_CHECKPT_NAME)
        torchutils.save_image(clear_img, SAVE_PATH + img_name + ".png")
        print("Saved: " + SAVE_PATH + img_name)


def measure_performance(INPUT_PATH, GT_PATH):
    input_img = cv2.imread(INPUT_PATH)
    input_img = cv2.resize(input_img, (512, 512))
    img_name = INPUT_PATH.split("/")[-1].split(".")[0]  # save new image as PNG

    gt_img = cv2.imread(GT_PATH)
    gt_img = cv2.resize(gt_img, (512, 512))

    PSNR = np.round(peak_signal_noise_ratio(gt_img, input_img), 4)
    print("[Ours] PSNR of ", img_name, " : ", PSNR)

    SSIM = np.round(tensor_utils.measure_ssim(gt_img, input_img), 4)
    print("[Ours] SSIM of ", img_name, " : ", SSIM)


def produce_ohaze(T_CHECKPT_NAME, A_ESTIMATOR_NAME, use_unlit):
    HAZY_PATH = "E:/Hazy Dataset Benchmark/O-HAZE/hazy/"
    SAVE_PATH = "results/Ours - Results - O-Haze/"
    SAVE_TRANSMISSION_PATH = "results/Ours - Results - O-Haze/Transmission/"
    SAVE_ATMOSPHERE_PATH = "results/Ours - Results - O-Haze/Atmosphere/"

    hazy_list = glob.glob(HAZY_PATH + "*.jpg")

    ALBEDO_CHECKPT = "albedo_transfer_v1.04_1"
    TRANSMISSION_CHECKPT = T_CHECKPT_NAME
    AIRLIGHT_ESTIMATOR_CHECKPT = A_ESTIMATOR_NAME

    model_dehazer = dehazing_proper.ModelDehazer()
    model_dehazer.set_models_v2(ALBEDO_CHECKPT, TRANSMISSION_CHECKPT, AIRLIGHT_ESTIMATOR_CHECKPT)

    for i, (hazy_path) in enumerate(hazy_list):
        with torch.no_grad():
            img_name = hazy_path.split("\\")[1].split(".")[0] #save new image as PNG
            hazy_img = cv2.imread(hazy_path)
            hazy_img = cv2.resize(hazy_img, (512, 512))

            #clear_img = model_dehazer.perform_dehazing_direct_v2(hazy_img)
            #clear_img = model_dehazer.perform_dehazing_direct_v3(hazy_img, 0.8)
            clear_img, T_tensor, A_tensor = model_dehazer.perform_dehazing_direct_v4(hazy_img, 0.0, use_unlit)
            clear_img = cv2.normalize(clear_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            cv2.imwrite(SAVE_PATH + img_name + ".png", clear_img, [cv2.IMWRITE_PNG_COMPRESSION, 9])

            torchutils.save_image(T_tensor, SAVE_TRANSMISSION_PATH + img_name + ".png")
            torchutils.save_image(A_tensor, SAVE_ATMOSPHERE_PATH + img_name + ".png")

            print("Saved: " + SAVE_PATH + img_name)

def produce_ohaze_end_to_end(CHECKPT_NAME):
    HAZY_PATH = "E:/Hazy Dataset Benchmark/O-HAZE/hazy/"
    SAVE_PATH = "results/Ours - Results - O-Haze/"

    hazy_list = glob.glob(HAZY_PATH + "*.jpg")

    model_dehazer = dehazing_proper.ModelDehazer()

    for i, (hazy_path) in enumerate(hazy_list):
        with torch.no_grad():
            img_name = hazy_path.split("\\")[1].split(".")[0] #save new image as PNG
            hazy_img = cv2.imread(hazy_path)
            hazy_img = cv2.resize(hazy_img, (512, 512))

            clear_img = model_dehazer.perform_dehazing_end_to_end(hazy_img, CHECKPT_NAME)
            torchutils.save_image(clear_img, SAVE_PATH + img_name + ".png")

            print("Saved: " + SAVE_PATH + img_name)

def benchmark_ohaze(T_CHECKPT_NAME, A_GEN_NAME):
    HAZY_PATH = "E:/Hazy Dataset Benchmark/O-HAZE/hazy/"
    GT_PATH = "E:/Hazy Dataset Benchmark/O-HAZE/GT/"

    DCP_RESULTS_PATH = "results/DCP - Results - O-Haze/"
    AOD_RESULTS_PATH = "results/AODNet- Results - OHaze/"
    FFA_RESULTS_PATH = "results/FFA Net - Results - OHaze/"
    GRID_DEHAZE_RESULTS_PATH = "results/GridDehazeNet - Results - OHaze/"
    CYCLE_DEHAZE_PATH = "results/CycleDehaze - Results - OHaze/"
    EDPN_DEHAZE_PATH = "results/EDPN - Results - OHaze/"
    DA_DEHAZE_PATH = "results/DADehazing - OHaze/"
    PHYSICS_GAN_PATH = "results/PhysicsGAN - Results - OHaze/"
    SGID_PFF_PATH = "results/SGID-PFF - Results - OHaze/"
    OUR_PATH = "results/Ours - Results - O-Haze/"

    EXPERIMENT_NAME = "metrics - " +str(T_CHECKPT_NAME) + " - " +str(A_GEN_NAME)
    SAVE_PATH = "results/O-HAZE/"
    BENCHMARK_PATH = SAVE_PATH + EXPERIMENT_NAME + ".txt"

    hazy_list = glob.glob(HAZY_PATH + "*.jpg")
    gt_list = glob.glob(GT_PATH + "*.jpg")
    dcp_list = glob.glob(DCP_RESULTS_PATH + "*.png")
    aod_list = glob.glob(AOD_RESULTS_PATH + "*.jpg")
    ffa_list = glob.glob(FFA_RESULTS_PATH + "*.png")
    grid_list = glob.glob(GRID_DEHAZE_RESULTS_PATH + "*.jpg")
    cycle_dh_list = glob.glob(CYCLE_DEHAZE_PATH + "*.jpg")
    edpn_list = glob.glob(EDPN_DEHAZE_PATH + "*.png")
    da_list = glob.glob(DA_DEHAZE_PATH + "*.png")
    physicsgan_list = glob.glob(PHYSICS_GAN_PATH + "*.jpg")
    sgid_list = glob.glob(SGID_PFF_PATH + "*.jpg")
    our_list = glob.glob(OUR_PATH + "*.png")

    print(hazy_list)
    print(gt_list)
    print(our_list)

    FIG_ROWS = 12
    FIG_COLS = 4
    FIG_WIDTH = 10
    FIG_HEIGHT = 40
    fig, ax = plt.subplots(ncols=FIG_COLS, nrows=FIG_ROWS, constrained_layout=True, sharex=True)
    fig.set_size_inches(FIG_WIDTH, FIG_HEIGHT)
    column = 0
    fig_num = 0
    average_SSIM = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    average_PSNR = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    average_MSE = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    count = 0

    with open(BENCHMARK_PATH, "w") as f:
        for i, (hazy_path, gt_path, dcp_path, ffa_path, grid_path, cycle_dh_path, aod_path, edpn_path, da_path, physicsgan_path, sgid_path, our_path) in \
                enumerate(zip(hazy_list, gt_list, dcp_list, ffa_list, grid_list, cycle_dh_list, aod_list, edpn_list, da_list, physicsgan_list, sgid_list, our_list)):
            with torch.no_grad():
                count = count + 1
                img_name = hazy_path.split("\\")[1]

                im_size = (512, 512)
                hazy_img = tensor_utils.load_metrics_compatible_img(hazy_path, im_size)
                aod_img = tensor_utils.load_metrics_compatible_img(aod_path, im_size)
                edpn_img = tensor_utils.load_metrics_compatible_img(edpn_path, im_size)
                ffa_img = tensor_utils.load_metrics_compatible_img(ffa_path, im_size)
                grid_img = tensor_utils.load_metrics_compatible_img(grid_path, im_size)
                cycle_dehaze_img = tensor_utils.load_metrics_compatible_img(cycle_dh_path, im_size)
                da_img = tensor_utils.load_metrics_compatible_img(da_path, im_size)
                dcp_clear_img = tensor_utils.load_metrics_compatible_img(dcp_path, im_size)
                physicsgan_img = tensor_utils.load_metrics_compatible_img(physicsgan_path, im_size)
                sgid_img = tensor_utils.load_metrics_compatible_img(sgid_path, im_size)
                clear_img = tensor_utils.load_metrics_compatible_img(our_path, im_size)
                gt_img = tensor_utils.load_metrics_compatible_img(gt_path, im_size)

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

                PSNR = np.round(peak_signal_noise_ratio(gt_img, da_img), 4)
                print("[DA-Dehaze] PSNR of ", img_name, " : ", PSNR, file=f)
                average_PSNR[6] += PSNR

                PSNR = np.round(peak_signal_noise_ratio(gt_img, physicsgan_img), 4)
                print("[PhysicsGAN] PSNR of ", img_name, " : ", PSNR, file=f)
                average_PSNR[7] += PSNR

                PSNR = np.round(peak_signal_noise_ratio(gt_img, sgid_img), 4)
                print("[SGID-PFF] PSNR of ", img_name, " : ", PSNR, file=f)
                average_PSNR[8] += PSNR

                PSNR = np.round(peak_signal_noise_ratio(gt_img, clear_img), 4)
                print("[Ours] PSNR of ", img_name, " : ", PSNR, file=f)
                average_PSNR[9] += PSNR

                # measure MSE
                MSE = np.round(mean_squared_error(gt_img, dcp_clear_img), 4)
                print("[DCP] MSE of ", img_name, " : ", MSE, file=f)
                average_MSE[0] += MSE

                MSE = np.round(mean_squared_error(gt_img, aod_img), 4)
                print("[AOD-Net] MSE of ", img_name, " : ", MSE, file=f)
                average_MSE[1] += MSE

                MSE = np.round(mean_squared_error(gt_img, cycle_dehaze_img), 4)
                print("[CycleDehaze] MSE of ", img_name, " : ", MSE, file=f)
                average_MSE[2] += MSE

                MSE = np.round(mean_squared_error(gt_img, ffa_img), 4)
                print("[FFA-Net] MSE of ", img_name, " : ", MSE, file=f)
                average_MSE[3] += MSE

                MSE = np.round(mean_squared_error(gt_img, grid_img), 4)
                print("[GridDehazeNet] MSE of ", img_name, " : ", MSE, file=f)
                average_MSE[4] += MSE

                MSE = np.round(mean_squared_error(gt_img, edpn_img), 4)
                print("[EDPN] MSE of ", img_name, " : ", MSE, file=f)
                average_MSE[5] += MSE

                MSE = np.round(mean_squared_error(gt_img, da_img), 4)
                print("[DA-Dehaze] MSE of ", img_name, " : ", MSE, file=f)
                average_MSE[6] += MSE

                MSE = np.round(mean_squared_error(gt_img, physicsgan_img), 4)
                print("[PhysicsGAN] MSE of ", img_name, " : ", MSE, file=f)
                average_MSE[7] += MSE

                MSE = np.round(mean_squared_error(gt_img, sgid_img), 4)
                print("[SGID-PFF] MSE of ", img_name, " : ", MSE, file=f)
                average_MSE[8] += MSE

                MSE = np.round(mean_squared_error(gt_img, clear_img), 4)
                print("[Ours] MSE of ", img_name, " : ", MSE, file=f)
                print("[Ours] MSE of ", img_name, " : ", MSE)
                average_MSE[9] += MSE

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

                SSIM = np.round(tensor_utils.measure_ssim(gt_img, da_img), 4)
                print("[DA-Dehaze] SSIM of ", img_name, " : ", SSIM, file=f)
                average_SSIM[6] += SSIM

                SSIM = np.round(tensor_utils.measure_ssim(gt_img, physicsgan_img), 4)
                print("[PhysicsGAN] SSIM of ", img_name, " : ", SSIM, file=f)
                average_SSIM[7] += SSIM

                SSIM = np.round(tensor_utils.measure_ssim(gt_img, sgid_img), 4)
                print("[SGID-PFF] SSIM of ", img_name, " : ", SSIM, file=f)
                average_SSIM[8] += SSIM

                SSIM = np.round(tensor_utils.measure_ssim(gt_img, clear_img), 4)
                print("[Ours] SSIM of ", img_name, " : ", SSIM, file=f)
                print("[Ours] SSIM of ", img_name, " : ", SSIM)
                average_SSIM[9] += SSIM

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
                ax[7, column].imshow(da_img)
                ax[7, column].axis('off')
                ax[8, column].imshow(clear_img)
                ax[8, column].axis('off')
                ax[9, column].imshow(physicsgan_img)
                ax[9, column].axis('off')
                ax[10, column].imshow(sgid_img)
                ax[10, column].axis('off')
                ax[11, column].imshow(gt_img)
                ax[11, column].axis('off')

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
            average_MSE[i] = average_MSE[i] / count * 1.0

        print(file=f)
        print("[DCP] Average PSNR: ", np.round(average_PSNR[0], 5), file=f)
        print("[AOD-Net] Average PSNR: ", np.round(average_PSNR[1], 5), file=f)
        print("[CycleDehaze] Average PSNR: ", np.round(average_PSNR[2], 5), file=f)
        print("[FFA-Net] Average PSNR: ", np.round(average_PSNR[3], 5), file=f)
        print("[GridDehazeNet] Average PSNR: ", np.round(average_PSNR[4], 5), file=f)
        print("[EDPN] Average PSNR: ", np.round(average_PSNR[5], 5), file=f)
        print("[DA-Dehaze] Average PSNR: ", np.round(average_PSNR[6], 5), file=f)
        print("[PhysicsGAN] Average PSNR: ", np.round(average_PSNR[7], 5), file=f)
        print("[SGID-PFF] Average PSNR: ", np.round(average_PSNR[8], 5), file=f)
        print("[Ours] Average PSNR: ", np.round(average_PSNR[9], 5), file=f)
        print(file=f)
        print("[DCP] Average SSIM: ", np.round(average_SSIM[0], 5), file=f)
        print("[AOD-Net] Average SSIM: ", np.round(average_SSIM[1], 5), file=f)
        print("[CycleDehaze] Average SSIM: ", np.round(average_SSIM[2], 5), file=f)
        print("[FFA-Net] Average SSIM: ", np.round(average_SSIM[3], 5), file=f)
        print("[GridDehazeNet] Average SSIM: ", np.round(average_SSIM[4], 5), file=f)
        print("[EDPN] Average SSIM: ", np.round(average_SSIM[5], 5), file=f)
        print("[DA-Dehaze] Average SSIM: ", np.round(average_SSIM[6], 5), file=f)
        print("[PhysicsGAN] Average SSIM: ", np.round(average_SSIM[7], 5), file=f)
        print("[SGID-PFF] Average SSIM: ", np.round(average_SSIM[8], 5), file=f)
        print("[Ours] Average SSIM: ", np.round(average_SSIM[9], 5), file=f)
        print(file=f)
        print("[DCP] Average MSE: ", np.round(average_MSE[0], 5), file=f)
        print("[AOD-Net] Average MSE: ", np.round(average_MSE[1], 5), file=f)
        print("[CycleDehaze] Average MSE: ", np.round(average_MSE[2], 5), file=f)
        print("[FFA-Net] Average MSE: ", np.round(average_MSE[3], 5), file=f)
        print("[GridDehazeNet] Average MSE: ", np.round(average_MSE[4], 5), file=f)
        print("[EDPN] Average MSE: ", np.round(average_MSE[5], 5), file=f)
        print("[DA-Dehaze] Average MSE: ", np.round(average_MSE[6], 5), file=f)
        print("[PhysicsGAN] Average MSE: ", np.round(average_MSE[7], 5), file=f)
        print("[SGID-PFF] Average MSE: ", np.round(average_MSE[8], 5), file=f)
        print("[Ours] Average MSE: ", np.round(average_MSE[9], 5), file=f)

def output_best_worst(T_CHECKPT_NAME, A_CHECKPT_NAME, best_threshold, worst_threshold):
    HAZY_PATH = "E:/Hazy Dataset Benchmark/O-HAZE/hazy/"
    GT_PATH = "E:/Hazy Dataset Benchmark/O-HAZE/GT/"
    AOD_RESULTS_PATH = "results/AODNet- Results - OHaze/"
    FFA_RESULTS_PATH = "results/FFA Net - Results - OHaze/"
    GRID_DEHAZE_RESULTS_PATH = "results/GridDehazeNet - Results - OHaze/"
    CYCLE_DEHAZE_PATH = "results/CycleDehaze - Results - OHaze/"
    EDPN_DEHAZE_PATH = "results/EDPN - Results - OHaze/"
    OUR_PATH = "results/Ours - Results - O-Haze/"

    EXPERIMENT_NAME = "Images - " + str(T_CHECKPT_NAME) + " - " + str(A_CHECKPT_NAME)
    SAVE_PATH = "results/O-HAZE - Print/"
    BENCHMARK_PATH = SAVE_PATH + EXPERIMENT_NAME + ".txt"

    hazy_list = glob.glob(HAZY_PATH + "*.jpg")
    gt_list = glob.glob(GT_PATH + "*.jpg")
    aod_list = glob.glob(AOD_RESULTS_PATH + "*.jpg")
    ffa_list = glob.glob(FFA_RESULTS_PATH + "*.png")
    grid_list = glob.glob(GRID_DEHAZE_RESULTS_PATH + "*.jpg")
    cycle_dh_list = glob.glob(CYCLE_DEHAZE_PATH + "*.jpg")
    edpn_list = glob.glob(EDPN_DEHAZE_PATH + "*.png")
    our_list = glob.glob(OUR_PATH + "*.png")

    FIG_ROWS = 9
    FIG_COLS = 4
    FIG_WIDTH = 10
    FIG_HEIGHT = 30
    fig, ax = plt.subplots(ncols=FIG_COLS, nrows=FIG_ROWS, constrained_layout=True, sharex=True)
    fig.set_size_inches(FIG_WIDTH, FIG_HEIGHT)
    column = 0
    fig_num = 0

    with open(BENCHMARK_PATH, "w") as f:
        for i, (hazy_path, gt_path, ffa_path, grid_path, cycle_dh_path, aod_path, edpn_path, our_path) in \
                enumerate(zip(hazy_list, gt_list, ffa_list, grid_list, cycle_dh_list, aod_list, edpn_list, our_list)):
            with torch.no_grad():
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
                aod_img = cv2.normalize(aod_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
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

                SSIM = np.round(tensor_utils.measure_ssim(gt_img, clear_img), 4)
                if(SSIM > best_threshold):
                    print("[BEST] SSIM of ", img_name, " : ", SSIM, file=f)
                    print("[BEST] SSIM of ", img_name, " : ", SSIM)

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
                        file_name = SAVE_PATH + "best_" + str(fig_num) + "_" + EXPERIMENT_NAME + ".jpg"
                        plt.savefig(file_name)
                        plt.show()

                        # create new figure
                        fig, ax = plt.subplots(ncols=FIG_COLS, nrows=FIG_ROWS, constrained_layout=True, sharex=True)
                        fig.set_size_inches(FIG_WIDTH, FIG_HEIGHT)
                        column = 0

                if(SSIM < worst_threshold):
                    print("[WORST] SSIM of ", img_name, " : ", SSIM, file=f)
                    print("[WORST] SSIM of ", img_name, " : ", SSIM)

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
                        file_name = SAVE_PATH + "worst_" + str(fig_num) + "_" + EXPERIMENT_NAME + ".jpg"
                        plt.savefig(file_name)
                        plt.show()

                        # create new figure
                        fig, ax = plt.subplots(ncols=FIG_COLS, nrows=FIG_ROWS, constrained_layout=True, sharex=True)
                        fig.set_size_inches(FIG_WIDTH, FIG_HEIGHT)
                        column = 0


def main():
    # produce_ohaze("transmission_albedo_estimator_v1.15_6", "airlight_estimator_v1.15_6", True)
    # benchmark_ohaze("transmission_albedo_estimator_v1.15_6", "airlight_estimator_v1.15_6")

    # produce_ohaze("transmission_albedo_estimator_v1.15_7", "airlight_estimator_v1.15_7", True)
    # benchmark_ohaze("transmission_albedo_estimator_v1.15_7", "airlight_estimator_v1.15_7")
    #
    # produce_ohaze("transmission_albedo_estimator_v1.15_8", "airlight_estimator_v1.15_8", True)
    # benchmark_ohaze("transmission_albedo_estimator_v1.15_8", "airlight_estimator_v1.15_8")
    #
    # produce_ohaze("transmission_albedo_estimator_v1.15_9", "airlight_estimator_v1.15_9", True)
    # benchmark_ohaze("transmission_albedo_estimator_v1.15_9", "airlight_estimator_v1.15_9")
    #
    # produce_ohaze("transmission_albedo_estimator_v1.15_10", "airlight_estimator_v1.15_10", True)
    # benchmark_ohaze("transmission_albedo_estimator_v1.15_10", "airlight_estimator_v1.15_10")

    # produce_ohaze("transmission_albedo_estimator_v1.16_6", "airlight_estimator_v1.16_6", True)
    benchmark_ohaze("transmission_albedo_estimator_v1.16_6", "airlight_estimator_v1.16_6")

    # produce_ohaze_end_to_end("end_to_end_dehazer_v1.00_1")
    # benchmark_ohaze("end_to_end_dehazer_v1.00_1", "")
    # output_best_worst("end_to_end_dehazer_v1.00_1", "", 0.88, 0.77)

    # dehaze_single_end_to_end("end_to_end_dehazer_v1.00_1", "E:/Hazy Dataset Benchmark/OTS_BETA/haze/0899_0.95_0.2.jpg")

    # dehaze_single("transmission_albedo_estimator_v1.15_6", "airlight_estimator_v1.15_6",
    #               "E:/Hazy Dataset Benchmark/OTS_BETA/haze/0352_0.95_0.2.jpg", True)
    # dehaze_single("transmission_albedo_estimator_v1.15_6", "airlight_estimator_v1.15_6",
    #               "E:/Hazy Dataset Benchmark/OTS_BETA/haze/0899_0.95_0.2.jpg", True)
    # dehaze_single("transmission_albedo_estimator_v1.15_6", "airlight_estimator_v1.15_6",
    #               "E:/Hazy Dataset Benchmark/OTS_BETA/haze/0920_0.95_0.2.jpg", True)

    # measure_performance("./results/Single/li_hazy_1.png", "E:/Hazy Dataset Benchmark/Standard/li_clear_1.png")
    # measure_performance("E:/Hazy Dataset Benchmark/Standard/li_produced_1.png", "E:/Hazy Dataset Benchmark/Standard/li_clear_1.png")

    # CHECKPT_NAME = "dehazer_v2.09_4"
    # produce_ohaze(CHECKPT_NAME, CHECKPT_NAME)
    # benchmark_ohaze(CHECKPT_NAME, CHECKPT_NAME)
    # output_best_worst(CHECKPT_NAME, CHECKPT_NAME, 0.88, 0.77)

    #
    # CHECKPT_NAME = "dehazer_v2.07_4"
    # produce_ohaze(CHECKPT_NAME, CHECKPT_NAME)
    # benchmark_ohaze(CHECKPT_NAME, CHECKPT_NAME)
    #
    # CHECKPT_NAME = "dehazer_v2.07_5"
    # produce_ohaze(CHECKPT_NAME, CHECKPT_NAME)
    # benchmark_ohaze(CHECKPT_NAME, CHECKPT_NAME)

    #CHECKPT_NAME = "dehazer_v2.07_3"
    #produce_ohaze(CHECKPT_NAME, CHECKPT_NAME)
    #benchmark_ohaze(CHECKPT_NAME, CHECKPT_NAME)
    #output_best_worst(CHECKPT_NAME, CHECKPT_NAME, 0.86, 0.77)

    # CHECKPT_NAME = "dehazer_v2.06_3"
    # produce_ohaze(CHECKPT_NAME, "airlight_gen_v1.06_3")
    # benchmark_ohaze(CHECKPT_NAME, "airlight_gen_v1.06_3")


# FIX for broken pipe num_workers issue.
if __name__ == "__main__":
    main()
