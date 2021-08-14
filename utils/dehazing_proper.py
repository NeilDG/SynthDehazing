import kornia

import constants
import torch
import numpy as np
from enum import Enum
from torchvision import transforms
import cv2
from model import dehaze_discriminator as dh
from model import vanilla_cycle_gan as cg
from model import unet_gan as un
from model import ffa_net as ffa_gan
import math
from utils import pytorch_colors
from itertools import combinations_with_replacement
from collections import defaultdict
from torch.nn.functional import interpolate

R, G, B = 0, 1, 2  # index for convenience
def boxfilter(I, r):
    """Fast box filter implementation.
    Parameters
    ----------
    I:  a single channel/gray image data normalized to [0.0, 1.0]
    r:  window radius
    Return
    -----------
    The filtered image data.
    """
    M, N = I.shape
    dest = np.zeros((M, N))

    # cumulative sum over Y axis
    sumY = np.cumsum(I, axis=0)
    # difference over Y axis
    dest[:r + 1] = sumY[r: 2 * r + 1]
    dest[r + 1:M - r] = sumY[2 * r + 1:] - sumY[:M - 2 * r - 1]
    dest[-r:] = np.tile(sumY[-1], (r, 1)) - sumY[M - 2 * r - 1:M - r - 1]

    # cumulative sum over X axis
    sumX = np.cumsum(dest, axis=1)
    # difference over Y axis
    dest[:, :r + 1] = sumX[:, r:2 * r + 1]
    dest[:, r + 1:N - r] = sumX[:, 2 * r + 1:] - sumX[:, :N - 2 * r - 1]
    dest[:, -r:] = np.tile(sumX[:, -1][:, None], (1, r)) - \
        sumX[:, N - 2 * r - 1:N - r - 1]

    return dest

def guided_filter(I, p, r=40, eps=1e-3):
    """Refine a filter under the guidance of another (RGB) image.
    Parameters
    -----------
    I:   an M * N * 3 RGB image for guidance.
    p:   the M * N filter to be guided
    r:   the radius of the guidance
    eps: epsilon for the guided filter
    Return
    -----------
    The guided filter.
    """
    M, N = p.shape
    base = boxfilter(np.ones((M, N)), r)

    # each channel of I filtered with the mean filter
    means = [boxfilter(I[:, :, i], r) / base for i in range(3)]
    # p filtered with the mean filter
    mean_p = boxfilter(p, r) / base
    # filter I with p then filter it with the mean filter
    means_IP = [boxfilter(I[:, :, i] * p, r) / base for i in range(3)]
    # covariance of (I, p) in each local patch
    covIP = [means_IP[i] - means[i] * mean_p for i in range(3)]

    # variance of I in each local patch: the matrix Sigma in ECCV10 eq.14
    var = defaultdict(dict)
    for i, j in combinations_with_replacement(range(3), 2):
        var[i][j] = boxfilter(
            I[:, :, i] * I[:, :, j], r) / base - means[i] * means[j]

    a = np.zeros((M, N, 3))
    for y, x in np.ndindex(M, N):
        #         rr, rg, rb
        # Sigma = rg, gg, gb
        #         rb, gb, bb
        Sigma = np.array([[var[R][R][y, x], var[R][G][y, x], var[R][B][y, x]],
                          [var[R][G][y, x], var[G][G][y, x], var[G][B][y, x]],
                          [var[R][B][y, x], var[G][B][y, x], var[B][B][y, x]]])
        cov = np.array([c[y, x] for c in covIP])
        a[y, x] = np.dot(cov, np.linalg.inv(Sigma + eps * np.eye(3)))  # eq 14

    # ECCV10 eq.15
    b = mean_p - a[:, :, R] * means[R] - \
        a[:, :, G] * means[G] - a[:, :, B] * means[B]

    # ECCV10 eq.16
    q = (boxfilter(a[:, :, R], r) * I[:, :, R] + boxfilter(a[:, :, G], r) *
         I[:, :, G] + boxfilter(a[:, :, B], r) * I[:, :, B] + boxfilter(b, r)) / base

    return q

def generate_transmission(depth_map, beta, is_exponential_squared=False):
    if is_exponential_squared:
        return np.exp(-np.power(beta * depth_map, 2))
    else:
        return np.exp(-beta * depth_map).astype(float)
        #return np.power(np.e, -beta * depth_map)

def estimate_transmission(im, A, dark_channel):
    # im = im.cpu().numpy()

    omega = 0.95;
    im3 = np.empty(im.shape, im.dtype);

    for ind in range(0, 3):
        im3[:, :, ind] = im[:, :, ind] / A[ind]

    transmission = 1 - omega * dark_channel
    return transmission

def remove_haze(rgb_tensor, dark_channel_old, dark_channel_new):
    yuv_tensor = pytorch_colors.rgb_to_yuv(rgb_tensor)

    yuv_tensor = yuv_tensor.transpose(0, 1)
    dark_channel_old = dark_channel_old.transpose(0, 1)
    dark_channel_new = dark_channel_new.transpose(0, 1)

    (y, u, v) = torch.chunk(yuv_tensor, 3)

    # remove dark channel from y
    y = y - dark_channel_old

    print("Shape of YUV tensor: ", np.shape(yuv_tensor))

    # replace with atmosphere and transmission from new dark channel
    atmosphere = estimate_atmosphere(yuv_tensor[:, 0, :, :], dark_channel_new[:, 0, :, :])
    transmission = estimate_transmission(yuv_tensor[:, 0, :, :], atmosphere, dark_channel_new[:, 0, :, :]).to('cuda:0')

    y = y * transmission

    yuv_tensor = torch.cat((y, u, v))
    rgb_tensor = pytorch_colors.yuv_to_rgb(yuv_tensor.transpose(0, 1))
    return rgb_tensor


def get_dark_channel(I, w=1):
    b, g, r = cv2.split(I)
    dc = cv2.min(cv2.min(r, g), b);
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w, w))
    dark = cv2.erode(dc, kernel)
    return dark


def get_dark_channel_and_mask(r, g, b):
    min_1 = cv2.min(r, g)

    mask_r = cv2.bitwise_and(min_1, r)
    mask_g = cv2.bitwise_and(min_1, g)

    min_2 = cv2.min(min_1, b)
    mask_b = cv2.bitwise_and(min_2, b)

    _, mask_r = cv2.threshold(mask_r, 1, 1, cv2.THRESH_BINARY)
    _, mask_g = cv2.threshold(mask_g, 1, 1, cv2.THRESH_BINARY)
    _, mask_b = cv2.threshold(mask_b, 1, 1, cv2.THRESH_BINARY)

    # plt.imshow(mask_r, cmap = 'gray')
    # plt.show()

    # plt.imshow(mask_g, cmap = 'gray')
    # plt.show()

    # plt.imshow(mask_b, cmap = 'gray')
    # plt.show()

    dc = cv2.min(cv2.min(r, g), b);

    # plt.imshow(dc, cmap = 'gray')
    # plt.show()

    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(w,w))
    # dark = cv2.erode(dc,kernel)
    return dc, mask_r, mask_g, mask_b

def estimate_atmosphere(im, dark, h, w):
    # im = im.cpu().numpy()
    #[h, w] = [np.shape(im)[0], np.shape(im)[1]]

    imsz = h * w
    #numpx = int(max(math.floor(imsz / 1000), 1))
    numpx = int(max(math.floor(imsz / 10), 1))
    darkvec = dark.reshape(imsz, 1);
    imvec = im.reshape(imsz, 3)

    indices = darkvec.argsort();
    indices = indices[imsz - numpx::]

    atmsum = np.zeros([1, 3])
    for ind in range(1, numpx):
        atmsum = atmsum + imvec[indices[ind]]
        #print(atmsum)

    A = atmsum / numpx;

    return A
    # return np.max(A)

class AtmosphereMethod(Enum):
    SCENE_RADIANCE = 0
    DIRECT = 1
    NETWORK_ESTIMATOR_V1 = 2,
    NETWORK_ESTIMATOR_V2 = 3

class ModelDehazer():

    def __init__(self):
        self.gpu_device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        self.albedo_models = {}
        self.transmission_models = {}
        self.atmosphere_models = {}

        #add models
        checkpt = torch.load("checkpoint/albedo_transfer_v1.04_1.pt")
        self.albedo_models["albedo_transfer_v1.04_1"] = ffa_gan.FFA(gps=3, blocks=18).to(self.gpu_device)
        self.albedo_models["albedo_transfer_v1.04_1"].load_state_dict(checkpt[constants.GENERATOR_KEY + "A"])

        checkpt = torch.load("checkpoint/transmission_albedo_estimator_v1.06_4.pt")
        self.transmission_models["transmission_albedo_estimator_v1.06_4"] = cg.GeneratorNoDropOut(input_nc=3, output_nc=1, n_residual_blocks=8).to(self.gpu_device)
        self.transmission_models["transmission_albedo_estimator_v1.06_4"].load_state_dict(checkpt[constants.GENERATOR_KEY + "A"])

        # checkpt = torch.load("checkpoint/dehazer_v2.07_1.pt")
        # self.transmission_models["dehazer_v2.07_1"] = cg.Generator(input_nc=3, output_nc=1, n_residual_blocks= 8).to(self.gpu_device)
        # self.transmission_models["dehazer_v2.07_1"].load_state_dict(checkpt[constants.GENERATOR_KEY + "T"])
        #
        # checkpt = torch.load("checkpoint/dehazer_v2.07_1.pt")
        # self.atmosphere_models["dehazer_v2.07_1"] = cg.Generator(input_nc=6, output_nc=3, n_residual_blocks= 8).to(self.gpu_device)
        # self.atmosphere_models["dehazer_v2.07_1"].load_state_dict(checkpt[constants.GENERATOR_KEY + "A"])

        checkpt = torch.load("checkpoint/dehazer_v2.07_2.pt")
        self.transmission_models["dehazer_v2.07_2"] = cg.Generator(input_nc=3, output_nc=1, n_residual_blocks=10).to(self.gpu_device)
        self.transmission_models["dehazer_v2.07_2"].load_state_dict(checkpt[constants.GENERATOR_KEY + "T"])

        checkpt = torch.load("checkpoint/dehazer_v2.07_2.pt")
        self.atmosphere_models["dehazer_v2.07_2"] = cg.Generator(input_nc=6, output_nc=3, n_residual_blocks=10).to(self.gpu_device)
        self.atmosphere_models["dehazer_v2.07_2"].load_state_dict(checkpt[constants.GENERATOR_KEY + "A"])

        checkpt = torch.load("checkpoint/dehazer_v2.07_3.pt")
        self.transmission_models["dehazer_v2.07_3"] = cg.Generator(input_nc=3, output_nc=1, n_residual_blocks=10).to(self.gpu_device)
        self.transmission_models["dehazer_v2.07_3"].load_state_dict(checkpt[constants.GENERATOR_KEY + "T"])

        checkpt = torch.load("checkpoint/dehazer_v2.07_3.pt")
        self.atmosphere_models["dehazer_v2.07_3"] = cg.Generator(input_nc=6, output_nc=3, n_residual_blocks=10).to(self.gpu_device)
        self.atmosphere_models["dehazer_v2.07_3"].load_state_dict(checkpt[constants.GENERATOR_KEY + "A"])

        checkpt = torch.load("checkpoint/dehazer_v2.07_4.pt")
        self.transmission_models["dehazer_v2.07_4"] = cg.Generator(input_nc=3, output_nc=1, n_residual_blocks=10).to(self.gpu_device)
        self.transmission_models["dehazer_v2.07_4"].load_state_dict(checkpt[constants.GENERATOR_KEY + "T"])

        checkpt = torch.load("checkpoint/dehazer_v2.07_4.pt")
        self.atmosphere_models["dehazer_v2.07_4"] = cg.Generator(input_nc=6, output_nc=3, n_residual_blocks=10).to(self.gpu_device)
        self.atmosphere_models["dehazer_v2.07_4"].load_state_dict(checkpt[constants.GENERATOR_KEY + "A"])

        checkpt = torch.load("checkpoint/dehazer_v2.07_5.pt")
        self.transmission_models["dehazer_v2.07_5"] = cg.Generator(input_nc=3, output_nc=1, n_residual_blocks=10).to(self.gpu_device)
        self.transmission_models["dehazer_v2.07_5"].load_state_dict(checkpt[constants.GENERATOR_KEY + "T"])

        checkpt = torch.load("checkpoint/dehazer_v2.07_5.pt")
        self.atmosphere_models["dehazer_v2.07_5"] = cg.Generator(input_nc=6, output_nc=3, n_residual_blocks=10).to(self.gpu_device)
        self.atmosphere_models["dehazer_v2.07_5"].load_state_dict(checkpt[constants.GENERATOR_KEY + "A"])

        checkpt = torch.load("checkpoint/dehazer_v2.03_2.pt", map_location=self.gpu_device)
        self.transmission_models["dehazer_v2.03_2"] = cg.Generator(input_nc=3, output_nc=1, n_residual_blocks=10).to(self.gpu_device)
        self.transmission_models["dehazer_v2.03_2"].load_state_dict(checkpt[constants.GENERATOR_KEY + "T"])

        checkpt = torch.load("checkpoint/dehazer_v2.03_2.pt", map_location=self.gpu_device)
        self.atmosphere_models["dehazer_v2.03_2"] = cg.Generator(input_nc=6, output_nc=3, n_residual_blocks=10).to(self.gpu_device)
        self.atmosphere_models["dehazer_v2.03_2"].load_state_dict(checkpt[constants.GENERATOR_KEY + "A"])

        checkpt = torch.load("checkpoint/dehazer_v2.03_3.pt", map_location=self.gpu_device)
        self.transmission_models["dehazer_v2.03_3"] = cg.Generator(input_nc=3, output_nc=1, n_residual_blocks=10).to(self.gpu_device)
        self.transmission_models["dehazer_v2.03_3"].load_state_dict(checkpt[constants.GENERATOR_KEY + "T"])

        checkpt = torch.load("checkpoint/dehazer_v2.03_3.pt", map_location=self.gpu_device)
        self.atmosphere_models["dehazer_v2.03_3"] = cg.Generator(input_nc=6, output_nc=3, n_residual_blocks=10).to(self.gpu_device)
        self.atmosphere_models["dehazer_v2.03_3"].load_state_dict(checkpt[constants.GENERATOR_KEY + "A"])

        checkpt = torch.load("checkpoint/dehazer_v2.03_4.pt", map_location=self.gpu_device)
        self.transmission_models["dehazer_v2.03_4"] = cg.Generator(input_nc=3, output_nc=1, n_residual_blocks=10).to(self.gpu_device)
        self.transmission_models["dehazer_v2.03_4"].load_state_dict(checkpt[constants.GENERATOR_KEY + "T"])

        checkpt = torch.load("checkpoint/dehazer_v2.03_4.pt", map_location=self.gpu_device)
        self.atmosphere_models["dehazer_v2.03_4"] = cg.Generator(input_nc=6, output_nc=3, n_residual_blocks=10).to(self.gpu_device)
        self.atmosphere_models["dehazer_v2.03_4"].load_state_dict(checkpt[constants.GENERATOR_KEY + "A"])

        checkpt = torch.load("checkpoint/dehazer_v2.03_5.pt", map_location=self.gpu_device)
        self.transmission_models["dehazer_v2.03_5"] = cg.Generator(input_nc=3, output_nc=1, n_residual_blocks=10).to(self.gpu_device)
        self.transmission_models["dehazer_v2.03_5"].load_state_dict(checkpt[constants.GENERATOR_KEY + "T"])

        checkpt = torch.load("checkpoint/dehazer_v2.03_5.pt", map_location=self.gpu_device)
        self.atmosphere_models["dehazer_v2.03_5"] = cg.Generator(input_nc=6, output_nc=3, n_residual_blocks=10).to(self.gpu_device)
        self.atmosphere_models["dehazer_v2.03_5"].load_state_dict(checkpt[constants.GENERATOR_KEY + "A"])

        checkpt = torch.load("checkpoint/dehazer_v2.03_8.pt", map_location=self.gpu_device)
        self.transmission_models["dehazer_v2.03_8"] = cg.Generator(input_nc=3, output_nc=1, n_residual_blocks=10).to(self.gpu_device)
        self.transmission_models["dehazer_v2.03_8"].load_state_dict(checkpt[constants.GENERATOR_KEY + "T"])

        checkpt = torch.load("checkpoint/dehazer_v2.03_8.pt", map_location=self.gpu_device)
        self.atmosphere_models["dehazer_v2.03_8"] = cg.Generator(input_nc=6, output_nc=3, n_residual_blocks=10).to(self.gpu_device)
        self.atmosphere_models["dehazer_v2.03_8"].load_state_dict(checkpt[constants.GENERATOR_KEY + "A"])

        checkpt = torch.load("checkpoint/dehazer_v2.03_9.pt", map_location=self.gpu_device)
        self.transmission_models["dehazer_v2.03_9"] = cg.Generator(input_nc=3, output_nc=1, n_residual_blocks=10).to(self.gpu_device)
        self.transmission_models["dehazer_v2.03_9"].load_state_dict(checkpt[constants.GENERATOR_KEY + "T"])

        checkpt = torch.load("checkpoint/dehazer_v2.03_9.pt", map_location=self.gpu_device)
        self.atmosphere_models["dehazer_v2.03_9"] = cg.Generator(input_nc=6, output_nc=3, n_residual_blocks=10).to(self.gpu_device)
        self.atmosphere_models["dehazer_v2.03_9"].load_state_dict(checkpt[constants.GENERATOR_KEY + "A"])

        checkpt = torch.load("checkpoint/dehazer_v2.05_1.pt", map_location=self.gpu_device)
        self.transmission_models["dehazer_v2.05_1"] = cg.Generator(input_nc=3, output_nc=1, n_residual_blocks=10).to(self.gpu_device)
        self.transmission_models["dehazer_v2.05_1"].load_state_dict(checkpt[constants.GENERATOR_KEY + "T"])

        checkpt = torch.load("checkpoint/dehazer_v2.05_1.pt", map_location=self.gpu_device)
        self.atmosphere_models["dehazer_v2.05_1"] = cg.Generator(input_nc=6, output_nc=3, n_residual_blocks=10).to(self.gpu_device)
        self.atmosphere_models["dehazer_v2.05_1"].load_state_dict(checkpt[constants.GENERATOR_KEY + "A"])

        checkpt = torch.load("checkpoint/dehazer_v2.05_2.pt", map_location=self.gpu_device)
        self.transmission_models["dehazer_v2.05_2"] = cg.Generator(input_nc=3, output_nc=1, n_residual_blocks=10).to(self.gpu_device)
        self.transmission_models["dehazer_v2.05_2"].load_state_dict(checkpt[constants.GENERATOR_KEY + "T"])

        checkpt = torch.load("checkpoint/dehazer_v2.05_2.pt", map_location=self.gpu_device)
        self.atmosphere_models["dehazer_v2.05_2"] = cg.Generator(input_nc=6, output_nc=3, n_residual_blocks=10).to(self.gpu_device)
        self.atmosphere_models["dehazer_v2.05_2"].load_state_dict(checkpt[constants.GENERATOR_KEY + "A"])

        checkpt = torch.load("checkpoint/dehazer_v2.05_3.pt", map_location=self.gpu_device)
        self.transmission_models["dehazer_v2.05_3"] = cg.Generator(input_nc=3, output_nc=1, n_residual_blocks=10).to(self.gpu_device)
        self.transmission_models["dehazer_v2.05_3"].load_state_dict(checkpt[constants.GENERATOR_KEY + "T"])

        checkpt = torch.load("checkpoint/dehazer_v2.05_3.pt", map_location=self.gpu_device)
        self.atmosphere_models["dehazer_v2.05_3"] = cg.Generator(input_nc=6, output_nc=3, n_residual_blocks=10).to(self.gpu_device)
        self.atmosphere_models["dehazer_v2.05_3"].load_state_dict(checkpt[constants.GENERATOR_KEY + "A"])

        checkpt = torch.load("checkpoint/dehazer_v2.06_2.pt", map_location=self.gpu_device)
        self.transmission_models["dehazer_v2.06_2"] = cg.Generator(input_nc=3, output_nc=1, n_residual_blocks=10).to(self.gpu_device)
        self.transmission_models["dehazer_v2.06_2"].load_state_dict(checkpt[constants.GENERATOR_KEY + "T"])

        checkpt = torch.load("checkpoint/dehazer_v2.06_2.pt", map_location=self.gpu_device)
        self.atmosphere_models["dehazer_v2.06_2"] = cg.Generator(input_nc=6, output_nc=3, n_residual_blocks=10).to(self.gpu_device)
        self.atmosphere_models["dehazer_v2.06_2"].load_state_dict(checkpt[constants.GENERATOR_KEY + "A"])

        checkpt = torch.load("checkpoint/dehazer_v2.06_3.pt", map_location=self.gpu_device)
        self.transmission_models["dehazer_v2.06_3"] = cg.Generator(input_nc=3, output_nc=1, n_residual_blocks=10).to(self.gpu_device)
        self.transmission_models["dehazer_v2.06_3"].load_state_dict(checkpt[constants.GENERATOR_KEY + "T"])

        checkpt = torch.load("checkpoint/dehazer_v2.06_3.pt", map_location=self.gpu_device)
        self.atmosphere_models["dehazer_v2.06_3"] = cg.Generator(input_nc=6, output_nc=3, n_residual_blocks=10).to(self.gpu_device)
        self.atmosphere_models["dehazer_v2.06_3"].load_state_dict(checkpt[constants.GENERATOR_KEY + "A"])

        checkpt = torch.load("checkpoint/dehazer_v2.06_4.pt", map_location=self.gpu_device)
        self.transmission_models["dehazer_v2.06_4"] = cg.Generator(input_nc=3, output_nc=1, n_residual_blocks=10).to(self.gpu_device)
        self.transmission_models["dehazer_v2.06_4"].load_state_dict(checkpt[constants.GENERATOR_KEY + "T"])

        checkpt = torch.load("checkpoint/dehazer_v2.06_4.pt", map_location=self.gpu_device)
        self.atmosphere_models["dehazer_v2.06_4"] = cg.Generator(input_nc=6, output_nc=3, n_residual_blocks=10).to(self.gpu_device)
        self.atmosphere_models["dehazer_v2.06_4"].load_state_dict(checkpt[constants.GENERATOR_KEY + "A"])

        checkpt = torch.load("checkpoint/dehazer_v2.06_5.pt", map_location=self.gpu_device)
        self.transmission_models["dehazer_v2.06_5"] = cg.Generator(input_nc=3, output_nc=1, n_residual_blocks=10).to(self.gpu_device)
        self.transmission_models["dehazer_v2.06_5"].load_state_dict(checkpt[constants.GENERATOR_KEY + "T"])

        checkpt = torch.load("checkpoint/dehazer_v2.06_5.pt", map_location=self.gpu_device)
        self.atmosphere_models["dehazer_v2.06_5"] = cg.Generator(input_nc=6, output_nc=3, n_residual_blocks=10).to(self.gpu_device)
        self.atmosphere_models["dehazer_v2.06_5"].load_state_dict(checkpt[constants.GENERATOR_KEY + "A"])

        checkpt = torch.load("checkpoint/airlight_estimator_v1.05_1.pt")
        self.atmosphere_models["airlight_estimator_v1.05_1" + str(AtmosphereMethod.NETWORK_ESTIMATOR_V1)] = \
            dh.AirlightEstimator_V2(num_channels=3, disc_feature_size=64, out_features=3).to(self.gpu_device)
        self.atmosphere_models["airlight_estimator_v1.05_1" + str(AtmosphereMethod.NETWORK_ESTIMATOR_V1)].load_state_dict(checkpt[constants.DISCRIMINATOR_KEY + "A"])

        checkpt = torch.load("checkpoint/airlight_estimator_v1.06_1.pt")
        self.atmosphere_models["airlight_estimator_v1.06_1"] = \
            dh.AirlightEstimator_V2(num_channels=3, disc_feature_size=64, out_features=3).to(self.gpu_device)
        self.atmosphere_models["airlight_estimator_v1.06_1"].load_state_dict(checkpt[constants.DISCRIMINATOR_KEY + "A"])

        checkpt = torch.load("checkpoint/airlight_estimator_v1.07_1.pt")
        self.atmosphere_models["airlight_estimator_v1.07_1"] = \
            dh.AirlightEstimator_V2(num_channels=3, disc_feature_size=64, out_features=3).to(self.gpu_device)
        self.atmosphere_models["airlight_estimator_v1.07_1"].load_state_dict(checkpt[constants.DISCRIMINATOR_KEY + "A"])

        checkpt = torch.load("checkpoint/airlight_gen_v1.03_1.pt")
        self.atmosphere_models["airlight_gen_v1.03_1"] = cg.Generator(input_nc=6, output_nc=3, n_residual_blocks=10).to(self.gpu_device)
        self.atmosphere_models["airlight_gen_v1.03_1"].load_state_dict(checkpt[constants.GENERATOR_KEY + "A"])

        checkpt = torch.load("checkpoint/airlight_gen_v1.03_2.pt")
        self.atmosphere_models["airlight_gen_v1.03_2"] = cg.Generator(input_nc=6, output_nc=3, n_residual_blocks=10).to(self.gpu_device)
        self.atmosphere_models["airlight_gen_v1.03_2"].load_state_dict(checkpt[constants.GENERATOR_KEY + "A"])

        checkpt = torch.load("checkpoint/airlight_gen_v1.06_4.pt")
        self.atmosphere_models["airlight_gen_v1.06_4"] = cg.Generator(input_nc=6, output_nc=3, n_residual_blocks=10).to(self.gpu_device)
        self.atmosphere_models["airlight_gen_v1.06_4"].load_state_dict(checkpt[constants.GENERATOR_KEY + "A"])

        checkpt = torch.load("checkpoint/airlight_gen_v1.07_1.pt")
        self.atmosphere_models["airlight_gen_v1.07_1"] = cg.Generator(input_nc=6, output_nc=3, n_residual_blocks=10).to(self.gpu_device)
        self.atmosphere_models["airlight_gen_v1.07_1"].load_state_dict(checkpt[constants.GENERATOR_KEY + "A"])

        checkpt = torch.load("checkpoint/airlight_gen_v1.08_1.pt")
        self.atmosphere_models["airlight_gen_v1.08_1"] = cg.Generator(input_nc=6, output_nc=1, n_residual_blocks=10).to(self.gpu_device)
        self.atmosphere_models["airlight_gen_v1.08_1"].load_state_dict(checkpt[constants.GENERATOR_KEY + "A"])

        checkpt = torch.load("checkpoint/airlight_estimator_v1.08_1.pt")
        self.atmosphere_models["airlight_estimator_v1.08_1"] = dh.AirlightEstimator_Residual(num_channels = 3, out_features = 3, num_layers = 4).to(self.gpu_device)
        self.atmosphere_models["airlight_estimator_v1.08_1"].load_state_dict(checkpt[constants.DISCRIMINATOR_KEY + "A"])

        print("Dehazing models loaded.")

    def set_models(self, albedo_model_name, transmission_model_name, airlight_estimator_name, atmosphere_method):
        self.albedo_model_key = albedo_model_name
        self.transmission_model_key = transmission_model_name
        self.airlight_model_key = airlight_estimator_name + str(atmosphere_method)


    def set_models_v2(self, albedo_model_name, transmission_model_name, airlight_estimator_name):
        self.albedo_model_key = albedo_model_name
        self.transmission_model_key = transmission_model_name
        #self.airlight_gen_key = airlight_gen_name
        self.airlight_model_key = airlight_estimator_name

    def perform_dehazing_direct_v2(self, hazy_img):
        transform_op = transforms.Compose([transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        hazy_tensor = transform_op(cv2.cvtColor(hazy_img, cv2.COLOR_BGR2RGB)).to(self.gpu_device)
        hazy_tensor = torch.unsqueeze(hazy_tensor, 0)

        # translate to albedo first
        unlit_tensor = self.albedo_models[self.albedo_model_key](hazy_tensor)
        concat_input = torch.cat([hazy_tensor, unlit_tensor], 1)

        transmission_img = self.transmission_models[self.transmission_model_key](hazy_tensor)
        transmission_img = torch.squeeze(transmission_img).cpu().numpy()

        # remove 0.5 normalization for dehazing equation
        T = ((transmission_img * 0.5) + 0.5)
        #T = T * 0.95
        #hazy_tensor = interpolate(hazy_tensor, constants.TEST_IMAGE_SIZE)
        # print("Shape: ", np.shape(hazy_tensor))

        atmosphere_map = self.atmosphere_models[self.airlight_model_key](concat_input)  # normalize to 0.0 - 1.0
        atmosphere_map = torch.squeeze(atmosphere_map).cpu().numpy()
        #atmosphere_map = atmosphere_map - 0.1

        # remove 0.5 normalization for dehazing equation
        A = ((atmosphere_map * 0.5) + 0.5)
        #A = A * 1.2

        print("Shape of atmosphere map: ", np.shape(atmosphere_map))

        hazy_img = cv2.normalize(hazy_img, dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        clear_img = np.ones_like(hazy_img)
        T = np.resize(T, np.shape(clear_img[:, :, 0]))
        # print("Airlight estimator network loaded. Shapes: ", np.shape(clear_img), np.shape(hazy_img), np.shape(T), " Atmosphere estimate: ", np.shape(atmosphere))

        clear_img[:, :, 0] = (hazy_img[:, :, 0] - A[0]) / T
        clear_img[:, :, 1] = (hazy_img[:, :, 1] - A[1]) / T
        clear_img[:, :, 2] = (hazy_img[:, :, 2] - A[2]) / T

        return np.clip(clear_img, 0.0, 1.0)

    def extract_atmosphere_element(self, hazy_tensor):
        #extract dark channel
        dc_kernel = torch.ones(3, 3).to(self.gpu_device)

        hazy_tensor = hazy_tensor.transpose(0, 1)
        (r, g, b) = torch.chunk(hazy_tensor, 3)
        (h, w) = (np.shape(r)[2], np.shape(r)[3])
        #print("R G B shape: ", np.shape(r), np.shape(g), np.shape(b))
        dc_tensor = torch.minimum(torch.minimum(r, g), b)
        dc_tensor = kornia.morphology.erosion(dc_tensor, dc_kernel)

        #estimate atmosphere
        dc_tensor = dc_tensor.transpose(0, 1)
        hazy_tensor = hazy_tensor.transpose(0, 1)
        A_map = torch.zeros_like(hazy_tensor)
        for i in range(np.shape(dc_tensor)[0]):
            A = estimate_atmosphere(hazy_tensor.cpu().numpy()[i], dc_tensor.cpu().numpy()[i], h, w)
            A = np.ndarray.flatten(A)

            A_map[i, 0] = torch.full_like(A_map[i, 0], A[0])
            A_map[i, 1] = torch.full_like(A_map[i, 1], A[1])
            A_map[i, 2] = torch.full_like(A_map[i, 2], A[2])

        a_tensor = A_map.to(self.gpu_device)

        return a_tensor

    def perform_dehazing_direct_v3(self, hazy_img, filter_strength):
        transform_op = transforms.Compose([transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        hazy_tensor = transform_op(cv2.cvtColor(hazy_img, cv2.COLOR_BGR2RGB)).to(self.gpu_device)
        hazy_tensor = torch.unsqueeze(hazy_tensor, 0)
        #a_tensor = self.extract_atmosphere_element(hazy_tensor)

        # translate to albedo first
        unlit_tensor = self.albedo_models[self.albedo_model_key](hazy_tensor)
        concat_input = torch.cat([hazy_tensor, unlit_tensor], 1)

        transmission_img = self.transmission_models[self.transmission_model_key](hazy_tensor)
        transmission_img = torch.squeeze(transmission_img).cpu().numpy()

        # remove 0.5 normalization for dehazing equation
        T = ((transmission_img * 0.5) + 0.5)
        #T = T * 1.2

        atmosphere_map = self.atmosphere_models[self.airlight_model_key](concat_input)
        atmosphere_map = torch.squeeze(atmosphere_map).cpu().numpy()
        #atmosphere_map = atmosphere_map - 0.1

        # remove 0.5 normalization for dehazing equation
        A = ((atmosphere_map * 0.5) + 0.5)
        #A = A * 0.35

        print("Shape of atmosphere map: ", np.shape(atmosphere_map))

        hazy_img = cv2.normalize(hazy_img, dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        clear_img = np.ones_like(hazy_img)
        T = np.resize(T, np.shape(clear_img[:, :, 0]))
        # print("Airlight estimator network loaded. Shapes: ", np.shape(clear_img), np.shape(hazy_img), np.shape(T), " Atmosphere estimate: ", np.shape(atmosphere))

        clear_img[:, :, 0] = (hazy_img[:, :, 0] - A[0]) / np.maximum(T, filter_strength)
        clear_img[:, :, 1] = (hazy_img[:, :, 1] - A[1]) / np.maximum(T, filter_strength)
        clear_img[:, :, 2] = (hazy_img[:, :, 2] - A[2]) / np.maximum(T, filter_strength)

        return np.clip(clear_img, 0.0, 1.0)

    def perform_dehazing_direct_v4(self, hazy_img, filter_strength):
        grey_transform_op = transforms.Compose([transforms.ToTensor(),
                                               transforms.Normalize((0.5), (0.5))])

        rgb_transform_op = transforms.Compose([transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        hazy_tensor = rgb_transform_op(cv2.cvtColor(hazy_img, cv2.COLOR_BGR2RGB)).to(self.gpu_device)
        hazy_tensor = torch.unsqueeze(hazy_tensor, 0)

        # translate to albedo first
        unlit_tensor = self.albedo_models[self.albedo_model_key](hazy_tensor)

        concat_input = torch.cat([hazy_tensor, unlit_tensor], 1)

        transmission_img = self.transmission_models[self.transmission_model_key](hazy_tensor)
        transmission_img = torch.squeeze(transmission_img).cpu().numpy()

        # remove 0.5 normalization for dehazing equation
        T = ((transmission_img * 0.5) + 0.5)
        #T = T * 1.2

        # one_minus_t = self.atmosphere_models[self.airlight_gen_key](concat_input)
        # one_minus_t = torch.squeeze(one_minus_t).cpu().numpy()
        # one_minus_t = ((one_minus_t * 0.5) + 0.5)

        atmospheric_term = self.atmosphere_models[self.airlight_model_key](hazy_tensor)
        atmospheric_term = torch.squeeze(atmospheric_term).cpu().numpy()

        atmosphere_map = np.zeros_like(hazy_img, dtype=np.float32)
        atmosphere_map[:, :, 0] = atmospheric_term[0] * (1 - T)
        atmosphere_map[:, :, 1] = atmospheric_term[1] * (1 - T)
        atmosphere_map[:, :, 2] = atmospheric_term[2] * (1 - T)

        # remove 0.5 normalization for dehazing equation
        A = atmosphere_map
        #A = ((atmosphere_map * 0.5) + 0.5)
        #A = A * 0.5

        #print("Shape of atmosphere map: ", np.shape(atmosphere_map))

        hazy_img = cv2.normalize(hazy_img, dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        clear_img = np.ones_like(hazy_img)
        T = np.resize(T, np.shape(clear_img[:, :, 0]))
        # print("Airlight estimator network loaded. Shapes: ", np.shape(clear_img), np.shape(hazy_img), np.shape(T), " Atmosphere estimate: ", np.shape(atmosphere))

        clear_img[:, :, 0] = (hazy_img[:, :, 0] - A[:, :, 0]) / np.maximum(T, filter_strength)
        clear_img[:, :, 1] = (hazy_img[:, :, 1] - A[:, :, 1]) / np.maximum(T, filter_strength)
        clear_img[:, :, 2] = (hazy_img[:, :, 2] - A[:, :, 2]) / np.maximum(T, filter_strength)

        T = transforms.F.to_tensor(T)
        A = transforms.F.to_tensor(A)

        return np.clip(clear_img, 0.0, 1.0), T, A

    def perform_dehazing_direct(self, hazy_img, atmosphere_sensitivity):

        transform_op = transforms.Compose([transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        hazy_tensor = transform_op(cv2.cvtColor(hazy_img, cv2.COLOR_BGR2RGB)).to(self.gpu_device)
        hazy_tensor = torch.unsqueeze(hazy_tensor, 0)

        #translate to albedo first
        hazy_tensor = self.albedo_models[self.albedo_model_key](hazy_tensor)

        transmission_img = self.transmission_models[self.transmission_model_key](hazy_tensor)
        transmission_img = torch.squeeze(transmission_img).cpu().numpy()

        # remove 0.5 normalization for dehazing equation
        T = ((transmission_img * 0.5) + 0.5)
        #T = T * 0.9
        hazy_tensor = interpolate(hazy_tensor, constants.TEST_IMAGE_SIZE)
        #print("Shape: ", np.shape(hazy_tensor))

        A1 = (self.atmosphere_models[self.airlight_model_key](hazy_tensor).cpu() * 0.5) + 0.5  # normalize to 0.0 - 1.0
        #A1 = (self.atmosphere_models[self.airlight_model_key](hazy_tensor).cpu())
        A1 = torch.squeeze(A1)
        A1 = A1 - atmosphere_sensitivity
        A1 = A1.numpy()

        A2 = estimate_atmosphere(hazy_img, get_dark_channel(hazy_img, 4)) / 255.0
        print("A difference: ", (A1 - A2))

        atmosphere = A1 - atmosphere_sensitivity
        atmosphere = torch.from_numpy(atmosphere)
        atmosphere = torch.squeeze(atmosphere).numpy()

        hazy_img = cv2.normalize(hazy_img, dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        clear_img = np.ones_like(hazy_img)
        T = np.resize(T, np.shape(clear_img[:, :, 0]))
        #print("Airlight estimator network loaded. Shapes: ", np.shape(clear_img), np.shape(hazy_img), np.shape(T), " Atmosphere estimate: ", np.shape(atmosphere))

        clear_img[:, :, 0] = (hazy_img[:, :, 0] - np.multiply(1 - T, atmosphere[0])) / T
        clear_img[:, :, 1] = (hazy_img[:, :, 1] - np.multiply(1 - T, atmosphere[1])) / T
        clear_img[:, :, 2] = (hazy_img[:, :, 2] - np.multiply(1 - T, atmosphere[2])) / T

        return np.clip(clear_img, 0.0, 1.0)

    def perform_dehazing(self, hazy_img, filter_strength, atmosphere_sensitivity):

        transform_op = transforms.Compose([transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        hazy_tensor = transform_op(cv2.cvtColor(hazy_img, cv2.COLOR_BGR2RGB)).to(self.gpu_device)
        hazy_tensor = torch.unsqueeze(hazy_tensor, 0)
        transmission_img = self.transmission_models[self.transmission_model_key](hazy_tensor)
        transmission_img = torch.squeeze(transmission_img).cpu().numpy()

        # remove 0.5 normalization for dehazing equation
        T = ((transmission_img * 0.5) + 0.5)
        hazy_tensor = interpolate(hazy_tensor, constants.TEST_IMAGE_SIZE)
        print("Shape: ", np.shape(hazy_tensor))
        # translate to albedo first
        unlit_tensor = self.albedo_models[self.albedo_model_key](hazy_tensor)
        concat_input = torch.cat([hazy_tensor, unlit_tensor], 1)

        atmosphere = (self.atmosphere_models[self.airlight_model_key](hazy_tensor).cpu() * 0.5) + 0.5  # normalize to 0.0 - 1.0
        atmosphere = torch.squeeze(atmosphere)
        atmosphere = atmosphere - atmosphere_sensitivity
        atmosphere = atmosphere.numpy()

        hazy_img = cv2.normalize(hazy_img, dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        clear_img = np.ones_like(hazy_img)
        T = np.resize(T, np.shape(clear_img[:, :, 0]))
        print("Airlight estimator network loaded. Shapes: ", np.shape(clear_img), np.shape(hazy_img), np.shape(T),
        "Atmosphere estimate: ", atmosphere)
        clear_img[:, :, 0] = ((hazy_img[:, :, 0] - np.full(np.shape(hazy_img[:, :, 0]), atmosphere[0])) / np.maximum(T, filter_strength)) + np.full(
            np.shape(hazy_img[:, :, 0]), atmosphere[0])
        clear_img[:, :, 1] = ((hazy_img[:, :, 1] - np.full(np.shape(hazy_img[:, :, 1]), atmosphere[1])) / np.maximum(T, filter_strength)) + np.full(
            np.shape(hazy_img[:, :, 1]), atmosphere[1])
        clear_img[:, :, 2] = ((hazy_img[:, :, 2] - np.full(np.shape(hazy_img[:, :, 2]), atmosphere[2])) / np.maximum(T, filter_strength)) + np.full(
            np.shape(hazy_img[:, :, 2]), atmosphere[2])

        return np.clip(clear_img, 0.0, 1.0)

    # def derive_T_and_A(self, hazy_img):
    #     transform_op = transforms.Compose([transforms.ToTensor(),
    #                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    #
    #
    #     hazy_tensor = transform_op(cv2.cvtColor(hazy_img, cv2.COLOR_BGR2RGB)).to(self.gpu_device)
    #     hazy_tensor = torch.unsqueeze(hazy_tensor, 0)
    #     hazy_tensor = interpolate(hazy_tensor, constants.TEST_IMAGE_SIZE)
    #
    #     # translate to albedo first
    #     unlit_tensor = self.albedo_models[self.albedo_model_key](hazy_tensor)
    #     concat_input = torch.cat([hazy_tensor, unlit_tensor], 1)
    #
    #     transmission_img = self.transmission_models[self.transmission_model_key](hazy_tensor)
    #     transmission_img = torch.squeeze(transmission_img).cpu()
    #
    #     # remove 0.5 normalization for dehazing equation
    #     T = ((transmission_img * 0.5) + 0.5)
    #     #T = T * 0.5
    #     # hazy_tensor = interpolate(hazy_tensor, constants.TEST_IMAGE_SIZE)
    #     # print("Shape: ", np.shape(hazy_tensor))
    #
    #     atmosphere_map = self.atmosphere_models[self.airlight_model_key](concat_input)  # normalize to 0.0 - 1.0
    #     atmosphere_map = torch.squeeze(atmosphere_map).cpu()
    #     # atmosphere_map = atmosphere_map - 0.1
    #
    #     # remove 0.5 normalization for dehazing equation
    #     #A = ((atmosphere_map * 0.5) + 0.5)
    #     A = ((atmosphere_map * 0.5) + 0.5)
    #
    #     #A_img = cv2.normalize(A_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    #
    #     return T, A


