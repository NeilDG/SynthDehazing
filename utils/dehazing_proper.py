import constants
import torch
import numpy as np
from enum import Enum
from torchvision import transforms
import cv2
from model import dehaze_discriminator as dh
from model import vanilla_cycle_gan as cycle_gan
from model import ffa_net as ffa_gan
import math
from utils import pytorch_colors
from itertools import combinations_with_replacement
from collections import defaultdict

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
    # return np.power(np.e, -beta * depth_map)

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

def estimate_atmosphere(im, dark):
    # im = im.cpu().numpy()
    [h, w] = [np.shape(im)[0], np.shape(im)[1]]

    imsz = h * w
    numpx = int(max(math.floor(imsz / 1000), 1))
    darkvec = dark.reshape(imsz, 1);
    imvec = im.reshape(imsz, 3)

    indices = darkvec.argsort();
    indices = indices[imsz - numpx::]

    atmsum = np.zeros([1, 3])
    for ind in range(1, numpx):
        atmsum = atmsum + imvec[indices[ind]]

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
        self.transmission_models = {}
        self.atmosphere_models = {}

        #add models
        checkpt = torch.load("checkpoint/transmission_estimator_v1.02_2.pt")
        self.transmission_models["transmission_estimator_v1.02_2"] = cycle_gan.Generator(input_nc=3, output_nc=1, n_residual_blocks=8).to(self.gpu_device)
        self.transmission_models["transmission_estimator_v1.02_2"].load_state_dict(checkpt[constants.GENERATOR_KEY + "A"])

        checkpt = torch.load("checkpoint/airlight_estimator_v1.02_1.pt")
        self.atmosphere_models["airlight_estimator_v1.02_1" + str(AtmosphereMethod.NETWORK_ESTIMATOR_V1)] = dh.AirlightEstimator_V1(input_nc=3, downsampling_layers=3, residual_blocks=5).to(self.gpu_device)
        self.atmosphere_models["airlight_estimator_v1.02_1" + str(AtmosphereMethod.NETWORK_ESTIMATOR_V1)].load_state_dict(checkpt[constants.DISCRIMINATOR_KEY + "A"])

        self.atmosphere_models["airlight_estimator_v1.02_1" + str(AtmosphereMethod.NETWORK_ESTIMATOR_V2)] = dh.AirlightEstimator_V1(input_nc=6, downsampling_layers = 3, residual_blocks = 5).to(self.gpu_device)
        self.atmosphere_models["airlight_estimator_v1.02_1" + str(AtmosphereMethod.NETWORK_ESTIMATOR_V2)].load_state_dict(checkpt[constants.DISCRIMINATOR_KEY + "B"])

        print("Dehazing models loaded.")

    def set_models(self, transmission_model_name, airlight_estimator_name, atmosphere_method):
        self.transmission_model_key = transmission_model_name
        self.airlight_model_key = airlight_estimator_name + str(atmosphere_method)

    def perform_dehazing(self, hazy_img, hazy_tensor, filter_strength):
        transmission_img = self.transmission_models[self.transmission_model_key](torch.unsqueeze(hazy_tensor, 0))
        transmission_img = torch.squeeze(transmission_img).cpu().numpy()

        # remove 0.5 normalization for dehazing equation
        T = ((transmission_img * 0.5) + 0.5)
        hazy_img = ((hazy_img * 0.5) + 0.5)
        hazy_img = cv2.normalize(hazy_img, dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        transform_op = transforms.Compose([transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        hazy_tensor = torch.unsqueeze(transform_op(hazy_img), 0).to(self.gpu_device)
        atmosphere = self.atmosphere_models[self.airlight_model_key](hazy_tensor)
        atmosphere = atmosphere.cpu().item()

        clear_img = np.ones_like(hazy_img)
        T = np.resize(T, np.shape(clear_img[:, :, 0]))
        print("Airlight estimator network loaded. Shapes: ", np.shape(clear_img), np.shape(hazy_img), np.shape(T),
        "Atmosphere estimate: ", atmosphere)
        clear_img[:, :, 0] = ((hazy_img[:, :, 0] - np.full(np.shape(hazy_img[:, :, 0]), atmosphere)) / np.maximum(T, filter_strength)) + np.full(
            np.shape(hazy_img[:, :, 0]), atmosphere)
        clear_img[:, :, 1] = ((hazy_img[:, :, 1] - np.full(np.shape(hazy_img[:, :, 1]), atmosphere)) / np.maximum(T, filter_strength)) + np.full(
            np.shape(hazy_img[:, :, 1]), atmosphere)
        clear_img[:, :, 2] = ((hazy_img[:, :, 2] - np.full(np.shape(hazy_img[:, :, 2]), atmosphere)) / np.maximum(T, filter_strength)) + np.full(
            np.shape(hazy_img[:, :, 2]), atmosphere)

        return np.clip(clear_img, 0.0, 1.0)

def perform_dehazing_equation_with_transmission(hazy_img, T, atmosphere_method, airlight_checkpt, filter_strength=0.1):
    hazy_img = cv2.normalize(hazy_img, dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    atmosphere = estimate_atmosphere(hazy_img, get_dark_channel(hazy_img, w=15))
    atmosphere = np.squeeze(atmosphere)
    #print("Estimated atmosphere: ", atmosphere)

    print("Min of input: ", np.min(hazy_img), " Max of input: ", np.max(hazy_img),
          "Min of T: ", np.min(T), " Max of T: ", np.max(T))

    # compute clear image with radiance term
    if (atmosphere_method == AtmosphereMethod.SCENE_RADIANCE):
        clear_img = np.ones_like(hazy_img)
        T = np.resize(T, np.shape(clear_img[:, :, 0]))
        print("Shapes: ", np.shape(clear_img), np.shape(hazy_img), np.shape(T))
        clear_img[:, :, 0] = ((hazy_img[:, :, 0] - np.full(np.shape(hazy_img[:, :, 0]), atmosphere[0])) / np.maximum(T,filter_strength)) + np.full(
            np.shape(hazy_img[:, :, 0]), atmosphere[0])
        clear_img[:, :, 1] = ((hazy_img[:, :, 1] - np.full(np.shape(hazy_img[:, :, 1]), atmosphere[1])) / np.maximum(T,filter_strength)) + np.full(
            np.shape(hazy_img[:, :, 1]), atmosphere[1])
        clear_img[:, :, 2] = ((hazy_img[:, :, 2] - np.full(np.shape(hazy_img[:, :, 2]), atmosphere[2])) / np.maximum(T,filter_strength)) + np.full(
            np.shape(hazy_img[:, :, 2]), atmosphere[2])

    elif(atmosphere_method == AtmosphereMethod.DIRECT):
        clear_img = np.ones_like(hazy_img)
        T = np.resize(T, np.shape(clear_img[:, :, 0]))
        print("Shapes: ", np.shape(clear_img), np.shape(hazy_img), np.shape(T))
        clear_img[:, :, 0] = (hazy_img[:, :, 0] - (np.full(np.shape(hazy_img[:, :, 0]), atmosphere[0]) * (1 - T))) / T
        clear_img[:, :, 1] = (hazy_img[:, :, 1] - (np.full(np.shape(hazy_img[:, :, 1]), atmosphere[1]) * (1 - T))) / T
        clear_img[:, :, 2] = (hazy_img[:, :, 2] - (np.full(np.shape(hazy_img[:, :, 2]), atmosphere[2]) * (1 - T))) / T

    elif (atmosphere_method == AtmosphereMethod.NETWORK_ESTIMATOR_V1):
        device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        airlight_model = dh.AirlightEstimator_V2(num_channels = 3, disc_feature_size = 64, out_features=3).to(device)

        checkpt = torch.load(airlight_checkpt)
        airlight_model.load_state_dict(checkpt[constants.DISCRIMINATOR_KEY + "A"])
        #airlight_model.load_state_dict(checkpt[constants.DISCRIMINATOR_KEY])

        transform_op = transforms.Compose([transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        resized_hazy_img = cv2.resize(hazy_img, constants.TEST_IMAGE_SIZE, cv2.INTER_LINEAR)
        hazy_tensor = torch.unsqueeze(transform_op(resized_hazy_img), 0).to(device)
        atmosphere = (airlight_model(hazy_tensor).cpu() * 0.5) + 0.5 #normalize to 0.0 - 1.0
        atmosphere = torch.squeeze(atmosphere)
        atmosphere = atmosphere - 0.3

        clear_img = np.ones_like(hazy_img)
        T = np.resize(T, np.shape(clear_img[:, :, 0]))
        clear_img[:, :, 0] = ((hazy_img[:, :, 0] - np.full(np.shape(hazy_img[:, :, 0]), atmosphere[0])) / np.maximum(T,filter_strength)) + np.full(
            np.shape(hazy_img[:, :, 0]), atmosphere[0])
        clear_img[:, :, 1] = ((hazy_img[:, :, 1] - np.full(np.shape(hazy_img[:, :, 1]), atmosphere[1])) / np.maximum(T,filter_strength)) + np.full(
            np.shape(hazy_img[:, :, 1]), atmosphere[1])
        clear_img[:, :, 2] = ((hazy_img[:, :, 2] - np.full(np.shape(hazy_img[:, :, 2]), atmosphere[2])) / np.maximum(T,filter_strength)) + np.full(
            np.shape(hazy_img[:, :, 2]), atmosphere[2])

    elif(atmosphere_method == AtmosphereMethod.NETWORK_ESTIMATOR_V2):
        device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        airlight_model = dh.AirlightEstimator_V2(num_channels = 6, disc_feature_size = 64).to(device)
        albedo_model = ffa_gan.FFA(gps = 3, blocks = 18).to(device)

        checkpt = torch.load(airlight_checkpt)
        airlight_model.load_state_dict(checkpt[constants.DISCRIMINATOR_KEY + "B"])
        checkpt = torch.load("checkpoint/albedo_transfer_v1.04_1.pt")
        albedo_model.load_state_dict(checkpt[constants.GENERATOR_KEY + "A"])

        transform_op = transforms.Compose([transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        resized_hazy_img = cv2.resize(hazy_img, constants.TEST_IMAGE_SIZE, cv2.INTER_LINEAR)
        hazy_tensor = torch.unsqueeze(transform_op(resized_hazy_img), 0).to(device)
        atmosphere = airlight_model(torch.cat([hazy_tensor, albedo_model(hazy_tensor)], 1)).cpu().item()

        #S = np.full(np.shape(hazy_img[:, :, 0]), atmosphere) * (1 - T)

        clear_img = np.ones_like(hazy_img)
        T = np.resize(T, np.shape(clear_img[:, :, 0]))
        clear_img[:, :, 0] = ((hazy_img[:, :, 0] - np.full(np.shape(hazy_img[:, :, 0]), atmosphere)) / np.maximum(T,filter_strength)) + np.full(
            np.shape(hazy_img[:, :, 0]), atmosphere)
        clear_img[:, :, 1] = ((hazy_img[:, :, 1] - np.full(np.shape(hazy_img[:, :, 1]), atmosphere)) / np.maximum(T,filter_strength)) + np.full(
            np.shape(hazy_img[:, :, 1]), atmosphere)
        clear_img[:, :, 2] = ((hazy_img[:, :, 2] - np.full(np.shape(hazy_img[:, :, 2]), atmosphere)) / np.maximum(T,filter_strength)) + np.full(
            np.shape(hazy_img[:, :, 2]), atmosphere)

    clear_img = np.clip(clear_img, 0.0, 1.0)
    #clear_img = cv2.normalize(clear_img, dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return clear_img


