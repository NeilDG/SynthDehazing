import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

def get_dark_channel(I, w = 15):
    b,g,r = cv2.split(I)
    dc = cv2.min(cv2.min(r,g),b);
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(w,w))
    dark = cv2.erode(dc,kernel)
    return dark


def estimate_transmission(im, A, dark_channel):
    omega = 0.95;
    im3 = np.empty(im.shape, im.dtype);

    for ind in range(0, 3):
        im3[:, :, ind] = im[:, :, ind] / A[0, ind]

    transmission = 1 - omega * dark_channel
    return transmission

def estimate_atmosphere(im, dark):
    #im = im.cpu().numpy()
    [h,w] = [np.shape(im)[0], np.shape(im)[1]]

    imsz = h * w
    numpx = int(max(math.floor(imsz/1000),1))
    darkvec = dark.reshape(imsz,1);
    imvec = im.reshape(imsz,3)

    indices = darkvec.argsort();
    indices = indices[imsz-numpx::]

    atmsum = np.zeros([1,3])
    for ind in range(1,numpx):
       atmsum = atmsum + imvec[indices[ind]]

    A = atmsum / numpx;

    return A

def recover(im,t,A,tx = 0.1):
    res = np.empty(im.shape,im.dtype);
    t = cv2.max(t,tx);

    for ind in range(0,3):
        res[:,:,ind] = (im[:,:,ind]-A[0,ind])/t + A[0,ind]

    return res

def perform_dcp_dehaze(input_img, normalize = False):
    input_img = cv2.normalize(input_img, dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    dark_channel = get_dark_channel(input_img)

    if(normalize):
        dark_channel = cv2.normalize(dark_channel, dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    atmosphere = estimate_atmosphere(input_img, dark_channel)
    T = estimate_transmission(input_img, atmosphere, dark_channel)

    #if(normalize):
    T = cv2.normalize(T, dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # plt.imshow(T)
    # plt.show()

    clear_img = np.clip(recover(input_img, T, atmosphere, 0.1), 0.0, 1.0)
    #clear_img = recover(input_img, T, atmosphere, 0.8)
    return clear_img

