# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 20:28:09 2019

Image and tensor utilities
@author: delgallegon
"""
import numbers
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import math
import cv2
from torch.autograd import Variable
import torch
from utils import pytorch_colors
import matplotlib.pyplot as plt

#for attaching hooks on pretrained models
class SaveFeatures(nn.Module):
	features = None;
	def __init__(self, m):
		self.hook = m.register_forward_hook(self.hook_fn);
	def hook_fn(self, module, input, output):
		self.features = output;
	def close(self):
		self.hook.remove();
        

class CombineFeatures(nn.Module):
    features = None;
    def __init(self, m, features):
        self.hook = m.register_forward_hook(self.hook_fn)
        self.features = features
    
    def hook_fn(self, module, input, output):
        self.features = self.features + output
    
    def close(self):
        self.hook.remove()

def normalize_to_matplotimg(img_tensor, batch_idx, std, mean):
    img = img_tensor[batch_idx,:,:,:].numpy()
    img = np.moveaxis(img, -1, 0)
    img = np.moveaxis(img, -1, 0) #for properly displaying image in matplotlib
    
    img = ((img * std) + mean) #normalize back to 0-1 range
    
    img = cv2.convertScaleAbs(img, alpha=(255.0))
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def convert_to_matplotimg(img_tensor, batch_idx):
    img = img_tensor[batch_idx,:,:,:].numpy()
    img = np.moveaxis(img, -1, 0)
    img = np.moveaxis(img, -1, 0) #for properly displaying image in matplotlib
    
    img = cv2.convertScaleAbs(img, alpha=(255.0))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def convert_to_opencv(img_tensor):
    img = img_tensor
    img = np.moveaxis(img, -1, 0)
    img = np.moveaxis(img, -1, 0)
    
    return img

#loads an image compatible with opencv
def load_image(file_path):
    img = cv2.imread(file_path)
    if(img is not None):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    else:
        print("Image ",file_path, " not found.")
    return img

def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram

def preprocess_batch(batch):
    batch = batch.transpose(0, 1)
    (r, g, b) = torch.chunk(batch, 3)
    batch = torch.cat((b, g, r))
    batch = batch.transpose(0, 1)
    return batch

def merge_yuv_results_to_rgb(y_tensor, uv_tensor):
    uv_tensor = uv_tensor.transpose(0, 1)
    y_tensor = y_tensor.transpose(0, 1)
    
    (u, v) = torch.chunk(uv_tensor, 2)
    yuv_tensor = torch.cat((y_tensor, u, v))
    rgb_tensor = pytorch_colors.lab_to_rgb(yuv_tensor.transpose(0, 1))
    #rgb_tensor = ((rgb_tensor * 0.5) + 0.5) #normalize back to 0-1 range
    rgb_tensor = ((rgb_tensor * 1.0) + 1.0)
    return rgb_tensor

def yuv_to_rgb(yuv_tensor):
    rgb_tensor = pytorch_colors.yuv_to_rgb(yuv_tensor)
    return rgb_tensor

def rgb_to_yuv(rgb_tensor):
    return pytorch_colors.rgb_to_yuv(rgb_tensor)

def change_yuv(y_tensor, yuv_tensor):
    yuv_tensor = yuv_tensor.transpose(0, 1)
    y_tensor = y_tensor.transpose(0, 1)
    (y, u, v) = torch.chunk(yuv_tensor, 3)
    yuv_tensor = torch.cat((y_tensor, u, v))
    return yuv_tensor.transpose(0, 1)

def replace_dark_channel(rgb_tensor, dark_channel_old, dark_channel_new, alpha = 0.7, beta = 0.7):
    yuv_tensor = pytorch_colors.rgb_to_yuv(rgb_tensor)
    
    yuv_tensor = yuv_tensor.transpose(0, 1)
    dark_channel_old = dark_channel_old.transpose(0, 1)
    dark_channel_new = dark_channel_new.transpose(0, 1)
    
    (y, u, v) = torch.chunk(yuv_tensor, 3)
    
    #deduct old dark channel from all channels and add new one
    #r = r - dark_channel_old + dark_channel_new
    #g = g - dark_channel_old + dark_channel_new
    #b = b - dark_channel_old + dark_channel_new
    y = y - (dark_channel_old * alpha) + (dark_channel_new * beta)
    
    yuv_tensor = torch.cat((y, u, v))
    rgb_tensor = pytorch_colors.yuv_to_rgb(yuv_tensor.transpose(0, 1))
    return rgb_tensor

def replace_y_channel(rgb_tensor, y_new):
    yuv_tensor = pytorch_colors.rgb_to_yuv(rgb_tensor)
    
    yuv_tensor = yuv_tensor.transpose(0, 1)
    y_new = y_new.transpose(0, 1)
    
    (y, u, v) = torch.chunk(yuv_tensor, 3)
    
    yuv_tensor = torch.cat((y_new, u, v))
    rgb_tensor = pytorch_colors.yuv_to_rgb(yuv_tensor.transpose(0, 1))
    return rgb_tensor

def remove_haze(rgb_tensor, dark_channel_old, dark_channel_new):
    yuv_tensor = pytorch_colors.rgb_to_yuv(rgb_tensor)
    
    yuv_tensor = yuv_tensor.transpose(0, 1)
    dark_channel_old = dark_channel_old.transpose(0, 1)
    dark_channel_new = dark_channel_new.transpose(0, 1)
    
    (y, u, v) = torch.chunk(yuv_tensor, 3)
    
    #remove dark channel from y
    y = y - dark_channel_old
    
    print("Shape of YUV tensor: ", np.shape(yuv_tensor))
    
    #replace with atmosphere and transmission from new dark channel
    atmosphere = estimate_atmosphere(yuv_tensor[:,0,:,:], dark_channel_new[:,0,:,:])
    transmission = estimate_transmission(yuv_tensor[:,0,:,:], atmosphere, dark_channel_new[:,0,:,:]).to('cuda:0')
    
    y = y * transmission
    
    yuv_tensor = torch.cat((y, u, v))
    rgb_tensor = pytorch_colors.yuv_to_rgb(yuv_tensor.transpose(0, 1))
    return rgb_tensor

def get_dark_channel(I, w = 1):
    b,g,r = cv2.split(I)
    dc = cv2.min(cv2.min(r,g),b);
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(w,w))
    dark = cv2.erode(dc,kernel)
    return dark

# def replace_dark_channel(rgb_tensor, dark_channel_new):
#     rgb_img = normalize_to_matplotimg(rgb_tensor.cpu(), 0, 0.5, 0.5)
#     dark_channel_new_img = normalize_to_matplotimg(dark_channel_new.cpu(), 0, 0.5, 0.5)
#     b, g, r = cv2.split(rgb_img)
    
#     dc, mask_r, mask_g, mask_b = get_dark_channel_and_mask(r, g, b)
    
#     #print(np.shape(r), np.shape(dc), np.shape(dark_channel_new_img), np.shape(mask_r))
#     r = cv2.subtract(r, dc, mask = mask_r)
#     g = cv2.subtract(g, dc, mask = mask_g)
#     b = cv2.subtract(b, dc, mask = mask_b)
    
#     r = cv2.add(r, dark_channel_new_img, mask = mask_r)
#     g = cv2.add(g, dark_channel_new_img, mask = mask_g)
#     b = cv2.add(b, dark_channel_new_img, mask = mask_b)
    
#     rgb_img = cv2.merge((b,g,r))
#     return rgb_img

def get_dark_channel_and_mask(r, g, b):
    min_1 = cv2.min(r,g)

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
    
    dc = cv2.min(cv2.min(r,g),b);
    
    # plt.imshow(dc, cmap = 'gray')
    # plt.show()
    
    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(w,w))
    #dark = cv2.erode(dc,kernel)
    return dc, mask_r, mask_g, mask_b

def get_y_channel(I):
    y,u,v = cv2.split(I)
    return y

def get_uv_channel(I):
    y,u,v = cv2.split(I)
    return cv2.merge((u,v))

def estimate_atmosphere(im,dark):
    im = im.cpu().numpy()
    [h,w] = [128, 128]
    imsz = h*w
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

def generate_transmission(depth_map, beta):
    return np.exp(-beta * depth_map).astype(float)

# Estimates transmission map given a depth image
def estimate_transmission_depth(hazy_img, depth_map):
    plt.imshow(hazy_img)
    plt.show()

    depth_map = np.ones_like(depth_map)
    T = estimate_transmission(depth_map, 5.0) #5.0 seems okay for synth hazy data. but real-world has 0.1 - 1.8 range only.
    #plt.imshow(T)
    #plt.show()

    A = 1 - T
    #plt.imshow(A)
    #plt.show()

    # clear_img = np.zeros_like(hazy_img)
    # clear_img[:,:,0] = (hazy_img[:,:,0] - A)/ T
    # clear_img[:, :, 1] = (hazy_img[:,:,1] - A)/ T
    # clear_img[:, :, 2] = (hazy_img[:,:, 2] - A)/ T

    #compute clear image with radiance term
    clear_img = np.zeros_like(hazy_img)
    clear_img[:, :, 0] = ((hazy_img[:, :, 0] - A) / np.maximum(T, 0.75)) + A
    clear_img[:, :, 1] = ((hazy_img[:, :, 1] - A) / np.maximum(T, 0.75)) + A
    clear_img[:, :, 2] = ((hazy_img[:, :, 2] - A) / np.maximum(T, 0.75)) + A

    plt.imshow(clear_img)
    plt.show()

def estimate_transmission(im, A, dark_channel):
    im = im.cpu().numpy()
    
    omega = 0.95;
    im3 = np.empty(im.shape,im.dtype);

    for ind in range(0,3):
        im3[:,:,ind] = im[:,:,ind]/A[0,ind]

    transmission = 1 - omega * dark_channel
    return transmission
    
def make_rgb(batch):
    batch = batch.transpose(0, 1)
    (b, g, r) = torch.chunk(batch, 3)
    batch = torch.cat((r, g, b))
    batch = batch.transpose(0, 1)
    return batch

def subtract_imagenet_mean_batch(batch):
    """Subtract ImageNet mean pixel-wise from a BGR image."""
    tensortype = type(batch.data)
    mean = tensortype(batch.data.size())
    mean[:, 0, :, :] = 103.939
    mean[:, 1, :, :] = 116.779
    mean[:, 2, :, :] = 123.680
    return batch - Variable(mean).cuda()


def add_imagenet_mean_batch(batch):
    """Add ImageNet mean pixel-wise from a BGR image."""
    tensortype = type(batch.data)
    mean = tensortype(batch.data.size())
    mean[:, 0, :, :] = 103.939
    mean[:, 1, :, :] = 116.779
    mean[:, 2, :, :] = 123.680
    return batch + Variable(mean).cuda()

def imagenet_clamp_batch(batch, low, high):
    batch[:,0,:,:].data.clamp_(low-103.939, high-103.939)
    batch[:,1,:,:].data.clamp_(low-116.779, high-116.779)
    batch[:,2,:,:].data.clamp_(low-123.680, high-123.680)

#computes a z_signal based on image size. Image size must always be a power of 2 and greater than 16x16.
def compute_z_signal(value, batch_size, image_size):
    z_size = (int(image_size[0] / 16), int(image_size[1] / 16))
    torch.manual_seed(value)
    z_signal = torch.randn((batch_size, 100, z_size[0], z_size[1]))
    return z_signal

#computes a z signal to be conacated with another image tensor.
def compute_z_signal_concat(value, batch_size, image_size):
    torch.manual_seed(value)
    z_signal = torch.randn((batch_size, 100, image_size[0], image_size[1]))
    return z_signal

class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)
    
