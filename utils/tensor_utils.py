# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 20:28:09 2019

Image and tensor utilities
@author: delgallegon
"""

import torch.nn as nn
import numpy as np
import math
import cv2
from torch.autograd import Variable
import torch
from utils import pytorch_colors

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

def merge_yuv_results_to_rgb(y_tensor, yuv_tensor):
    yuv_tensor = yuv_tensor.transpose(0, 1)
    y_tensor = y_tensor.transpose(0, 1)
    (y, u, v) = torch.chunk(yuv_tensor, 3)
    yuv_tensor = torch.cat((y_tensor, u, v))
    rgb_tensor = pytorch_colors.yuv_to_rgb(yuv_tensor.transpose(0, 1))
    return rgb_tensor

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
    
