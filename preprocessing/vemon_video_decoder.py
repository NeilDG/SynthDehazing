# -*- coding: utf-8 -*-
"""
Produces frame images from VEMON videos
Created on Wed May  6 09:48:45 2020

@author: delgallegon
"""

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from loaders import dataset_loader

PATH = "E:/VEMON Dataset/videos/"
SAVE_PATH = "E:/VEMON Dataset/frames/"
HOMOG_PATH = "E:/VEMON Dataset/homog_frames/"
HOMOG_CROP_PATH = "E:/VEMON Dataset/homog_crop_frames/"
    
def perform():
    
    videos = os.listdir(PATH)
    
    count = 0
    for i in range(len(videos)):
        video_path = PATH + videos[i]
        video_name = videos[i].split(".")[0]
        vidcap = cv2.VideoCapture(video_path)
        success,image = vidcap.read()
        success = True
        while success:
          success,image = vidcap.read()
          if(success):
              image = enhance_image(image)
              
              cv2.imwrite(SAVE_PATH + "frame_%d.png" % count, image)
              print("Saved: " + video_name + "_frame_%d.png" % count)
              
              result_img = warp_bird_view(image)
              crop_img = polish_border(result_img)
        
              cv2.imwrite(HOMOG_PATH +"frame_%d.png" % count, result_img)
              cv2.imwrite(HOMOG_CROP_PATH +"frame_%d.png" % count, crop_img)
              count += 1

def enhance_image(img):
    src = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dst = cv2.equalizeHist(src)
    
    src = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    src[:,:,0] = (src[:,:,0] * 0.4) + (dst * 0.6)
    
    enhance_img = cv2.cvtColor(src, cv2.COLOR_YUV2BGR)
    
    return enhance_img

def warp_batch():
    img_list = dataset_loader.assemble_test_data()
    
    for i in range(len(img_list)):
        result_img = warp_bird_view(img_list[i])
        crop_img = polish_border(result_img)
        
        plt.imshow(result_img)
        plt.show()
    
        #plt.imshow(crop_img)
        #plt.show()
        
        cv2.imwrite(HOMOG_PATH +"homog_%d.png" % i, result_img)
        cv2.imwrite(HOMOG_CROP_PATH +"crop_%d.png" % i, crop_img)
        

def warp_bird_view(img):
    
    x_dim = np.shape(img)[1]; y_dim = np.shape(img)[0] - 20;
    
    upper_start = ([-15, 79], [697, 79])
    lower_start = ([239, y_dim], [461, y_dim])
    
    
    pts1 = np.float32([[0,y_dim],[x_dim,y_dim],upper_start[0], upper_start[1]])
    pts2 = np.float32([lower_start[0],lower_start[1],[0,0], [x_dim, 0]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(img, M, (x_dim, y_dim))
    
    return result
    
def validate_bird_view(image_path):
    img = cv2.imread(image_path)
    plt.imshow(img)
    plt.show()
    
    x_dim = np.shape(img)[1]; y_dim = np.shape(img)[0] - 20;
    
    upper_start = ([297, 79], [385, 79])
    lower_start = ([200, y_dim], [500, y_dim])
    
    for i in range(13):
        pts1 = np.float32([[0,y_dim],[x_dim,y_dim],upper_start[0], upper_start[1]])
        pts2 = np.float32([lower_start[0],lower_start[1],[0,0], [x_dim, 0]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        result = cv2.warpPerspective(img, M, (x_dim, y_dim))
        plt.imshow(result)
        plt.show()
        
        upper_start[0][0] = upper_start[0][0] - 24
        upper_start[1][0] = upper_start[1][0] + 24
        
        lower_start[0][0] = lower_start[0][0] + 3
        lower_start[1][0] = lower_start[1][0] - 3
        
        print("I: ", i, " : ", upper_start, lower_start)
    
    #best values identified
    #current upper_start = [-15, 79], [697, 79]
    #current lower_start = [239,470], [461, 470]
    return result
    

"""
Polishes the image by further removing the border via non-zero checking
"""
def polish_border(warp_img, zero_threshold = 100, cut = 10): 
    h,w,c = np.shape(warp_img)
    
    box = [0,h,260,440]
    crop = warp_img[box[0]: box[1], box[2]: box[3]]
    return crop
    
    
def main():
    perform()
   #warp_batch()

#FIX for broken pipe num_workers issue.
if __name__=="__main__": 
    main()   