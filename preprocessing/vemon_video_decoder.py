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
def perform():
    PATH = "E:/VEMON Dataset/videos/"
    SAVE_PATH = "E:/VEMON Dataset/frames/"
    
    videos = os.listdir(PATH)
    for i in range(len(videos)):
        video_path = PATH + videos[i]
        video_name = videos[i].split(".")[0]
        vidcap = cv2.VideoCapture(video_path)
        success,image = vidcap.read()
        count = 0
        success = True
        while success:
          success,image = vidcap.read()
          if(success):
              cv2.imwrite(SAVE_PATH + video_name + "_frame_%d.png" % count, image)
              print("Saved: " + video_name + "_frame_%d.png" % count)
              count += 1

def warp_batch():
    img_list = dataset_loader.assemble_test_data()
    
    for img_path in img_list:
        warp_to_bird_view(img_path)
        break
    
def warp_to_bird_view(image_path):
    img = cv2.imread(image_path)
    plt.imshow(img)
    plt.show()
    
    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # gray = np.float32(gray)

    # corners = cv2.cornerHarris(gray, 2, 3, 0.04)
    
    # #result is dilated for marking the corners, not important
    # corners = cv2.dilate(corners,None)
    # corner_img = img
    # # Threshold for an optimal value, it may vary depending on the image.
    # corner_img[corners>0.01*corners.max()]=[0,0,255]
    
    # plt.imshow(corner_img)
    # plt.show()

    x_dim = np.shape(img)[1]; y_dim = np.shape(img)[0];
    
    upper_start = ([297, 79], [385, 79])
    lower_start = ([200, y_dim], [500, y_dim])
    
    for i in range(30):
        pts1 = np.float32([[0,y_dim],[x_dim,y_dim],upper_start[0], upper_start[1]])
        pts2 = np.float32([lower_start[0],lower_start[1],[0,0], [x_dim, 0]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        result = cv2.warpPerspective(img, M, (x_dim, y_dim))
        plt.imshow(result)
        plt.show()
        
        upper_start[0][0] = upper_start[0][0] - 12
        upper_start[1][0] = upper_start[1][0] + 12
        
        lower_start[0][0] = lower_start[0][0] + 3
        lower_start[1][0] = lower_start[1][0] - 3
        
        print("I: ", i, " : ", lower_start)
    
def main():
    warp_batch()

#FIX for broken pipe num_workers issue.
if __name__=="__main__": 
    main()   