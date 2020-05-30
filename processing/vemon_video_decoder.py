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
import ipm

PATH = "E:/VEMON Dataset/videos/"
SAVE_PATH = "E:/VEMON Dataset/pending/frames/"
HOMOG_PATH = "E:/VEMON Dataset/pending/homog_frames/"
HOMOG_CROP_PATH = "E:/VEMON Dataset/pending/homog_crop_frames/"
TOPDOWN_PATH = "E:/VEMON Dataset/pending/topdown_frames/"
  
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
              
              result_img = warp_synth_view(image)
              crop_img = polish_border(result_img)
        
              #cv2.imwrite(HOMOG_PATH +"frame_%d.png" % count, result_img)
              #cv2.imwrite(HOMOG_CROP_PATH +"frame_%d.png" % count, crop_img)
              count += 1

def perform_synth():
    normal_list = dataset_loader.assemble_normal_data(-1)
    topdown_list = dataset_loader.assemble_topdown_data(-1)
    
    count = 0
    for img_path, topdown_path in zip(normal_list, topdown_list):
        image = cv2.imread(img_path)
        topdown_image = cv2.imread(topdown_path)
        
        h,w,c = np.shape(image)
        image = image[117:h, 0:w]
        
        result_1, result_2 = warp_synth_view(image)
        image = cv2.resize(image, (int(w/4), int(h/4)), interpolation = cv2.INTER_LINEAR)
        result_img = cv2.resize(result_1, (int(w/4), int(h/4)), interpolation = cv2.INTER_LINEAR)
        
        crop_img = polish_synth_border(result_img)
        plt.imshow(image)
        plt.show()
        plt.imshow(result_1)
        plt.show()
        plt.imshow(result_2)
        plt.show()
        break
        # cv2.imwrite(SAVE_PATH + "frame_%d.png" % count, image)
        # cv2.imwrite(TOPDOWN_PATH + "frame_%d.png" % count, topdown_image)
        # cv2.imwrite(HOMOG_PATH +"frame_%d.png" % count, result_img)
        # cv2.imwrite(HOMOG_CROP_PATH +"frame_%d.png" % count, crop_img)
        # print("Saved: frame_%d.png" % count)
        count += 1

def perform_superres():
    normal_list = dataset_loader.assemble_vemon_style_data(-1)
    
    for i in range(10, len(normal_list)):
        img_list = []
        for j in range(10):
            img = cv2.imread(normal_list[j + i])
            img_list.append(img)
        
        result = align_merge(img_list)
        fig, ax = plt.subplots(2, 1)
        fig.set_size_inches(18, 7)
        fig.tight_layout()
        ax[0].imshow(img_list[0])
        ax[1].imshow(result)
        img_list.clear()
        plt.show()
    
#aligns and combines multiple frames
def align_merge(img_list):

    base_img = img_list[0]
    width,height,_ = np.shape(base_img)
    for i in range(1,len(img_list)):
        im1Gray = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
        im2Gray = cv2.cvtColor(img_list[i], cv2.COLOR_BGR2GRAY)
    
        # Detect ORB features and compute descriptors.
        orb = cv2.ORB_create(700)
        keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
        keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None) 
        
        # Match features.
        matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
        matches = matcher.match(descriptors1, descriptors2, None)
      
        # Sort matches by score
        matches.sort(key=lambda x: x.distance, reverse=False)
        
        #remove not so good matches
        good_match_threshold = 0.20
        numGoodMatches = int(len(matches) * good_match_threshold)
        matches = matches[:numGoodMatches]
  
        # Draw top matches
        im_matches = cv2.drawMatches(base_img, keypoints1, img_list[i], keypoints2, matches, None)
      
        # Extract location of good matches
        points1 = np.zeros((len(im_matches), 2), dtype=np.float32)
        points2 = np.zeros((len(im_matches), 2), dtype=np.float32)
     
        for k, match in enumerate(matches):
            points1[k, :] = keypoints1[match.queryIdx].pt
            points2[k, :] = keypoints2[match.trainIdx].pt
       
        # Find homography
        h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
        #print("Resulting H:" , h, np.shape(h), "Key points shape: ", np.shape(keypoints1), np.shape(keypoints2))
        # Use homography
        height, width, channels = base_img.shape
        if(np.shape(h) == (3,3)):
            result_img = cv2.warpPerspective(img_list[i], h, (width, height),
                                             borderValue = (255,255,255))
        else:
            print("H is not 3x3!")
            h = np.ones((3,3))
            result_img = img_list[i]
        
        img_list[i] = result_img
    
    #combine
    result_img = base_img.astype(np.uint16)
    print("Base mean: ", np.mean(result_img))
    for i in range(1,len(img_list)):
        result_img = result_img + img_list[i]
        #plt.imshow(result_img)
        #plt.show()
    
    result_img = result_img / (1.0 * len(img_list))
    result_img = result_img.astype(np.uint8)                               
    return result_img    
        
def enhance_image(img):
    src = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dst = cv2.equalizeHist(src)
    
    src = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    src[:,:,0] = (src[:,:,0] * 0.4) + (dst * 0.6)
    
    enhance_img = cv2.cvtColor(src, cv2.COLOR_YUV2BGR)
    h,w,c = np.shape(enhance_img)
    enhance_img = enhance_img[27:463, 0:w]
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
        
#for synthethic images only
def warp_synth_view(img):
    
    x_dim = np.shape(img)[1]; y_dim = np.shape(img)[0] - 20;
    
    upper_start = ([-2000, 0], [3920, 0])
    lower_start = ([0, y_dim], [1920, y_dim])
    
    pts1 = np.float32([[0,y_dim],[x_dim,y_dim],upper_start[0], upper_start[1]])
    pts2 = np.float32([lower_start[0],lower_start[1],[0,0], [x_dim, 0]])
    M1 = ipm.calculateHMatrix(pts1, pts2)
    M2 = cv2.getPerspectiveTransform(pts1, pts2)
    
    result_1 = cv2.warpPerspective(img, M1, (x_dim, y_dim))
    result_2 = cv2.warpPerspective(img, M2, (x_dim, y_dim))
    return result_1, result_2

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
    
    upper_start = ([0, 0], [x_dim, 0])
    lower_start = ([0, y_dim], [x_dim, y_dim])
    
    for i in range(50):
        pts1 = np.float32([[0,y_dim],[x_dim,y_dim],upper_start[0], upper_start[1]])
        pts2 = np.float32([lower_start[0],lower_start[1],[0,0], [x_dim, 0]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        result = cv2.warpPerspective(img, M, (x_dim, y_dim))
        plt.imshow(result)
        plt.show()
        
        upper_start[0][0] = upper_start[0][0] - 40
        upper_start[1][0] = upper_start[1][0] + 40
        
        #lower_start[0][0] = lower_start[0][0] - 10
        #lower_start[1][0] = lower_start[1][0] + 10
        
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

def polish_synth_border(warp_img):
   h,w,c = np.shape(warp_img)
    
   box = [0,h,170,320]
   crop = warp_img[box[0]: box[1], box[2]: box[3]]
   return crop 
    
def main():
    #perform()
    #perform_synth()
    perform_superres()

#FIX for broken pipe num_workers issue.
if __name__=="__main__": 
    main()   