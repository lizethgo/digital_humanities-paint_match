# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 12:11:10 2023
@author: 888254
Description : SIFT algorithm applied to a dataset of paintings
to cmpare similarity points
"""

import cv2 
import matplotlib.pyplot as plt
import  numpy as np
from PIL import Image  
import PIL 


#You can add more images into the following list
dataset = ['Edinburgh_Nat_Gallery.jpg',
           'London_Nat_Gallery.jpg',
           'London_OrderStJohn.jpg',
           'Milan_private.jpg',
           'Naples_Museo_Capodimonte.jpg',
           'Nolay_MaisonRetraite.jpg',
           'Oxford_Ashmolean.jpg',
           'Oxford_Christ_Church.jpg',
           'UK_Nostrell_Priory.jpg',
           'UK_Warrington_Museum.jpg',
           'Zurich_KunyCollection.jpg']

#create a result matrix, this matrix contains the matches of all paintings
#compared with each other
result_m = np.zeros((len(dataset), len(dataset)))


#iterate over the existing dataset
for i, num in enumerate(dataset):
    for j, num in enumerate(dataset):
        img1 = cv2.imread(dataset[i]) # initial baseline  
        img2 = cv2.imread(dataset[j])
    
        # change to B&W
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
        #sift
        sift = cv2.SIFT_create()
    
        keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)
        keypoints_2, descriptors_2 = sift.detectAndCompute(img2,None)
        
        #feature matching
        #bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        
        #matches = bf.match(descriptors_1,descriptors_2)
        #matches = sorted(matches, key = lambda x:x.distance)
        #img3 = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, matches[:150], img2, flags=2)
        #img3 = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, matches[:], img2, flags=2)
        #plt.imshow(img3),plt.show() 
        
        # feature matching - default parameters
        bf = cv2.BFMatcher() 
        # finding matches from BFMatcher() 
        matches = bf.knnMatch(descriptors_1,descriptors_2, k=2) 
        # stored 'good' matches
        good = [] 
        
        for m, n in matches:  
            if m.distance < 0.8 * n.distance: 
                good.append([m]) 

        img3 = cv2.drawMatchesKnn(img1,  
                   keypoints_1, img2, keypoints_2, good, None, 
                   matchColor=(0, 255, 0), matchesMask=None, 
                   singlePointColor=(255, 0, 0), flags=0)
        
        if(False) : #True : print images, False otherwise
            plt.imshow(img3)
            plt.show() 
        if(True) :
            img_3 = Image.fromarray(img3)
            img_3.save(dataset[i]+"_VS_"+dataset[j])

             
        #img_1 = cv2.drawKeypoints(img1,keypoints_1,img1)
        #plt.imshow(img_1)
        
        result_m[i,j] = len(good)
        np.savetxt("comparisson_matrix.csv", result_m, delimiter=",")
