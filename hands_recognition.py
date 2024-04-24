# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 11:53:53 2024

@author: 888254
"""

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
import mediapipe as mp
import os
import math


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

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize MediaPipe Drawing module for drawing landmarks
mp_drawing = mp.solutions.drawing_utils

landmarks = []
connections = []
landmark_array = np.zeros(shape=(len(dataset), 21, 2), dtype=float)
landmark_array_r = np.zeros(shape=(len(dataset), 21, 2), dtype=float)

for i, num in enumerate(dataset):
    # Image directory 
    directory = 'C:/Users/888254/Documents/research/art_history/comparisson'
    # Change the current directory  
    # to specified directory  
    os.chdir(directory) 
    img = cv2.imread(dataset[i])
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    results = hands.process(img_rgb)
    
    # Check if hands are detected
    
    image_height, image_width, _ = img_rgb.shape
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            
            # Draw landmarks on the frame
            # Here is How to Get All the Coordinates
            for ids, landmrk in enumerate(hand_landmarks.landmark):
 
                cx_r, cy_r = landmrk.x * image_width, landmrk.y*image_height
                cx, cy = landmrk.x, landmrk.y
                landmark_array[i, ids, 0] = cx
                landmark_array[i, ids, 1] = cy
                
                landmark_array_r[i, ids, 0] = cx_r
                landmark_array_r[i, ids, 1] = cy_r

    
            mp_drawing.draw_landmarks(img_rgb, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            connections.append(list(mp_hands.HAND_CONNECTIONS))
            
        
 

    
    # Display the frame with hand landmarks
    plt.imshow(img_rgb)
    plt.show()
    
    # Image directory 
    directory = 'C:/Users/888254/Documents/research/art_history/comparisson/hands'
    # Change the current directory  
    # to specified directory  
    os.chdir(directory)  
    print("Before saving image:")   
    print(os.listdir(directory))   
      
    # Filename 
    filename = '{}_hand.jpg'.format(dataset[i])
    cv2.imwrite(filename, img_rgb) 


#extract numerical information from landmarks (coordinates)
connections_arr = np.array(connections)
lengths = np.zeros((len(dataset), len(connections_arr[0])))


for i in range(len(dataset)):
    for j in range(len(connections_arr[0])): # number of connections
        x1 = landmark_array[i][connections_arr[i][j][0]][0]
        y1 = landmark_array[i][connections_arr[i][j][0]][1]
        
        x2 = landmark_array[i][connections_arr[i][j][1]][0]
        y2 = landmark_array[i][connections_arr[i][j][1]][1]
        
        lengths[i,j] = math.sqrt(pow((x2-x1),2) + pow((y2-y1),2) )
#plot some sample landmarks

# Create a black image
img = np.zeros((200,200,3), np.uint8)
# Draw a diagonal blue line with thickness of 5 px

for j, num in enumerate(dataset):
    img = np.zeros((200,200,3), np.uint8)
    print('DEBUG', j)
    for i in range(21):
        x1 = int(landmark_array_r[j][connections_arr[j][i][0]][0])-200
        y1 = int(landmark_array_r[j][connections_arr[j][i][0]][1])-300
            
        x2 = int(landmark_array_r[j][connections_arr[j][i][1]][0])-200
        y2 = int(landmark_array_r[j][connections_arr[j][i][1]][1])-300
             
        cv2.line(img,(x1,y1),(x2,y2),(255,0,0),1)
        cv2.circle(img,(int(landmark_array_r[j][i][0]-200),int(landmark_array_r[j][i][1]-300)), 3, (0,0,255), -1)
    plt.title(dataset[j])
    plt.imshow(img)
    plt.show()

# compute differences
differences = np.zeros((len(dataset), len(dataset), 21))
score = np.zeros((len(dataset), len(dataset)))

        
for i in range(len(dataset)):
    
    for j in range(len(dataset)):
        score_ = 0
        for k in range(21):
            differences[i,j,k] =  lengths[i,k] - lengths[j,k]
            score_ = score_ + abs(differences[i,j,k])
        score[i,j] = score_
        

## writing algorithm for comparisson
# https://developers.google.com/mediapipe/solutions/vision/hand_landmarker




import seaborn as sns
import pandas as pd
import numpy as np
 
# Create a dataset (just as an example)
df = pd.DataFrame(abs(differences[0])*100, index=dataset)

# plot a heatmap with annotation
ax = plt.axes()
sns.heatmap(df, annot=True, annot_kws={"size": 5})
ax.set_title('{} vs all'.format(dataset[0]))
plt.show()


# Create a dataset (just as an example)
df = pd.DataFrame(score*100, index=dataset, columns = dataset)

# plot a heatmap with annotation
ax = plt.axes()
sns.heatmap(df, annot=True, annot_kws={"size": 5})
ax.set_title('{} vs all'.format(dataset[0]))
plt.show()



