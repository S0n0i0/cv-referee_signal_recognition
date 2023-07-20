import os
import pickle
import numpy as np
import math

import mediapipe as mp
import cv2
import matplotlib.pyplot as plt
from src.commons.utils import PATH

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode = True, min_detection_confidence = 0.3)

data = []
labels = []
#code if normalization is needed
cal_small = []
cal_big = []
img_cal = cv2.imread(os.path.join(PATH.DATA, "hands", '11','0.jpg'))
img_cal_rgb = cv2.cvtColor(img_cal, cv2.COLOR_BGR2RGB)
results_cal = hands.process(img_cal_rgb)
if results_cal.multi_hand_landmarks:
    if(len(results_cal.multi_hand_landmarks)==2):
        for hand_landmarks in results_cal.multi_hand_landmarks:
            cs = 0
            cb = 0
            x0 = hand_landmarks.landmark[0].x
            y0 = hand_landmarks.landmark[0].y
            x9 = hand_landmarks.landmark[9].x
            y9 = hand_landmarks.landmark[9].y
            x12 = hand_landmarks.landmark[12].x
            y12 = hand_landmarks.landmark[12].y
                    
            cs = math.sqrt(math.pow(x9-x0,2)+math.pow(y9-y0,2))
            cb = math.sqrt(math.pow(x12-x0,2)+math.pow(y12-y0,2))
            cal_big.append(cb)
            cal_small.append(cs)

for dir_ in os.listdir(os.path.join(PATH.DATA, "penalties")):
    for img_path in os.listdir(os.path.join(PATH.DATA, "penalties", dir_)):
        data_aux = []
        img = cv2.imread(os.path.join(PATH.DATA, "penalties", dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            if(len(results.multi_hand_landmarks)==2):
                '''for i in range(len(results.multi_hand_landmarks)):
                    x0 = results.multi_hand_landmarks[i].landmark[0].x
                    y0 = results.multi_hand_landmarks[i].landmark[0].y
                    x1 = results.multi_hand_landmarks[i].landmark[1].x
                    y1 = results.multi_hand_landmarks[i].landmark[1].y
                    for j in [4,8,12,16,20]:
                        x = results.multi_hand_landmarks[i].landmark[j].x
                        y = results.multi_hand_landmarks[i].landmark[j].y
                        dx = 0
                        dy = 0
                        if(j==4):
                            dx = (x - x1)
                            dy = (y - y1)
                        else:
                            dx = (x - x0)
                            dy = (y - y0)
                        data_aux.append(dx)
                        data_aux.append(dy)'''
                for i in range(len(results.multi_hand_landmarks)):
                    x0 = results.multi_hand_landmarks[i].landmark[0].x
                    y0 = results.multi_hand_landmarks[i].landmark[0].y
                    x1 = results.multi_hand_landmarks[i].landmark[1].x
                    y1 = results.multi_hand_landmarks[i].landmark[1].y
                    x9 = results.multi_hand_landmarks[i].landmark[9].x
                    y9 = results.multi_hand_landmarks[i].landmark[9].y
                    small_dist = math.sqrt(math.pow(x9-x0,2)+math.pow(y9-y0,2))
                    norm_factor = small_dist * cal_big[i] / cal_small[i]
                    for j in [4,8,12,16,20]:
                        x = results.multi_hand_landmarks[i].landmark[j].x
                        y = results.multi_hand_landmarks[i].landmark[j].y
                        dx = 0
                        dy = 0
                        if(j==4):
                            dx = (x - x1)/norm_factor
                            dy = (y - y1)/norm_factor
                        else:
                            dx = (x - x0)/norm_factor
                            dy = (y - y0)/norm_factor
                        data_aux.append(dx)
                        data_aux.append(dy)
                '''for hand_landmarks in results.multi_hand_landmarks:

                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x)
                        data_aux.append(y)
                '''  
                data.append(data_aux)
                #print(data)
                labels.append(dir_)

f = open(PATH.DATA_FILES.format("penalty_calibrated"), 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()