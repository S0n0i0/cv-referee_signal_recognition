import os
import pickle
import numpy as np

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
for dir_ in os.listdir(os.path.join(PATH.DATA, "penalties")):
    for img_path in os.listdir(os.path.join(PATH.DATA, "penalties", dir_)):
        data_aux = []
        img = cv2.imread(os.path.join(PATH.DATA, "penalties", dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            if(len(results.multi_hand_landmarks)==2):
                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x)
                        data_aux.append(y)
                        
                data.append(data_aux)
                labels.append(dir_)

f = open(PATH.DATA_FILES.format("penalty"), 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()