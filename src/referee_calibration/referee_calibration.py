import cv2
import mediapipe as mp
import numpy as np
import math

def referee_calibration(frame, hands) -> list:
    cal_small = []
    cal_big = []
    frame_rbg = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rbg)
    if results.multi_hand_landmarks:
        if len(results.multi_hand_landmarks)==2:
            for hand_landmarks in results.multi_hand_landmarks:
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
    if(len(cal_small)!=0 and len(cal_big)!=0 and len(cal_big)==len(cal_small)):
        scaling = []
        for i in range(len(cal_big)):
            scaling.append(cal_big[i]/cal_small[i])
        return scaling
    return None