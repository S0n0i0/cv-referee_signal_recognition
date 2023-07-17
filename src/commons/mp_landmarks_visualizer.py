import cv2
import mediapipe as mp
import pickle
import numpy as np
from src.commons.data_structures import PATH
from enum import Enum

def hand_gesture_recognition(frame, hands, mp_hands, mp_drawing, mp_drawing_styles) -> str:
    frame_rbg = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rbg)
    if results.multi_hand_landmarks:
        if len(results.multi_hand_landmarks)==2:
            data_aux = []
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                        frame, # image to draw
                        hand_landmarks, # model output
                        mp_hands.HAND_CONNECTIONS, # hand connections
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
            
            return None
    return None

if __name__ == "__main__":
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(static_image_mode = True, min_detection_confidence = 0.4)

    cap = cv2.VideoCapture("samples\\numbers\\1-p.avi")
    last_predicted_num = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Oh no")
            break
        
        predicted_number = hand_gesture_recognition(frame, hands, mp_hands, mp_drawing, mp_drawing_styles)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()