import os

import cv2
from src.commons.utils import PATH,mp_holistic,mediapipe_detection,draw_styled_landmarks,extract_keypoints
import pyzed.sl as sl
import numpy as np
import sys
import mediapipe as mp

from pathlib import Path as Pt #Avoid confusion with class path

category = "fouls"
sequence_length = 30
reinitialize_data = True

PATH.SAMPLES = os.path.join(Pt(__file__).parent.parent.parent, "samples")
PATH.DATA = os.path.join(Pt(__file__).parent.parent.parent, "data")

# List of directories contained into PATH.SAMPLES+category
actions = os.listdir(os.path.join(PATH.SAMPLES, category))

'''if not os.path.exists(PATH.DATA):
    os.makedirs(PATH.DATA)

if not os.path.exists(os.path.join(PATH.DATA, category)):
    os.makedirs(os.path.join(PATH.DATA, category))
elif reinitialize_data:
    for file in os.listdir(os.path.join(PATH.DATA, category)):
        os.remove(os.path.join(PATH.DATA, category, file))'''

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for action in actions:
        
        if not os.path.exists(os.path.join(PATH.DATA, category, action)):
            os.makedirs(os.path.join(PATH.DATA, category, action))
        
        # List of svo files contained in action
        files = [os.path.join(PATH.SAMPLES, category, action, file) for file in os.listdir(os.path.join(PATH.SAMPLES, category, action)) if file.endswith('.mp4')]

        for file in files:
            file_name = file.split('\\')[-1].split('.')[0]
            if not os.path.exists(os.path.join(PATH.DATA, category, action, file_name)):
                os.makedirs(os.path.join(PATH.DATA, category, action, file_name))
            
            print(f'Collecting data for class {action} - file {file}')

            # analyze video with zed until it is finished
            cap = cv2.VideoCapture(file)
            for frame_num in range(sequence_length):
                ret, frame = cap.read()

                if ret:
                    # Make detections
                    image, results = mediapipe_detection(frame, holistic)

                    # Draw landmarks
                    draw_styled_landmarks(image, results)
                    
                    # Save keypoints
                    keypoints = extract_keypoints(results)
                    npy_path = os.path.join(PATH.DATA, category, action, file_name, str(frame_num))
                    np.save(npy_path, keypoints)
                    cv2.waitKey(10)
                else:
                    break
            
            cap.release()
            cv2.destroyAllWindows()
            print("Finished")


# Try to train with spatial temporal positional coding