import os
import cv2
from src.commons.utils import mediapipe_detection,draw_styled_landmarks,extract_keypoints
from src.commons.data_structures import PATH
import numpy as np
import mediapipe as mp
import threading

def get_resized_dims(img,scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    return (width, height)

def analyze_frame(frame,analyzed_frame,frame_num,holistic,mp_holistic,mp_drawing,category,action,file_name):
    # Make detections
    image, results = mediapipe_detection(frame, holistic)

    # Draw landmarks
    draw_styled_landmarks(image, results, mp_holistic, mp_drawing)
    
    # Save keypoints
    keypoints = extract_keypoints(results)
    npy_path = os.path.join(PATH.DATA, category, action, f"{file_name}_{analyzed_frame}", str(frame_num))
    np.save(npy_path, keypoints)

category = "fouls"
sequence_length = 30
reinitialize_data = True

frame_alternatives = ["original","flipped","shrinked","enlarged"]

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# List of directories contained into PATH.SAMPLES+category
actions = os.listdir(os.path.join(PATH.SAMPLES, category))

if not os.path.exists(PATH.DATA):
    os.makedirs(PATH.DATA)

if not os.path.exists(os.path.join(PATH.DATA, category)):
    os.makedirs(os.path.join(PATH.DATA, category))
elif reinitialize_data:
    for action in os.listdir(os.path.join(PATH.DATA, category)):
        for samples in os.listdir(os.path.join(PATH.DATA, category, action)):
            data_file_path = os.path.join(PATH.DATA, category, action, samples)
            for file in os.listdir(data_file_path):
                os.remove(os.path.join(data_file_path,file))
            os.rmdir(data_file_path)
        os.rmdir(os.path.join(PATH.DATA, category, action))

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for action in actions:
        
        if not os.path.exists(os.path.join(PATH.DATA, category, action)):
            os.makedirs(os.path.join(PATH.DATA, category, action))
        
        # List of svo files contained in action
        files = [os.path.join(PATH.SAMPLES, category, action, file) for file in os.listdir(os.path.join(PATH.SAMPLES, category, action)) if file.endswith('.mp4')]

        threads = []
        for file in files:
            file_name = file.split('\\')[-1].split('.')[0]
            for alternative in frame_alternatives:
                if not os.path.exists(os.path.join(PATH.DATA, category, action, f"{file_name}_{alternative}")):
                    os.makedirs(os.path.join(PATH.DATA, category, action, f"{file_name}_{alternative}"))
            
            print(f'Collecting data for class {action} - file {file}')

            # analyze video with zed until it is finished
            cap = cv2.VideoCapture(file)
            for frame_num in range(sequence_length):
                ret, frame = cap.read()

                if ret:
                    # Make detections
                    image, results = mediapipe_detection(frame, holistic)

                    # Draw landmarks
                    draw_styled_landmarks(image, results, mp_holistic, mp_drawing)

                    frames_to_analyze = {
                        "original": frame,
                        "flipped": cv2.flip(frame,1),
                        "shrinked": cv2.resize(frame, get_resized_dims(frame,90), interpolation = cv2.INTER_AREA),
                        "enlarged": cv2.resize(frame, get_resized_dims(frame,110), interpolation = cv2.INTER_AREA)
                    }
                    
                    for analyzed_frame in frames_to_analyze.keys():
                        thread = threading.Thread(target=analyze_frame,args=(frames_to_analyze[analyzed_frame],analyzed_frame,frame_num,holistic,mp_holistic,mp_drawing,category,action,file_name))
                        thread.start()
                        threads.append(thread)

                        cv2.waitKey(10)
                    
                    for thread in threads:
                        thread.join()
                else:
                    break
            
            cap.release()
            cv2.destroyAllWindows()
            print(f"Finished: {file}")
    
    print(f"{action} analyzed")