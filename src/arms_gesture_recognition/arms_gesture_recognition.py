import cv2
import numpy as np
import os
import mediapipe as mp

from src.commons.utils import mediapipe_detection,draw_styled_landmarks,extract_keypoints
from src.commons.data_structures import PATH, portable_model, model_type

def recognize_arms_gesture(frame, holistic, mp_holistic, mp_drawing, dataset_size, model, fouls_controls):

    threshold = 0.5
    category = "fouls"
    category_path = os.path.join(PATH.DATA, category)
    actions = np.array(os.listdir(category_path))
    prediction = None

    # Make detections
    image, results = mediapipe_detection(frame, holistic)
    
    # Draw landmarks
    draw_styled_landmarks(frame, results, mp_holistic, mp_drawing)
    
    # 2. Prediction logic
    keypoints = extract_keypoints(results)

    fouls_controls["sequence"].append(keypoints)
    fouls_controls["sequence"] = fouls_controls["sequence"][-1*dataset_size:]
    
    if len(fouls_controls["sequence"]) == dataset_size:
        res = model.predict(np.expand_dims(fouls_controls["sequence"], axis=0), verbose = 0)
        fouls_controls["predictions"].append(np.argmax(res))
        
        #3. Viz logic
        if np.unique(fouls_controls["predictions"][-10:])[0]==np.argmax(res):
            if res[0][np.argmax(res)] > threshold: 
                if len(fouls_controls["sentence"]) > 0: 
                    if actions[np.argmax(res)] != fouls_controls["sentence"][-1]:
                        fouls_controls["sentence"].append(actions[np.argmax(res)])
                else:
                    fouls_controls["sentence"].append(actions[np.argmax(res)])

        if len(fouls_controls["sentence"]) >= 1:
            fouls_controls["sentence"] = fouls_controls["sentence"][-1:]
            prediction = fouls_controls["sentence"][0]

    return prediction

if __name__ == "__main__":
    # 1. New detection variables
    fouls_controls = {
        "sequence": [],
        "sentence": [],
        "predictions": []
    }

    category = "fouls"
    model_path = os.path.join(PATH.MODELS,"model_files")
    category_path = os.path.join(PATH.DATA, category)
    dataset_size = 30

    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic

    # load model_fouls.keras from model_pth
    model = portable_model(model_type.FOULS)
    actions = np.array(os.listdir(category_path))

    cap = cv2.VideoCapture(0)
    # Set mediapipe model 
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():

            # Read feed
            ret, frame = cap.read()

            if not ret:
                print("Oh no")
                break

            prediction = recognize_arms_gesture(frame,holistic,mp_holistic,mp_drawing,dataset_size,model.model,fouls_controls)
            print("Prediction: ",prediction)

            if(prediction!=None):
                cv2.putText(frame, prediction, (0,50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,255,0),3, cv2.LINE_AA)

            cv2.imshow('OpenCV Feed', frame)
            
            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()