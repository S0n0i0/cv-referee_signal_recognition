import cv2
import mediapipe as mp
import pickle
import numpy as np
from src.commons.utils import PATH

def hand_gesture_recognition(frame, hands, mp_hands, mp_drawing, mp_drawing_styles, model_dict, model_name = "number"):
    model = model_dict["model"]
    if(model_name == "number"):
        labels_dict = {0: '0', 1: '00', 2: '1', 3: '2', 4: '3', 5: '4', 6: '5', 7: '6', 8: '7', 9: '8', 10: '9', 11: '10', 12: '11', 13: '12', 14: '13', 15: '14', 16: '15'}
    elif(model_name == "penalty"):
        labels_dict = {0: 'One free shoot', 1: 'Two free shoot', 2: 'Three free shoot', 3: 'Left throw-in', 4: 'Right throw-in', 5: 'Left throw-in, barging', 6: 'Right throw-in, barging'}
    else:
        return -1

    frame_rbg = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rbg)
    if results.multi_hand_landmarks:
        if len(results.multi_hand_landmarks)==2:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                        frame, # image to draw
                        hand_landmarks, # model output
                        mp_hands.HAND_CONNECTIONS, # hand connections
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
            
            for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x)
                        data_aux.append(y)
            
            prediction = model.predict([np.asarray(data_aux)])
            
            predicted_number = labels_dict[int(prediction[0])]
            return predicted_number
    return None

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

model_dict_numbers = pickle.load(open(PATH.MODELS.format("number"), 'rb'))
model_dict_penalties = pickle.load(open(PATH.MODELS.format("penalty"), 'rb'))

hands = mp_hands.Hands(static_image_mode = True, min_detection_confidence = 0.4)

cap = cv2.VideoCapture(0)
last_predicted_num = None

while cap.isOpened():
    data_aux = []
    x_ = []
    y_ = []
    ret, frame = cap.read()
    if not ret:
        print("Oh no")
        break

    H, W, _ = frame.shape

    
    predicted_number = hand_gesture_recognition(frame, hands, mp_hands, mp_drawing, mp_drawing_styles, model_dict_penalties, "penalty")
    if(predicted_number==-1):
            print("Model not found. Please check you write the right model name")
            break
    if(predicted_number!=None):
        if(last_predicted_num!=predicted_number):
            last_predicted_num = predicted_number
            print(last_predicted_num)
        cv2.putText(frame, predicted_number, (0,50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,0,255),3, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    if cv2.waitKey(25) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()