import cv2
import mediapipe as mp
import pickle
import numpy as np
from src.commons.utils import PATH

model_dict = pickle.load(open(PATH.MODELS.format("model"), 'rb'))
model = model_dict["model"]

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode = True, min_detection_confidence = 0.4)
labels_dict = {0: '0', 1: '00', 2: '1', 3: '2', 4: '3', 5: '4', 6: '5', 7: '6', 8: '7', 9: '8', 10: '9', 11: '10', 12: '11', 13: '12', 14: '13', 15: '14', 16: '15'}

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
                        x_.append(x)
                        y_.append(y)

            x1 = int(min(x_) * W)
            y1 = int(min(y_) * H)

            x2 = int(max(x_) * W)
            y2 = int(max(y_) * H)
            
            prediction = model.predict([np.asarray(data_aux)])
            
            predicted_number = labels_dict[int(prediction[0])]

            if(last_predicted_num!=predicted_number):
                last_predicted_num = predicted_number
                print(last_predicted_num)

            cv2.rectangle(frame, (x1,y1),(x2,y2), (0,0,0), 4)
            cv2.putText(frame, predicted_number, (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,0,0),3, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    if cv2.waitKey(25) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()