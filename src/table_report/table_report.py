import cv2
import numpy as np
import mediapipe as mp
from src.hands_gesture_recognition.hand_gesture_recognition import hand_gesture_recognition, portable_model, model_type

class table_report_obj:
    number: any
    foul: any
    penalty: any

    def __init__(self) -> None:
        self.number = None
        self.foul = None
        self.penalty = None
    
def table_report(frame, hands, mp_hands, mp_drawing, mp_drawing_styles, table_rep: table_report_obj, cont) -> table_report_obj:
    pt_model: portable_model
    if table_rep.number == None:
        pt_model = portable_model(model_type.NUMBERS)
        num_pred = hand_gesture_recognition(frame, hands, mp_hands, mp_drawing, mp_drawing_styles, pt_model)
        if(num_pred == -1):
            print("Model not found. Please try again")
        elif(num_pred!=None):
            if(num_pred!=table_rep.number):
                table_rep.number = num_pred
    elif table_rep.foul == None:
        pass
    elif table_rep.penalty == None:
        pt_model = portable_model(model_type.PENALTY)
        hand_gesture_recognition(frame, hands, mp_hands, mp_drawing, mp_drawing_styles, pt_model)
    return table_rep

if __name__ == "__main__":
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(static_image_mode = True, min_detection_confidence = 0.4)

    cap = cv2.VideoCapture(0)
    table_rep: table_report_obj = table_report_obj()
    cont = 0
    stop_prediction = 300
    table_rep.foul = "Personal"
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Oh no")
            break
        
        if(cont<stop_prediction):
            table_rep = table_report(frame, hands, mp_hands, mp_drawing, mp_drawing_styles, table_rep, cont)
        else:
            cont = 0

        print(table_rep.number)
        print(table_rep.foul)
        print(table_rep.penalty)
        
        cv2.putText(frame, table_rep.number, (0,50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,0,255),3, cv2.LINE_AA)

        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break
        if cv2.waitKey(25) == ord('r'):
            table_rep = table_report_obj()
            table_rep.foul = "Personal"

    cap.release()
    cv2.destroyAllWindows()