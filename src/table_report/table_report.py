import cv2
import numpy as np
import mediapipe as mp
from src.hands_gesture_recognition.hand_gesture_recognition import hand_gesture_recognition, portable_model, model_type, hand_class_recognition
from src.referee_calibration.referee_calibration import referee_calibration

class table_report_obj:
    number: str
    foul: str
    penalty: str
    predictions: dict
    back_hand: bool

    def __init__(self) -> None:
        self.number = None
        self.foul = None
        self.penalty = None
        self.predictions = {}
        self.back_hand = False

    
def table_report(frame, hands, mp_hands, mp_drawing, mp_drawing_styles, table_rep: table_report_obj, cont_frame, hands_norm_factor) -> table_report_obj:
    pt_model: portable_model
    if(cont_frame==0 or cont_frame==101):
        table_rep.predictions.clear()
    if table_rep.number == None:
        table_rep.back_hand = table_rep.back_hand or hand_class_recognition(frame, hands, mp_hands)
        pt_model = portable_model(model_type.NUMBERS)
        num_pred = hand_gesture_recognition(frame, hands, mp_hands, mp_drawing, mp_drawing_styles, pt_model, hands_norm_factor)
        if cont_frame < 100:            
            if(num_pred == -1):
                print("Model not found. Please try again")
            elif(num_pred!=None):
                keys = table_rep.predictions.keys()
                if(num_pred in keys):
                    val = table_rep.predictions[num_pred]
                    table_rep.predictions[num_pred] = val+1
                else:
                    table_rep.predictions[num_pred] = 1
        if(cont_frame == 100):
            print("FINISH")
            if len(table_rep.predictions) == 0:
                return None
            table_rep.number = max(table_rep.predictions, key= lambda x: table_rep.predictions[x])
            if(table_rep.number == '0' or table_rep.number == '00' or int(table_rep.number) >= 10):
                table_rep.back_hand = False
            # TODO: se è dorso fai cose diverse per table_rep.number (salva in altra variabile nell'oggetto?)
    elif table_rep.number != None and cont_frame<=100:
        table_rep.back_hand = False
        pt_model = portable_model(model_type.NUMBERS)
        num_pred = hand_gesture_recognition(frame, hands, mp_hands, mp_drawing, mp_drawing_styles, pt_model, hands_norm_factor)
        if(num_pred == -1):
            print("Model not found. Please try again")
        elif(num_pred!=None):
            keys = table_rep.predictions.keys()
            if(num_pred in keys):
                val = table_rep.predictions[num_pred]
                table_rep.predictions[num_pred] = val+1
            else:
                table_rep.predictions[num_pred] = 1
        if(cont_frame == 100):
            print("FINISH")
            if len(table_rep.predictions) == 0:
                return None
            new_num = table_rep.number + str(max(table_rep.predictions, key= lambda x: table_rep.predictions[x]))
            print(new_num)
            table_rep.number = new_num
    elif table_rep.foul == None:
        table_rep.foul = "Personal"
    elif table_rep.penalty == None:
        pt_model = portable_model(model_type.PENALTY)
        penalty_pred = hand_gesture_recognition(frame, hands, mp_hands, mp_drawing, mp_drawing_styles, pt_model, hands_norm_factor)
        if cont_frame < 200:
            if(penalty_pred == -1):
                print("Model not found. Please try again")
            elif(penalty_pred!=None):
                keys = table_rep.predictions.keys
                if(penalty_pred in keys()):
                    val = table_rep.predictions[penalty_pred]
                    table_rep.predictions[penalty_pred] = val+1
                else:
                    table_rep.predictions[penalty_pred] = 1
        if(cont_frame == 200):
            if len(table_rep.predictions) == 0:
                return None
            table_rep.penalty = max(table_rep.predictions, key= lambda x: table_rep.predictions[x])
    return table_rep

if __name__ == "__main__":
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(static_image_mode = True, min_detection_confidence = 0.4)

    cap = cv2.VideoCapture(0)
    table_rep: table_report_obj = table_report_obj()
    op = 1
    cont = 0
    calibration = False # True if we are doing calibration of hands
    norm_factor = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Oh no")
            break
        if(calibration):
            cal_result = referee_calibration(frame, hands)
            if cal_result != None:
                norm_factor = cal_result
            cv2.putText(frame, "Please, show both open hands", (0,30), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,0,0),2, cv2.LINE_AA)
            cv2.putText(frame, "Press 'c' to process with the table report", (0,470), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0),2, cv2.LINE_AA)
        else:
            if(table_rep.number == None or table_rep.penalty == None or table_rep.foul == None):
                table_rep = table_report(frame, hands, mp_hands, mp_drawing, mp_drawing_styles, table_rep, cont, norm_factor)
                cont = cont+1
            if(table_rep.number!= None and table_rep.back_hand == True):
                print("Detected back of the hands. Give me the second number")
                cont = 0
            if(cont%50==0):
                print(cont)

            if(table_rep == None):
                print("Element is still None after prediction")
                break
            if(table_rep.number != None):
                cv2.putText(frame, "Player: "+str(table_rep.number), (0,30), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,0,255),2, cv2.LINE_AA)
            if(table_rep.foul != None):
                cv2.putText(frame, "Foul: "+str(table_rep.foul), (0,60), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,255,0),2, cv2.LINE_AA)
            if(table_rep.penalty != None):
                cv2.putText(frame, "Penalty: "+str(table_rep.penalty), (0,90), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255,0,0),2, cv2.LINE_AA)
                cv2.putText(frame, "Press 'r' to init a new report", (0,440), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0),2, cv2.LINE_AA)
                cv2.putText(frame, "Press 'q' to exit the process", (0,470), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0),2, cv2.LINE_AA)
                
            
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break
        if cv2.waitKey(25) == ord('r'):
            table_rep = table_report_obj()
            cont = 0
        if cv2.waitKey(25) == ord('c'):
            if(len(norm_factor)!=0):
                calibration = False
                cont = 0
            else:
                print("You still haven't calibrated your hands! Please show both open hands")

    cap.release()
    cv2.destroyAllWindows()