import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode = True, min_detection_confidence = 0.3)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("Oh no")
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(img_rgb)
    if results.multi_hand_landmarks:
        if len(results.multi_hand_landmarks)==2:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                        img_rgb, # image to draw
                        hand_landmarks, # model output
                        mp_hands.HAND_CONNECTIONS, # hand connections
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
        else:
            print(len(results.multi_hand_landmarks))
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                        img_rgb, # image to draw
                        hand_landmarks, # model output
                        mp_hands.HAND_CONNECTIONS, # hand connections
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
    else:
        print("No hands")

    cv2.imshow('frame', img_rgb)
    if cv2.waitKey(25) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()