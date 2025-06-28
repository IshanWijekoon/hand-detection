import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import mediapipe as mp

mp_draw = mp.solutions.drawing_utils
mp_hand = mp.solutions.hands

style_landmark = mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=4)  # Red dots
style_connection = mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2)                # Green lines

vid_window = cv2.VideoCapture(0)
vid_window.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
vid_window.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
vid_window.set(cv2.CAP_PROP_FPS, 60)  # Try forcing higher FPS if supported

with mp_hand.Hands(max_num_hands = 2,
                   min_detection_confidence = 0.5,
                   min_tracking_confidence = 0.8,
                   model_complexity=1  # Enables temporal smoothing
                   ) as hands:
    while True:
        ret, image=vid_window.read()

        image = cv2.flip(image, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # convert frames into rgb

        # making more accurecy
        image.flags.writeable = False  
        output = hands.process(image) # process the image
        image.flags.writeable = True

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 

        if output.multi_hand_landmarks:
            for hand_landmark in output.multi_hand_landmarks:
                mp_draw.draw_landmarks(image, 
                                       hand_landmark, 
                                       mp_hand.HAND_CONNECTIONS,
                                       style_landmark,
                                       style_connection)
                                       
          # Get bounding box coordinates
                h, w, c = image.shape
                x_list = []
                y_list = []
                for lm in hand_landmark.landmark:
                    x_list.append(int(lm.x * w))
                    y_list.append(int(lm.y * h))

                x_min, x_max = min(x_list), max(x_list)
                y_min, y_max = min(y_list), max(y_list)

                # Draw bounding box
                cv2.rectangle(image, (x_min - 20, y_min - 20), (x_max + 20, y_max + 20), (255, 0, 0), 2)

        cv2.imshow("Hand Pose", image)

        key = cv2.waitKey(1)
        if key==ord('n'):
            break
        elif cv2.getWindowProperty("Hand Pose", cv2.WND_PROP_VISIBLE) < 1:
            break
    
vid_window.release()
cv2.destroyAllWindows()
