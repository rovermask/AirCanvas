import numpy as np
import cv2
from collections import deque
import mediapipe as mp
from Distance import distanceBetweenPoints
from PrepareCanvas import prepareCanvas
from SaveFile import save

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Set color for drawing
draw_color = (0, 0, 255)  # Red color for drawing

# Create a paint window (white canvas)
paintWindow = np.ones((480,640, 3), dtype=np.uint8) * 255
back_im=cv2.imread("PainWindow2.jpg")

# Giving different arrays to handle colour points of different colour
bpoints = [deque(maxlen=1024)]
gpoints = [deque(maxlen=1024)]
rpoints = [deque(maxlen=1024)]
wpoints = [deque(maxlen=1024)]

# These indexes will be used to mark the points in particular arrays of specific colour
blue_index = 0
green_index = 0
red_index = 0
white_index=0

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255),(255,255,255)]
colorIndex = 0

# Load the default webcam
cap = cv2.VideoCapture(0)




# def eraseOn(frame,index_center):
    # cv2.circle(frame,index_center,4,(255,255,255),-1)
    # pass



drawing_allowed = True  # Flag to control drawing
while True:
    # Read the frame from the camera
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)  # Flip the frame horizontally
    frame=prepareCanvas(frame, (255, 255, 255),back_im)
    paintWindow=prepareCanvas(paintWindow, (0, 0, 0),back_im)

    # Process the frame using MediaPipe
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Initialize centers for finger tracking
    index_center = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

            index_center = (int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0]))
            middle_center = (int(middle_tip.x * frame.shape[1]), int(middle_tip.y * frame.shape[0]))

            dist = distanceBetweenPoints(index_center, middle_center)

            # Draw the index finger tip position
            cv2.circle(frame, index_center, 4, draw_color, -1)
            cv2.circle(frame, middle_center, 4, draw_color, -1)
            
        # Draw lines based on the tracked points (index finger)
        if(dist > 10.0 and dist <35.0):
            bpoints.append(deque(maxlen=512))
            blue_index += 1
            gpoints.append(deque(maxlen=512))
            green_index += 1
            rpoints.append(deque(maxlen=512))
            red_index += 1
            wpoints.append(deque(maxlen=512))
            white_index += 1

        # Now checking if the user wants to click on any button above the screen 
        elif index_center[1] < 60:
            if(index_center[0]<=128): # Clear Button
                bpoints = [deque(maxlen=512)]
                gpoints = [deque(maxlen=512)]
                rpoints = [deque(maxlen=512)]
                wpoints = [deque(maxlen=512)]
                blue_index = 0
                green_index = 0
                red_index = 0
                white_index = 0
                paintWindow[60:,:,:] = 255
            elif(index_center[0]<=256):
                print("RED")
                colorIndex = 2 # Red
            elif(index_center[0]<=384):
                print("GREEN")
                colorIndex = 1 # Green
            elif(index_center[0]<=512):
                print("BLUE")
                colorIndex = 0 # Blue
            elif(index_center[0]<=640):
                print("ERASER")
                colorIndex = 3 # White
        else :
            if colorIndex == 0:
                bpoints[blue_index].appendleft(index_center)
            elif colorIndex == 1:
                gpoints[green_index].appendleft(index_center)
            elif colorIndex == 2:
                rpoints[red_index].appendleft(index_center)
            elif colorIndex == 3:
                wpoints[white_index].appendleft(index_center)
    else:
        bpoints.append(deque(maxlen=512))
        blue_index += 1
        gpoints.append(deque(maxlen=512))
        green_index += 1
        rpoints.append(deque(maxlen=512))
        red_index += 1
        wpoints.append(deque(maxlen=512))
        white_index += 1


    points = [bpoints, gpoints, rpoints, wpoints]
    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1, len(points[i][j])):
                if points[i][j][k - 1] is None or points[i][j][k] is None:
                    continue
                cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 2)
                cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], 2)

    # for i in wpoints:
        # for j in range(1,len(points[i])):
            
    
    # Draw lines based on the tracked index finger points
    # if len(points) > 0 and len(points[0]) > 1:
    #     for i in range(1, len(points[0])):
    #         if points[0][i - 1]['point'] is None or points[0][i]['point'] is None:
    #             continue
    #         if points[0][i]['color'] == (255, 255, 255):
    #             cv2.line(paintWindow, points[0][i - 1]['point'], points[0][i]['point'], points[0][i]['color'], 20)
    #         else:
    #             cv2.line(frame, points[0][i - 1]['point'], points[0][i]['point'], points[0][i]['color'], 2)
    #             cv2.line(paintWindow, points[0][i - 1]['point'], points[0][i]['point'], points[0][i]['color'], 2)

    # Show all the windows
    cv2.imshow("Tracking", frame)
    cv2.imshow("Paint", paintWindow)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):  # Exit by using 'q' key
        print(points)
        break
    elif key == ord("c"):  # Clear paint window using 'c' key
        bpoints = [deque(maxlen=512)]
        gpoints = [deque(maxlen=512)]
        rpoints = [deque(maxlen=512)]

        blue_index = 0
        green_index = 0
        red_index = 0

        paintWindow[60:,:,:] = 255
    elif key == ord("e"):  # Save the drawing using 'e' key
        file_name=save()
        # print(file_name.name)
        cv2.imwrite(file_name.name, paintWindow[60:,:,:])
        break
    else:
        pass

# Release the camera and all resources
cap.release()
cv2.destroyAllWindows()