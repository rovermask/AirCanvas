import numpy as np
import cv2
from collections import deque
import mediapipe as mp
from datetime import datetime

# Helper function to calculate distance
def distanceBetweenPoints(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def prepareCanvas(frame, color, background):
    # Ensure background matches frame dimensions
    if background.shape[:2] != frame.shape[:2]:
        background = cv2.resize(background, (frame.shape[1], frame.shape[0]))
    
    # Ensure the same number of channels
    if len(background.shape) != len(frame.shape):
        if len(background.shape) == 2:  # If background is grayscale
            background = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
    
    blended = cv2.addWeighted(frame, 0.7, background, 0.3, 0)
    cv2.rectangle(blended, (0, 0), (640, 60), color, -1)
    return blended

# Helper function to generate unique file names
def save_unique_filename():
    filename = f"Drawing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    return filename

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Create a paint window (white canvas)
paintWindow = np.ones((480, 640, 3), dtype=np.uint8) * 255
back_im = cv2.imread("PainWindow2.jpg")

# Unified system for colors and points
points_dict = {
    'blue': {'color': (255, 0, 0), 'points': [deque(maxlen=512)], 'index': 0},
    'green': {'color': (0, 255, 0), 'points': [deque(maxlen=512)], 'index': 0},
    'red': {'color': (0, 0, 255), 'points': [deque(maxlen=512)], 'index': 0},
    'white': {'color': (255, 255, 255), 'points': [deque(maxlen=512)], 'index': 0},
}
eraser_regions = []

# Default drawing settings
current_color = 'red'
drawing_allowed = True

# Load the default webcam
cap = cv2.VideoCapture(0)

# Error handling for camera access
if not cap.isOpened():
    print("Error: Unable to access the webcam.")
    exit()

try:
    while True:
        # Read the frame from the camera
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture frame.")
            break

        frame = cv2.flip(frame, 1)  # Flip the frame horizontally
        frame = prepareCanvas(frame, (255, 255, 255), back_im)
        paintWindow = prepareCanvas(paintWindow, (0, 0, 0), back_im)

        # Process the frame using MediaPipe
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Initialize centers for finger tracking
        index_center, middle_center = None, None

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

                index_center = (int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0]))
                middle_center = (int(middle_tip.x * frame.shape[1]), int(middle_tip.y * frame.shape[0]))

                dist = distanceBetweenPoints(index_center, middle_center)

                # Draw the index and middle finger tip positions
                cv2.circle(frame, index_center, 4, points_dict[current_color]['color'], -1)
                cv2.circle(frame, middle_center, 4, points_dict[current_color]['color'], -1)

                # Control drawing behavior with pen-up logic
                dynamic_threshold = 0.05 * min(frame.shape[0], frame.shape[1])  # 5% of frame height
                if dist > dynamic_threshold:
                    drawing_allowed = False
                else:
                    drawing_allowed = True

        # Handle drawing or erasing
        if index_center and drawing_allowed:
            if index_center[1] < 60:  # Button area
                if index_center[0] <= 128:  # Clear Button
                    for key in points_dict:
                        points_dict[key]['points'] = [deque(maxlen=512)]
                        points_dict[key]['index'] = 0
                    paintWindow[60:, :, :] = 255
                elif index_center[0] <= 256:
                    current_color = 'red'
                elif index_center[0] <= 384:
                    current_color = 'green'
                elif index_center[0] <= 512:
                    current_color = 'blue'
                elif index_center[0] <= 640:
                    current_color = 'white'
            else:
                points_dict[current_color]['points'][points_dict[current_color]['index']].appendleft(index_center)
        else:
            for key in points_dict:
                points_dict[key]['points'].append(deque(maxlen=512))
                points_dict[key]['index'] += 1

        # Draw lines based on the tracked points
        for key, data in points_dict.items():
            for point_group in data['points']:
                for i in range(1, len(point_group)):
                    if point_group[i - 1] is None or point_group[i] is None:
                        continue
                    cv2.line(frame, point_group[i - 1], point_group[i], data['color'], 2)
                    cv2.line(paintWindow, point_group[i - 1], point_group[i], data['color'], 2)

        # Show all the windows
        cv2.imshow("Tracking", frame)
        cv2.imshow("Paint", paintWindow)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):  # Exit the application
            break
        elif key == ord("c"):  # Clear paint window
            for key in points_dict:
                points_dict[key]['points'] = [deque(maxlen=512)]
                points_dict[key]['index'] = 0
            paintWindow[60:, :, :] = 255
        elif key == ord("e"):  # Save the drawing
            file_name = save_unique_filename()
            cv2.imwrite(file_name, paintWindow[60:, :, :])
            print(f"Saved drawing as {file_name}.")
        else:
            pass

except Exception as e:
    print(f"An unexpected error occurred: {e}")

# Release the camera and all resources
cap.release()
cv2.destroyAllWindows()
