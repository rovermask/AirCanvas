import cv2

# Function to overlay the top bar on the frame
def overlay_top_bar(frame, top_bar):
    h, w, _ = top_bar.shape
    frame[0:h, 0:w] = cv2.addWeighted(frame[0:h, 0:w], 0.2, top_bar, 1,-40)
    return frame