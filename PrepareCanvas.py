import cv2
from AddOverlay import overlay_top_bar

def prepareCanvas(frame,color,back_im):
    cv2.line(frame,(0,60),(frame.shape[1],60),color,2)
    cv2.rectangle(frame,(0,0),(frame.shape[1]//5,60),color,2)
    for i in range(1,5):
        cv2.rectangle(frame,((frame.shape[1]//5)*i,0),((frame.shape[1]//5)*(i+1),60),color,2)
    return overlay_top_bar(frame,back_im[0:63, :, :])