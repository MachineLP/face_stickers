import cv2
import sys
from test import interface
 
 
if __name__ == '__main__' :
 
    # Read video
    # video = cv2.VideoCapture("video/WeChatSight1395.mp4")
    video = cv2.VideoCapture(0)
 
    # Exit if video not opened.
    if not video.isOpened():
        print ("Could not open video")
        sys.exit()

    # Read first frame.
    for i in range(10):
        ok, frame = video.read()
    
    h, w, c = frame.shape
    
    while True:
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            break
        frame = cv2.resize( frame, (w//2, h//2) )
        frame = interface(frame)
 
        # Display result
        cv2.imshow("Tracking", frame)
 
        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27 : break

