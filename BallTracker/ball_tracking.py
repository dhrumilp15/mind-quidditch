from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time

# Argument Parser
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64, help="max buffer size")
args = vars(ap.parse_args())

# Green boundaries in HSV colour space
greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)

# List of tracked points
pts = deque(maxlen=args["buffer"])

# Get reference to webcam / video file
if not args.get("video", False):
    vs = VideoStream(src=0).start()
else:
    vs = cv2.VideoCapture(args["video"])
time.sleep(2.0) # warm up for video

while True:
    # grab the current frame 
    frame = vs.read()
    frame = frame[1] if args.get("video", False) else frame
    # end of video
    if frame is None:
        break
    
    # resize frame, blur, convert to HSV color space
    frame = imutils.resize(frame, width=600)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0) # reduce high freq noise
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV) # convert to HSV

    # construct a mask for the colour green, then erode and dilate to remove
    # small blobs
    mask = cv2.inRange(hsv, greenLower, greenUpper) # locate green things
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Stop video
if not args.get("video", False):
    vs.stop()
else:
    vs.release()

cv2.destroyAllWindows


    


