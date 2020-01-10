from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import cv2
import time

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64, help="max buffer size")
args = vars(ap.parse_args())

balllow = np.array([0,132,162])
ballhigh = np.array([26,255,255])
pts = deque(maxlen=args['buffer'])

if not args.get("video", False):
    cap = cv2.VideoCapture(0) # Webcam
else:
    cap = cv2.VideoCapture(args['video']) # Regular Video
time.sleep(2.0)

while True:
    ret, frame = cap.read()
    print(frame)
    # frame = frame[1] if args.get("video", False) else frame
    # if frame is None:
    #     print("frame is none, there's an error")
    #     break
    print("Frame.shape: {}".format(frame.shape))
    # W = 1000.0
    # scale = 60
    # (height, width, depth) = frame.shape
    # width = int(width * scale / 100)
    # height = int(height * scale / 100)
    # dim = (width, height)
    # frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, balllow, ballhigh)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    res = cv2.bitwise_and(frame, frame, mask=mask)
    cv2.imshow('frame', frame)
    cv2.imshow('Mask', mask)
    cv2.imshow('res', res)
    if cv2.waitKey(5) & 0xFF == 27: 
        break

    # Destroys all of the HighGUI windows. 
    cv2.destroyAllWindows() 

    # release the captured frame 
    cap.release()
    
    # cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cnts = imutils.grab_contours(cnts)
    # center = None

    # if len(cnts) > 0:
    #     c = max(cnts, key=cv2.contourArea)
    #     ((x, y), radius) = cv2.minEnclosingCircle(c)
    #     M = cv2.moments(c)
    #     center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

    #     if radius > 10.0:
    #         cv2.circle(frame, (int(x), int(y)), int(radius), (0,255,255), 2)
    #         cv2.circle(frame, center, 5, (0, 0, 255), -1)
    # pts.appendleft(center)
    # for i in range(1,len(pts)):
    #     if pts[i-1] is None or pts[i] is None:
    #         continue
    #     thickness = int(np.sqrt(args['buffer'] / float(i + 1)) * 2.5)
    #     cv2.line(frame, pts[i-1], pts[i], (0,0,255), thickness)
    # cv2.imshow('Frame', frame)
    
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
    
    # if not args.get('video', False):
    #     cap.release()
    # time.sleep(0.1)