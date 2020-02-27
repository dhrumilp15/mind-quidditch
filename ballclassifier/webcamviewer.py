import numpy as np
import cv2
from collections import deque
from imutils.video import VideoStream
import argparse
import imutils
import time

def screenDebug(frame, *messages):
    '''
    Prints a given string to the window supplied by frame
    :param frame: The window on which the message will be displayed
    :param message: The string to show on the given frame
    '''
    height, width, channels = frame.shape
    font                    = cv2.FONT_HERSHEY_SIMPLEX
    defwidth                = 10
    defheight               = height - 20
    fontScale               = 1
    fontColor               = (255,255,255)
    thickness               = 1
    for index, message in enumerate(messages):
        cv2.putText(frame, message, (defwidth, defheight - index * 30), font, fontScale, fontColor, thickness, cv2.LINE_AA)

def callback(value):
    pass

def setup_trackbars(range_filter):
    cv2.namedWindow("Trackbars", 0)

    for i in ["MIN", "MAX"]:    
        v = 0 if i == "MIN" else 255

        for j in range_filter:
            cv2.createTrackbar("%s_%s" % (j, i), "Trackbars", v, 255, callback)

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
	help="max buffer size") 
args = vars(ap.parse_args())
# Green
# balllow = np.array([28,98,63])
# ballhigh = np.array([65,255,255])


# # Another
# balllow = np.array([35,43,117])
# ballhigh = np.array([57,113,152])

# Orange
balllow = np.array([0,159,156])
ballhigh = np.array([25,255,234])

pts = deque(maxlen=args["buffer"])


# 11.7 px for 36in, 1.57in diameter of ping pong ball
def getdist(radius, ref_dist = (11.7, 36, 1.57)):
    return  np.prod(ref_dist) / radius

if not args.get("video", False):
    vs = VideoStream(src=0).start()
else:
    vs = cv2.VideoCapture(args["video"])

while True:
    
    # Capture frame-by-frame
    frame = vs.read()

    frame = frame[1] if args.get("video", False) else frame
    if frame is None:
        break

    frame = imutils.resize(frame, width=600)
    frame = cv2.GaussianBlur(frame,(3,3), 0)
    
    graysc = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    # Colour Mask Implementation:
    # mask = cv2.inRange(hsv, balllow, ballhigh)
    # mask = cv2.erode(mask, None, iterations=2)
    # mask = cv2.dilate(mask, None, iterations=2)
    # cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cnts = imutils.grab_contours(cnts)
    # center, radius = None, None
    
    # if len(cnts) > 0:
    #     c = max(cnts, key=cv2.contourArea)
    #     ((x,y), radius) = cv2.minEnclosingCircle(c)
    #     M = cv2.moments(c)
    #     center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    #     if radius > 10:
    #         cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
    #         cv2.circle(frame, center, 5, (0, 0, 255), -1)
    #         if radius:
    #             screenDebug(frame, f"radius(px): {radius:.4f}", f"Distance(in):{getdist(radius):.4f}")
    #         pts.appendleft(center)
    # for i in range(1, len(pts)):
    #     if not pts[i-1] or not pts[i]:
    #         continue
    #     thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
    #     cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
    
    # Laplacian filter Implementation:
    # dst = cv2.Laplacian(frame, cv2.CV_16S, ksize=3)
    # abs_dst = cv2.convertScaleAbs(dst)

    # Canny edge detection
    high = 300
    edges = cv2.Canny(graysc, high // 2, high)

    # HoughCircles
    # circles = cv2.HoughCircles(graysc,cv2.HOUGH_GRADIENT,1, 20, param1=200,param2=100,minRadius=0,maxRadius=0)
    dp = 1
    minDist = 100
    accthresh = 30
    circles = cv2.HoughCircles(graysc, cv2.HOUGH_GRADIENT, 1, minDist, param1=high,param2=accthresh,minRadius=0,maxRadius=200)
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            # draw the outer circle
            cv2.circle(frame,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv2.circle(frame,(i[0],i[1]),2,(0,0,255),3)
    
    # Display the resulting framed
    cv2.imshow('frame', frame)
    # cv2.imshow('Colour mask',mask)
    # cv2.imshow('Laplacian', abs_dst)
    cv2.imshow('Canny', edges)
    # cv2.imshow("Hough Circles", circles)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if not args.get("video", False):
    vs.stop()
else:
    vs.release()
# When everything done, release the capture
cv2.destroyAllWindows()