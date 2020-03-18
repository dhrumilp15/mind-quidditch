import numpy as np
import cv2
from collections import deque
from imutils.video import VideoStream
import argparse
import imutils
import time
import socket

def screenDebug(frame, *messages):
    '''
    Prints a given string to the window supplied by frame
    :param frame: The window on which the message will be displayed
    :param message: The string to show on the given frame
    '''
    height, width, ch4eannels = frame.shape
    font                    = cv2.FONT_HERSHEY_SIMPLEX
    defwidth                = 10
    defheight               = height - 20
    fontScale               = 1
    fontColor               = (255,255,255)
    thickness               = 1
    for index, message in enumerate(messages):
        cv2.putText(frame, message, (defwidth, defheight - index * 30), font, fontScale, fontColor, thickness, cv2.LINE_AA)

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
	help="max buffer size") 
args = vars(ap.parse_args())

# plushie
# balllow = np.array([32, 62, 67])
# ballhigh = np.array([51, 255, 255])

# Green
# balllow = np.array([28,98,63])
# ballhigh = np.array([65,255,255])

# # Another
# balllow = np.array([35,43,117])
# ballhigh = np.array([57,113,152])

# Apple
# balllow = np.array([18, 119, 81])
# ballhigh = np.array([39, 255, 255])

# Orange
balllow = np.array([0,159,156])
ballhigh = np.array([25,255,234])

pts = deque(maxlen=args["buffer"])
xtrace = deque(maxlen=args["buffer"])
ytrace = deque(maxlen=args["buffer"])

# 11.7 px for 36in, 1.57in diameter / 2 for .785in radius of ping pong ball
def get_dist(radius, ref_dist = (11.7, 36, 0.785)) -> float:
    '''
        Gets the distance in inches from the object to the camera
        :param radius: The radius of the ball (in px)
        :param ref_dist: Conditions that are true, calculated with the camera's focal length
        :return: The distance from the object to the camera
    '''
    return np.prod(ref_dist) / radius

#TODO Implement drone movement
def move(x = 0.0, y = 0.0, z = 0.0):
    ''' 
        Moves the drone according to the given parameters, currently assumes no movement perpendicular to its long axis
        (no movement in the z-direction)
        :param x: Movement in the x direction
        :param y: Movement in the y direction
        :param z: Movement in the z direction --> Currently not being used
    '''
    pass

def get_hough_frame(frame, x,y, multiplier = 1.5) -> list:
    '''
    Gets a smaller window of the frame for the Hough transform
    :param frame: The original frame
    :param x: The x value of the center of the current best circle
    :param y: The y value of the center of the current base circle
    :param multiplier: The value to multiply the raidus by since the colour mask underestimates the true area of the ball
    :return: A smaller window as an ndarray
    '''
    if x and y:
        # x, y = x, y
        ymin = int(max(0, y - multiplier * radius))
        ymax = int(min(600, y + multiplier * radius))
        xmax = int(max(0, x - multiplier * radius))
        xmin = int(min(600, x + multiplier * radius))
    # Simply returning frame[ymin:ymax, xmax: xmin] would be great, but then we get weird potential problems in 
    # Reduce frame to a circle?
    # givenframe = frame[ymin:ymax, xmax:xmin]
    
    # print(f"ymin: {y1min}, ymax: {ymax}, xmax: {xmax}, xmin: {xmin}")
    return frame[ymin:ymax, xmax:xmin]

if not args.get("video", False):
    vs = VideoStream(src=0).start()
else:
    vs = cv2.VideoCapture(args["video"])

# state = np.array([0,0,np.array([1,0,0,0]), 0])
pc_0 = np.array([0,0,0])
R = np.array([[1,0,0], [0,1,0], [0,0,1]], dtype=np.float32)

TCP_IP = '192.168.0.34'
TCP_PORT = 5000

while True:
    # Capture frame-by-frame
    frame = vs.read()

    frame = frame[1] if args.get("video", False) else frame
    if frame is None:
        break

    frame = imutils.resize(frame, width=600)
    frame = cv2.GaussianBlur(frame,(5,5), 0)
    graysc = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    # Colour Mask Implementation:
    mask = cv2.inRange(hsv, balllow, ballhigh)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    radius = None
    y, x = None, None
    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        ((x,y), radius) = cv2.minEnclosingCircle(c)
        radius = int(radius)

        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        
        # if radius > 10:
            # cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            # cv2.circle(frame, center, 5, (0, 0, 255), -1)
            # if radius:
            #     screenDebug(frame, f"radius(px): {radius:.4f}", f"Distance(in):{getdist(radius):.4f}")
            # pts.appendleft(center)    
    # for i in range(1, len(pts)):
    #     if not pts[i-1] or not pts[i]:
    #         continue
    #     thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
    #     cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
    
    # Laplacian filter Implementation:
    # dst = cv2.Laplacian(frame, cv2.CV_16S, ksize=3)
    # abs_dst = cv2.convertScaleAbs(dst)

    # Canny edge detection
    high = 200
    edges = cv2.Canny(graysc, high // 2, high)
    
    # HoughCircles
    dp = 1.2
    minDist = 100
    accthresh = 30
    if x and y:
        rad_mult = 1.5
        c0, c1 = center
        smallerframe = get_hough_frame(frame, c0,c1, rad_mult)
        graysc = cv2.cvtColor(smallerframe, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(graysc, cv2.HOUGH_GRADIENT, dp, minDist, param1=high,param2=accthresh,minRadius=0,maxRadius=200)

        if circles is not None:
            circles = np.uint16(np.around(circles))
            xadj, yadj, radius = circles[0,:][0]
            
            # Red circle is in the grayscale frame
            cv2.circle(img=graysc,center=(xadj, yadj), radius= radius, color= (0,0,255),thickness=2)
            # draw the center of the circle
            cv2.circle(img=graysc,center=(xadj,yadj), radius=2, color=(0,0,255),thickness=3)
            
            x += int(xadj - rad_mult * radius)
            y += int(yadj - rad_mult * radius)

        # If there are no circles found, use the colour mask
        
        # Green circle will be the image in the image
        x= int(x)
        y=int(y)
        cv2.circle(img=frame,center=(x,y), radius= radius, color= (0,255,0), thickness=2)
        # draw the center of the circle
        cv2.circle(img=frame, center=(x,y), radius=2, color=(0,255,0), thickness=3)
        # Add points to the front of the queue for better tracing
        pts.appendleft((x,y))
        xtrace.appendleft(x)
        ytrace.appendleft(y)
    
    # Quadratic fit approach
    # Ideally, we'd like to give a higher weight to more recent values, since they'll ideally be more accurate
    # The accuracy of the model at certain distances has yet to be established, but I think it's a little like a
    # gaussian distribution - so we'll have to play around with the weights
    
        # if len(set(xtrace)) > 3 and len(set(ytrace)) > 3: # Only if there are more than three distinct values
        #     coeffs, res, rank, singular_values, rcond = np.polyfit(x= xtrace, y= ytrace, deg= 2,full= True) # For testing purposes
        #     print(coeffs)
        # coeffs = np.polyfit(x = xtrace,y =  ytrace,deg =  2) # For real use

        # Super Simple, keep-the-ball-in-the-center(TM) Approach - Most other sources
        screencenter = (300,300)
        # Create ideal given path for which the drone will always be able to catch the ball, and then adjust to match that path
        dx = x - screencenter[0]    
        dy = y - screencenter[1]

        move(x=dx,y=dy)

        screenDebug(frame, f"radius(px): {radius:.4f}", f"Distance(in):{get_dist(radius):.4f}")

    cv2.imshow('frame', frame)
    cv2.imshow('graysc', graysc)
    # cv2.imshow('Colour mask',mask)
    # cv2.imshow('Laplacian', abs_dst)
    cv2.imshow('Canny', edges)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    
if not args.get("video", False):
	vs.stop()
# otherwise, release the camera
else:
	vs.release()
# When everything done, release the capture
cv2.destroyAllWindows()