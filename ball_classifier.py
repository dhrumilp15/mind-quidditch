import numpy as np
import cv2
from collections import deque
import argparse
import time
import os
import json
import logging
from utils import screenDebug
from calibrate import calibrate_camera
# logging.getLogger().setLevel(logging.INFO)

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

# Orange ping pong ball
balllow = np.array([14,135,149])
ballhigh = np.array([21,255,255])

class BallClassifier:
    def __init__(self, args, imgiter=None):
        self.pts = deque(maxlen=args.get("buffer", 64))
        self.args = args
        self.vs = None
        self.record = []
        self.imgiter = None
        
        if imgiter is not None:
            self.imgiter = imgiter
            self.counter = 2
        elif args.get("video", False):
            self.vs = cv2.VideoCapture(args["video"])
        else:
            self.vs = cv2.VideoCapture(0)
            self.camera_matrix, self.dist = calibrate_camera()
        if self.vs is not None:
            self.vs.set(cv2.CAP_PROP_FPS, 30)
    
    def get_hough_frame(self, frame, center, radius, multiplier = 1.2) -> list:
        '''
        Gets a smaller window of the frame for the Hough transform
        :param frame: The original frame
        :param x: The x value of the center of the current best circle
        :param y: The y value of the center of the current base circle
        :param multiplier: The value to multiply the raidus by since the colour mask underestimates the true area of the ball
        :return: A smaller window as an ndarray
        '''
        # print(frame.shape)
        x,y = center
        if x and y:
            # x, y = x, y
            ymin = int(max(0, y - multiplier * radius))
            ymax = int(min(frame.shape[0], max(0, y + multiplier * radius)))
            xmax = int(max(0, x - multiplier * radius))
            xmin = int(min(frame.shape[1], max(0, x + multiplier * radius)))
        frame = frame[ymin:ymax, xmax:xmin]
        return frame
    
    def hough(self, frame, center, radius):
        # Canny edge detection
        high = 90
        graysc = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dp = 1.2
        minDist = 10
        # accthresh = 10
        rad_mult = 1.15
        smallerframe = self.get_hough_frame(frame=graysc, center=center, radius=radius, multiplier=rad_mult)
        edges = cv2.Canny(smallerframe, high // 2, high)
        cv2.imshow('Edges', edges)
        circles = cv2.HoughCircles(smallerframe, cv2.HOUGH_GRADIENT, dp, minDist, param1=high, param2=50,minRadius=0,maxRadius=200)
        if circles is not None:
            # logging.info("Hough Circles found circles")
            circles = np.uint16(np.around(circles))
            xadj, yadj, radius = circles[0,:][0]
            cv2.circle(smallerframe, (xadj, yadj), radius, (255, 0, 0), thickness=2)
            cv2.circle(smallerframe, (xadj, yadj), 2, (0, 255, 0), thickness=2)
            return ((center[0] + int(xadj - rad_mult * radius), center[1] + int(yadj - rad_mult * radius)), radius)
    
    def colour_mask(self, frame):
        hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, balllow, ballhigh)
        mask = cv2.erode(mask, None, iterations=3)
        mask = cv2.dilate(mask, None, iterations=3)
        cnts, hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        radius = None
        y, x = None, None
        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            ((x,y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            return ((int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])), radius)

    def grab_calibration_points(self, center, radius):
        x, y = center
        r = radius
        return [(x,y), (x, y + r), (x, y - r), (x + r, y), (x - r, y), (x + r/np.sqrt(2), y + r/np.sqrt(2))]
    
    def find_center_and_radius(self, frame):
        # Acquiring Center and Radius
        center, radius = None, None
        cres = self.colour_mask(frame)
        if cres is not None:
            center, radius = cres
            # Updating Center and Radius
            res = self.hough(frame, center, radius)
            if res is not None:
                center, radius = res
                dres = self.hough(frame, res[0], res[1])
                if dres is not None:
                    center, radius = dres
        return center, radius

    def main(self):
        title = 'frame'
        while True:
            if self.imgiter is not None:
                try:
                    frame = next(self.imgiter)
                    title = str(self.counter)
                except StopIteration:
                    logging.info("Reached the end of the test images.")
                    break
            else:
                ret, frame = self.vs.read()
                if not ret:
                    break
            frame = cv2.undistort(frame, self.camera_matrix, self.dist)
            frame = cv2.bilateralFilter(frame, 5, 100, 100)
            
            center, radius = self.find_center_and_radius(frame)
            if center is not None and radius is not None:
                self.record.append((center, radius))
                cv2.circle(img=frame,center=center, radius= int(radius), color= (0,255,0), thickness=2)
                cv2.circle(img=frame,center=center, radius=2, color= (255,0,0), thickness=2)
                screenDebug(frame, f"radius(px): {radius:.4f}")
            if self.imgiter is not None:
                while True:
                    cv2.imshow(title,frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('o'):
                        cv2.destroyAllWindows()
                        break
                self.counter += 1
            else:
                cv2.imshow(title, frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        if self.vs is not None:
            self.vs.release()
        # When everything done, release the capture
        cv2.destroyAllWindows()

def configure_args() -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", help="path to the (optional) video file")
    ap.add_argument("-b", "--buffer", type=int, default=64, help="max buffer size")
    return vars(ap.parse_args())

if __name__ == "__main__":
    args = configure_args()
    bc = BallClassifier(args)
    bc.main()