import numpy as np
import cv2
from collections import deque
import argparse
import time
import os
import json
import logging

from WebcamStream import WebcamStream
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
balllow = np.array([14, 95, 149])
ballhigh = np.array([30, 255, 255])


class BallClassifier:
    '''Classifies balls from a video stream

    Classifies the ball using a colour mask and hough transform if indicated

    Attributes:
        self.vs: The video stream
        self.imgiter: An optional iter for images
        self.record: An optional record of estimated position and radius
        self.debug: Debug flag

    '''

    def __init__(self, args={}, imgiter=None):
        self.vs = None
        self.record = []
        self.imgiter = None
        self.debug = args.get("debug", False)

        if imgiter is not None:
            self.imgiter = imgiter
            self.counter = 2
        else:
            self.vs = WebcamStream(args.get("video", 0))
            self.vs.open_video_stream()

    def get_hough_frame(self, frame, center, radius, multiplier=1.2) -> list:
        '''
        Gets a smaller window of the frame for the Hough transform
        :param frame: The original frame
        :param x: The x value of the center of the current best circle
        :param y: The y value of the center of the current base circle
        :param multiplier: The value to multiply the raidus by since the colour mask underestimates the true area of the ball
        :return: A smaller window as an ndarray
        '''
        newframe = None
        if center is not None:
            x, y = center
            ymin = int(max(0, y - multiplier * radius))
            ymax = int(min(frame.shape[0], max(0, y + multiplier * radius)))
            xmax = int(max(0, x - multiplier * radius))
            xmin = int(min(frame.shape[1], max(0, x + multiplier * radius)))
            newframe = frame[ymin:ymax, xmax:xmin]
        return newframe

    def hough(self, frame, center, radius):
        # Canny edge detection
        high = 90
        graysc = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dp = 1.2
        minDist = 10
        # accthresh = 10
        rad_mult = 1.15
        smallerframe = self.get_hough_frame(
            frame=graysc, center=center, radius=radius, multiplier=rad_mult)
        if smallerframe is None:
            return None
        edges = cv2.Canny(smallerframe, high // 2, high)
        # cv2.imshow('Edges', edges)
        circles = cv2.HoughCircles(smallerframe, cv2.HOUGH_GRADIENT, dp,
                                   minDist, param1=high, param2=50, minRadius=0, maxRadius=200)
        if circles is not None:
            # logging.info("Hough Circles found circles")
            circles = np.uint16(np.around(circles))
            xadj, yadj, radius = circles[0, :][0]
            cv2.circle(smallerframe, (xadj, yadj),
                       radius, (255, 0, 0), thickness=2)
            cv2.circle(smallerframe, (xadj, yadj), 2, (0, 255, 0), thickness=2)
            return ((center[0] + int(xadj - rad_mult * radius), center[1] + int(yadj - rad_mult * radius)), radius)

    def colour_mask(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, balllow, ballhigh)
        mask = cv2.dilate(mask, None, iterations=2)
        mask = cv2.erode(mask, None, iterations=2)

        cnts, hier = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        radius = None
        # if self.debug:
        #     cv2.imshow('mask', mask)
        y, x = None, None
        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            _, radius = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            if M["m00"] == 0:
                return None
            else:
                return (int(M["m10"] // M["m00"]), int(M["m01"] // M["m00"])), radius

    def grab_calibration_points(self, center, radius):
        x, y = center
        r = radius
        return [(x, y), (x, y + r), (x, y - r), (x + r, y), (x - r, y), (x + r/np.sqrt(2), y + r/np.sqrt(2))]

    def find_center_and_radius(self, frame):
        # Acquiring Center and Radius
        center, radius = None, None
        cres = self.colour_mask(frame)
        if cres is not None:
            center, radius = cres
            # Updating Center and Radius
            # res = self.hough(frame, center, radius)
            # if res is not None:
            #     center, radius = res
            # dres = self.hough(frame, res[0], res[1])
            # if dres is not None:
            #     center, radius = dres
        return center, radius

    def main(self):
        title = 'frame'
        while True:
            frame = None
            if self.imgiter is not None:
                try:
                    frame = next(self.imgiter)
                    title = str(self.counter)
                except StopIteration:
                    logging.info("Reached the end of the test images.")
                    break
            else:
                frame = self.vs.read()
                print(frame.shape)
            frame = cv2.GaussianBlur(frame, (3, 3), 0)
            center, radius = self.find_center_and_radius(frame)

            if center is not None and radius is not None:
                cv2.circle(img=frame, center=center, radius=int(
                    radius), color=(0, 255, 0), thickness=2)
                cv2.circle(img=frame, center=center, radius=2,
                           color=(255, 0, 0), thickness=2)
                if self.debug:
                    self.record.append((center, radius))
                    screenDebug(
                        frame, f"radius(px): {radius:.4f}", f"position: {center}")

            if self.imgiter is not None:
                while True:
                    cv2.imshow(title, frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        cv2.destroyAllWindows()
                        break
                self.counter += 1
            else:
                if self.debug:
                    cv2.imshow(title, frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        if self.vs is not None:
            self.vs.release()
        # When everything done, release the capture
        cv2.destroyAllWindows()


def configure_args() -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--debug", action="store_true",
                    help="Show debug information")
    ap.add_argument("-v", "--video",
                    help="path to the (optional) video file", default=0)
    ap.add_argument("-b", "--buffer", type=int,
                    default=64, help="max buffer size")
    print(vars(ap.parse_args()))
    return vars(ap.parse_args())


if __name__ == "__main__":
    args = configure_args()
    bc = BallClassifier(args)
    bc.main()
