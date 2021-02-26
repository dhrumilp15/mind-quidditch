import numpy as np
import cv2
from collections import deque
import argparse
import time
import os
import json
import logging
from typing import Tuple
from collections.abc import Sequence

from WebcamStream import WebcamStream
from VideoFileStream import VideoFileStream
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

    def __init__(self, args=None, imgiter=None):
        '''Makes a BallClassifier object

        Args:
            args: A dict of config values
            imgiter: An image iterator for testing images

        Returns:
            A BallClassifier object
        '''
        self.vs = None
        self.record: list = []
        self.imgiter = None
        self.debug: bool = args.get("debug", False)

        if imgiter is not None:
            self.imgiter = imgiter
            self.counter = 1
        else:
            if "video" in args:
                if isinstance(args.get("video"), str) and os.path.exists(args.get("video")):
                    self.vs = VideoFileStream(args)
                else:
                    self.vs = WebcamStream(args.get("video", 0))
            self.vs.open_video_stream()

    def get_hough_frame(self, frame: np.array, center: tuple, radius: float, multiplier: float = 1.2) -> list:
        '''Gets a smaller window of the frame for the Hough transform

        Args:
            frame: An ndarray frame that should contain some circular object
            center: A tuple of the estimate for the x and y position of the ball
            multiplier A float value to consider a larger frame than the one defined by center and radius

        Returns:
            An ndarray that is a smaller window
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

    def hough(self, frame: np.array, center: tuple, radius: float) -> Tuple[Tuple[int, int], float]:
        '''Refines an estimate of the center and radius of a ball using circle Hough Transform

        Args:
            frame: An ndarray that reperesents the frame with the ball in it
            center: A tuple of the (x,y) position of the ball
            radius: A float of the estimated ball's radius

        Returns:
            A tuple that contains a refined estimate of the ball's center and radius 
            If the center is None, this eventually returns None
        '''
        if center is None or radius is None:
            return None
        high = 90
        graysc = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        minDistBetweenCircles = 10
        rad_mult = 1.15
        smallerframe = self.get_hough_frame(graysc, center, radius, rad_mult)
        circles = cv2.HoughCircles(smallerframe, cv2.HOUGH_GRADIENT, 1,
                                   minDistBetweenCircles, param1=high, param2=50, minRadius=5, maxRadius=200)
        if circles is not None:
            # logging.info("Hough Circles found circles")
            circles = np.uint16(np.around(circles))
            xadj, yadj, radius = circles[0, :][0]
            cv2.circle(smallerframe, (xadj, yadj),
                       radius, (255, 0, 0), thickness=2)
            cv2.circle(smallerframe, (xadj, yadj), 2, (0, 255, 0), thickness=2)
            return (center[0] + int(xadj - rad_mult * radius), center[1] + int(yadj - rad_mult * radius)), radius

    def colour_mask(self, frame: np.array) -> Tuple[Tuple[int, int], float]:
        '''Estimates the center and radius of the ball

        Args:
            frame: An ndarray of the image

        Returns:
            An estimate of the ((x, y), radius) from the frame.
            None if it can't find a circle.
        '''
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, balllow, ballhigh)
        mask = cv2.dilate(mask, None, iterations=2)
        mask = cv2.erode(mask, None, iterations=2)

        cnts, hier = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        radius = None
        if self.debug:
            cv2.imshow('mask', mask)
        y, x = None, None
        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            _, radius = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            if M["m00"] == 0:
                return None
            else:
                return (int(M["m10"] // M["m00"]), int(M["m01"] // M["m00"])), radius

    def find_center_and_radius(self, frame: np.array) -> Tuple[Tuple[int, int], float]:
        '''Finds the center and radius from the frame

        First attempts to find the projectile's center and radius using a colour mask
        Then attempts to find the projectile's center and radius using a circle Hough Transform on a smaller frame

        Args:
            frame: An ndarray that represents the seen image

        Returns:
            A tuple of ((x,y),radius) of the estimated ball's position
        '''
        center, radius = None, None
        cres = self.colour_mask(frame)
        if cres is not None:
            center, radius = cres
            # Updating Center and Radius
            res = self.hough(frame, center, radius)
            if res is not None:
                center, radius = res
        return center, radius

    def main(self) -> None:
        '''Main loop for running the system'''
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
            # frame = cv2.GaussianBlur(frame, (3, 3), 0)
            frame = cv2.medianBlur(frame, 5)
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
