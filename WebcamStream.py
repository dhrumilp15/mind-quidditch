import numpy as np
import argparse
from VideoInterface import VideoInterface
import cv2

from calibrate import calibrate_camera
from os import path


class WebcamStream(VideoInterface):
    '''Gets a stream from the webcam and lets other classes grab information
    about the stream.

    Attributes:
        capture_id: An integer or string indicating the camera device 
    '''

    def __init__(self, capture_id=0):
        '''Inits WebcamStream with the capture ID'''
        self.vs = cv2.VideoCapture(capture_id, cv2.CAP_DSHOW)
        # self.test = cv2.VideoCapture(
        #     "ball_images/test_throw.mp4", cv2.CAP_DSHOW)
        self.vs.set(cv2.CAP_PROP_FPS, 30)
        self.undistorted_frame = None
        self.camera_matrix, self.dist = calibrate_camera()[:2]

    def open_video_stream(self):
        '''Grabs a camera stream'''
        if not self.vs.isOpened():
            print("Video Stream couldn't be created")

    def read(self, undistort=True, shape=(640, 360)) -> np.array:
        '''Grabs a frame from the video stream.
        Waits until a frame can be read

        Returns:
            Image frame, np.array
        '''
        while self.vs.isOpened():
            ret, frame = self.vs.read()
            if ret:
                self.frame = frame
                # frame = cv2.resize(frame, shape)
                undistorted = cv2.undistort(
                    frame, self.camera_matrix, self.dist)
                self.undistorted_frame = undistorted
                if undistort:
                    return undistorted
                else:
                    return frame
            else:
                self.release()
                break

    def release(self):
        self.vs.release()
        cv2.destroyAllWindows()


def configure_args() -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--debug", action="store_true",
                    help="Show debug information")
    ap.add_argument("-v", "--video",
                    help="path to the (optional) video file", default=0)
    print(vars(ap.parse_args()))
    return vars(ap.parse_args())


if __name__ == "__main__":
    print(cv2.__version__)
    args = configure_args()
    ws = WebcamStream(args)
    ws.open_video_stream()
    while True:
        frame = ws.read()
        print(frame)
        cv2.imshow("frame", frame)
        if (cv2.waitKey(0) & 0xFF == ord('q')):
            break
