import cv2
import numpy as np
import argparse
from VideoInterface import VideoInterface
from calibrate import calibrate_camera
from os import path
import skvideo.io
from time import sleep


class VideoFileStream(VideoInterface):
    '''Gets a stream from a video file and lets other classes grab information
    about the stream.

    Attributes:
        path: A string indicating the video file path
    '''

    def __init__(self, args):
        '''Inits VideoFileStream with the video file'''
        video_file = args.get("video")
        if not path.exists(video_file):
            raise FileNotFoundError(f"{video_file} doesn't exist")

        self.vs = skvideo.io.FFmpegReader(video_file)
        self.undistorted_frame = None
        self.camera_matrix, self.dist = calibrate_camera()[:2]

    def open_video_stream(self):
        '''Grabs a camera stream'''
        if self.vs is None:
            print("Video Stream couldn't be created")

    def read(self, undistort=True, **kwargs) -> np.array:
        '''Grabs a frame from the video stream.
        Waits until a frame can be read

        Returns:
            Image frame, np.array
        '''
        frame = next(self.vs.nextFrame())
        if kwargs.get('shape', False):
            frame = cv2.resize(frame, kwargs.get('shape'))
        frame = np.flip(frame, axis=2)
        self.frame = frame
        undistorted = cv2.undistort(frame, self.camera_matrix, self.dist)
        self.undistorted_frame = undistorted
        if undistort:
            return undistorted
        else:
            return frame

    def release(self):
        pass


def configure_args() -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--debug", action="store_true", help="Show debug information")
    ap.add_argument("-v", "--video", help="path to the (optional) video file", default=0)
    return vars(ap.parse_args())


if __name__ == "__main__":
    args = configure_args()
    vfs = VideoFileStream(args=args)
    vfs.open_video_stream()
    try:
        while True:
            frame = vfs.read(shape=(640, 360))
            cv2.imshow('frame', frame)
            # if cv2.waitKey(0) & 0xFF == ord('q'):
            # continue
            cv2.waitKey(20)  # just a simple buffer
    except StopIteration:
        print("Reached the end of the video file!")
