import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import json
import logging
import os
import time
from utils import screenDebug
from calibrate import calibrate_camera

from WebcamStream import WebcamStream
from BallClassifier import BallClassifier
from VideoFileStream import VideoFileStream

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.elev = -50
ax.azim = 90


class TrajectoryPredictor(object):
    '''Predicts the trajectory of a ball identified by the BallClassifier

    Predicts the trajectory ball by predicting its intial position and velocity.
    All attributes are ndarrays unless otherwise noted.

    Attributes:
        self.pos_history: The record of position measurements.
        self.timestamps: The record of time measurements.
        self.position: The current position of the drone.
        self.position: The current rotation vector of the drone.

        self.a: The world acceleration. Note that the trajectory predictor uses
                x as left-right, y as up-down, and z as depth. This is why the y-value is -9.8
        self.camera_matrix: The camera matrix acquired from calibration images.
        self.dist: The matrix of distortion coefficients from calibration images.
        self.BC: The BallClassifier object used to identify where the ball is in the image.
        self.vs: The VideoCapture object to be analyzed
    '''

    def __init__(self, args):
        self.pos_history = np.array([], dtype=np.float32)
        self.timestamps = []
        self.position = np.array([0, 0, 0], dtype=np.float32)
        self.rvec = np.array([0, 0, 0], dtype=np.float32)

        # Assume acceleration is uniform and is Earth's gravitational constant (We're on Earth friends... or are we?)
        self.a = np.array([0, -9.8, 0], dtype=np.float32)
        self.camera_matrix, self.dist = calibrate_camera()
        self.debug = args.get("debug", False)

        # fovx, fovy, self.fL, self.pP, aR = cv2.calibrationMatrixValues(self.camera_matrix, (640, 360), 1, 1)
        self.BC = BallClassifier(args)
        if "video" in args:
            if isinstance(args.get("video"), str) and os.path.exists(args.get("video")):
                self.vs = VideoFileStream(args)
            else:
                self.vs = WebcamStream(args.get("video", 0))
            self.vs.open_video_stream()
        self.camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist, (640, 360), 1, (640, 360))

    def get_dist(self, radius) -> float:
        '''Gets the distance (in mm) from the object to the camera

        Calculates the distance in mm from the object to the camera using a reference diameter
        of the ping pong ball (19.939 mm) and the average focal length

        Args:
            radius: The radius (in px) of the ball

        Returns:
            The distance (in mm) from the object to the camera
        '''
        focal_length = np.average(
            [self.camera_matrix[0, 0], self.camera_matrix[1, 1]])  # focal length in px
        h_world = 19.939  # bad hard coded numbers
        dist = focal_length * h_world / radius
        logging.info(f'Estimated Distance: {dist}')
        return dist

    def find_ball_global_position(self, point: np.array, dist: np.array) -> np.array:
        '''Calculates the global position of given point knowing its depth.

        Args:
            points: An Nx2 ndarray of point as they appear in the camera screen
            dist: A Nxk ndarray of the depth of points from the camera

        Returns:
            A Nx3 ndarray of global positions
        '''
        f_x = self.camera_matrix[0, 0]
        f_y = self.camera_matrix[1, 1]
        c_x = self.camera_matrix[0, 2]
        c_y = self.camera_matrix[1, 2]

        # x of the ball
        A = ((point[:, 0] - c_x) / f_x).reshape(point.shape[0], -1)

        # y of the ball
        B = ((point[:, 1] - c_y) / f_y).reshape(point.shape[0], -1)

        # homogeneous coordinate
        Z = np.sqrt(dist**2 / (A**2 + B**2 + 1), dtype=np.float32)
        # Putting each component together
        final = A*Z
        final = np.hstack((final, B*Z))
        final = np.hstack((final, Z))
        return final  # world position

    def find_initial_conditions(self, path, timestamps):
        '''Calculates the initial position and velocity of the ball

        Calculates the initial position and velocity of the ball from timestamps and the path.
        The number of timestamps must be equal to the number of ball position measurements.

        Args:
            path: An Nx3 ndarray of positions of the ball
            timestamps: A 1xN ndarray of timestamps

        Returns:
            A tuple containing the initial position and initial velocity in that order.
        '''
        # print("path")
        # print(path)

        X_coeffs = np.polyfit(timestamps, path[:, 0], 1)
        Y_coeffs = np.polyfit(timestamps, path[:, 1], 2)
        Z_coeffs = np.polyfit(timestamps, path[:, 2], 1)

        p_0 = np.array([X_coeffs[-1], Y_coeffs[-1],
                        Z_coeffs[-1]], dtype=np.float32)
        p_1 = np.array([
            np.polyval(X_coeffs, timestamps[1]),
            np.polyval(Y_coeffs, timestamps[1]),
            np.polyval(Z_coeffs, timestamps[1]),
        ])

        v_0 = (p_1 - p_0) / timestamps[1] - 0.5*self.a * timestamps[1]
        return p_0, v_0

    def find_interception_point(self, p_0, v_0):
        '''Finds the point where the ball will be closest to the drone's current position.

        Finds the point that will take the longest to achieve to give
        the drone ample opportunity to catch the ball.

        Args:
            p_0: The initial position of the ball
            v_0: The initial velocity of the ball

        Returns:
            The interception point as a single 1x3 ndarray
        '''
        t_roots = np.roots([0.5*self.a, v_0, p_0 - self.position])
        t = max(t_roots)
        return np.polyval([0.5*self.a, v_0, p_0], t)

    def truncate_colormap(self, cmap, minval=0.0, maxval=1.0, n=100):
        new_cmap = colors.LinearSegmentedColormap.from_list(
            'trunc({n},{a:.2f},{b:.2f})'.format(
                n=cmap.name, a=minval, b=maxval),
            cmap(np.linspace(minval, maxval, n)))
        return new_cmap

    def main(self):
        while True:
            try:
                frame = self.vs.read(shape=(640, 480))
            except StopIteration:
                break

            # Undistort the frame before computing any measurements
            frame = cv2.GaussianBlur(frame, (3, 3), 0)
            center, radius = self.BC.find_center_and_radius(frame)

            if center is not None and radius is not None:
                self.timestamps.append(time.time())
                dist_hat = self.get_dist(radius)  # Predicted distance
                p_t = self.find_ball_global_position(
                    np.array([center]), dist_hat)
                p_t[:, 0] *= -1.0
                p_t[:, 1] *= -1.0
                # print(p_t)
                if len(self.pos_history) > 1:
                    v_t = (p_t - self.pos_history[-1]) / \
                        (time.time() - self.timestamps[-1])

                if len(self.pos_history) == 0:
                    self.pos_history = p_t
                else:
                    self.pos_history = np.vstack((self.pos_history, p_t))

                logging.info(self.pos_history)

                if len(self.pos_history) > 1:
                    p_0, v_0 = self.find_initial_conditions(
                        self.pos_history, self.timestamps)
                # interception = self.find_interception_point(p_0, v_0)
                if self.debug:
                    cv2.circle(img=frame, center=center, radius=int(
                        radius), color=(0, 255, 0), thickness=2)
                    cv2.circle(img=frame, center=center, radius=2,
                               color=(255, 0, 0), thickness=2)
                    screenDebug(
                        frame, f"radius: {radius:.4f} px", f"Distance:{self.get_dist(radius):.4f} mm")
                    print(p_t)
            if self.debug:
                cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        cv2.destroyAllWindows()

        # new_cmap1 = self.truncate_colormap(plt.get_cmap('jet'), 0.45, 1.0)
        # new_cmap2 = self.truncate_colormap(plt.get_cmap('brg'), 1.0, 0.45)

        # ax[0].imshow(a, aspect='auto', cmap=new_cmap1)
        # ax[1].imshow(a, aspect='auto', cmap=new_cmap2)
        ax.scatter(
            xs=self.pos_history[:, 0], ys=self.pos_history[:, 1], zs=self.pos_history[:, 2], c=self.timestamps)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()


def configure_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video",
                    help="path to the (optional) video file", default=0)
    ap.add_argument("-d", "--debug", action="store_true",
                    help="Show video stream + Debug info", default=False)
    return vars(ap.parse_args())


if __name__ == "__main__":
    args = configure_args()
    tp = TrajectoryPredictor(args)
    tp.main()
