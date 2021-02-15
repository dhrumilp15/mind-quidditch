import cv2
import numpy as np
from ball_classifier import BallClassifier
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import logging
import os
import time
from time import sleep
from utils import screenDebug
from calibrate import calibrate_camera

# fig = plt.figure()
# ax = Axes3D(fig)
# ax.elev = 30
# ax.azim = 45
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# plt.ion()


class TrajectoryPredictor(object):
    '''Predicts the trajectory of a ball identified by the BallClassifier

    Predicts the trajectory ball by predicting its intial position and velocity.
    Note that the trajectory predictor uses x as left-right, y as up-down, and z as depth. This is why the y-value is -9.8

    Attributes:
        self.pos_history: An ndarry of the record of position measurements.
        self.timestamps: An ndarray of the record of time measurements.
        self.position: An ndarray of the current position of the drone.
        self.rvec: An ndarray of the current rotation vector of the drone.

        self.a: The world acceleration.
        self.camera_matrix: The camera matrix acquired from calibration images.
        self.dist: An ndarray of the distortion coeffs
        self.BC: A BallClassifier object to identify the ball's camera coords.
        self.vs: A VideoCapture object from which to pull frames.
    '''

    def __init__(self, args):
        '''Inits TrajectoryPredictor with important information'''
        self.pos_history = np.array([], dtype=np.float32)
        self.timestamps = []
        self.position = np.array([0, 0, 0], dtype=np.float32)
        self.rvec = np.array([0, 0, 0], dtype=np.float32)

        # Assume acceleration is uniform and is Earth's gravitational constant
        # (We're on Earth friends... or are we?)
        self.a = np.array([0, -9.8, 0], dtype=np.float32)
        self.camera_matrix, self.dist = calibrate_camera()
        self.BC = BallClassifier(args)
        if args.get("video", False):
            self.vs = cv2.VideoCapture(args["video"])
        else:
            self.vs = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            self.vs.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
            self.vs.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

    def get_dist(self, radius) -> float:
        '''Gets the distance (in mm) from the object to the camera

        Calculates the distance in mm from the object to the camera using a reference height
        of the ping pong ball (19.939 mm) and the average focal length

        Args:
            radius: The radius (in px) of the ball

        Returns:
            The distance (in mm) from the object to the camera
        '''
        focal_length = np.average(
            [self.camera_matrix[0, 0], self.camera_matrix[1, 1]])  # focal length in px
        h_world = 19.939
        dist = focal_length * h_world / radius
        logging.info(f'Estimated Distance: {dist}')
        return dist

    def find_ball_global_position(self, points, dist) -> np.array:
        '''Calculates the global position of given points knowing their depth.

        Args:
            points: An Nx2 ndarray of points as they appear in the camera screen
            dist: A scalar distance of an object from a camera

        Returns:
            A Nx3 ndarray of global positions
        '''
        f_x = self.camera_matrix[0, 0]
        f_y = self.camera_matrix[1, 1]
        c_x = self.camera_matrix[0, 2]
        c_y = self.camera_matrix[1, 2]

        res = np.array([])
        for index, point in enumerate(points):
            A = ((point[0] - c_x) / f_x)
            B = ((point[1] - c_y) / f_y)
            Z = np.sqrt(dist[index]**2 / (A**2 + B**2 + 1))
            position = np.array([A*Z, B*Z, Z], dtype=np.float32)
            if len(res) == 0:
                res = position
            else:
                res = np.vstack((res, position))
        return res

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
        finals = np.array([], dtype=np.float32)
        fig, (ax, ax1, ax2) = plt.subplots(1, 3)
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

    def main(self):
        while True:
            ret, frame = self.vs.read()
            self.timestamps.append(time.time())
            if not ret:
                break
            new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
                self.camera_matrix, self.dist, (frame.shape[1], frame.shape[0]), 1, (frame.shape[1], frame.shape[0]))
            # Undistort the frame before computing any measurements
            frame = cv2.undistort(src=frame, cameraMatrix=self.camera_matrix,
                                  distCoeffs=self.dist, newCameraMatrix=new_camera_matrix)
            frame = cv2.bilateralFilter(frame, 5, 100, 100)
            center, radius = self.BC.find_center_and_radius(frame)

            if center is not None and radius is not None:
                dist_hat = self.get_dist(radius)  # Predicted distance
                p_t = self.find_ball_global_position([center], dist_hat)
                if len(self.pos_history) > 1:
                    v_t = (p_t - self.pos_history[-1]) / \
                        time.time() - self.timestamps[-1]
                pos = center[0] - self.camera_matrix[0,
                                                     2], center[1] - self.camera_matrix[1, 2]
                print(f"2D POS: {pos}")
                print(f"3D POS: {p_t}\n")

                cv2.circle(img=frame, center=center, radius=int(
                    radius), color=(0, 255, 0), thickness=2)
                cv2.circle(img=frame, center=center, radius=2,
                           color=(255, 0, 0), thickness=2)

                screenDebug(
                    frame, f"radius: {radius:.4f} px", f"Distance:{self.get_dist(radius):.4f} mm")

                if len(self.pos_history) == 0:
                    self.pos_history = np.array([p_t])
                else:
                    self.pos_history = np.vstack((self.pos_history, p_t))
                logging.info(self.pos_history)

                if len(self.pos_history) > 1:
                    p_0, v_0 = self.find_initial_conditions(
                        self.pos_history, self.timestamps)
                    interception = self.find_interception_point(p_0, v_0)

                # ax.scatter3D(xs = self.pos_history[-1][0], ys = self.pos_history[-1][1], zs = self.pos_history[-1][2])
                # fig.canvas.draw()
                # plt.show()
            cv2.imshow('frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


def configure_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video",
                    help="path to the (optional) video file", default=None)
    ap.add_argument("-b", "--buffer", type=int,
                    default=64, help="max buffer size")
    return vars(ap.parse_args())


if __name__ == "__main__":
    args = configure_args()
    tp = TrajectoryPredictor(args)
    tp.main()
