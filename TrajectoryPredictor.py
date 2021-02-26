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
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import RANSACRegressor
import seaborn as sns

from WebcamStream import WebcamStream
from BallClassifier import BallClassifier
from VideoFileStream import VideoFileStream

sns.set_style("whitegrid", {'axes.grid': False})
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
fig2, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax.elev = -69
ax.azim = 90


class TrajectoryPredictor(object):
    '''Predicts the trajectory of a ball identified by the BallClassifier

    Predicts the trajectory ball by predicting its intial position and velocity.
    All attributes are ndarrays unless otherwise noted.

    Attributes:
        self.pos_history: The record of position measurements.
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
        '''Creates a TrajectoryPredictor Object

        Args:
            args: A dict of config information
        '''
        self.position = np.array([0, 0, 0], dtype=np.float32)
        self.pos_history = np.array([])
        # Assume acceleration is uniform and is Earth's gravitational constant (We're on Earth friends... or are we?)
        self.a = np.array([0, -9.8, 0], dtype=np.float32)
        self.camera_matrix, self.dist, self.rvec, self.tvec = calibrate_camera()
        self.debug = args.get("debug", False)

        self.min_samples = 2
        self.Xr = RANSACRegressor(min_samples=self.min_samples)
        self.Yr = PolynomialFeatures()
        self.YrR = RANSACRegressor(min_samples=self.min_samples)
        self.Zr = RANSACRegressor(min_samples=self.min_samples)

        self.BC = BallClassifier(args)
        if "video" in args:
            if isinstance(args.get("video"), str) and os.path.exists(args.get("video")):
                self.vs = VideoFileStream(args)
            else:
                self.vs = WebcamStream(args.get("video", 0))
            self.vs.open_video_stream()
        self.camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist, (640, 360), 1, (640, 360))
        self.out = cv2.VideoWriter('predicted_path.avi', cv2.VideoWriter_fourcc(
            'M', 'J', 'P', 'G'), 10, (640, 480))

    def find_dist(self, radius) -> float:
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
        '''Calculates the global position of points knowing each point's depth.

        Args:
            points: An Nx2 ndarray of point as they appear in the camera screen
            dist: A Nx1 ndarray of the depth of points from the camera

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

    def predict_path(self, path: np.array, next_points: int = 2) -> np.array:
        '''Predicts the next next_points points of the ball's path

        Args:
            path: An Nx3 ndarray of positions of the ball
            next_points: An integer of the number of next points to predict

        Returns:
            A ndarray of the path of the ball
        '''
        ts = np.arange(path.shape[0])[:, np.newaxis]
        xr = self.Xr.fit(ts, path[:, 0])
        ts_transformed = self.Yr.fit_transform(ts)

        yr = self.YrR.fit(ts_transformed, path[:, 1])
        zr = self.Zr.fit(ts, path[:, 2])

        ts = np.arange(path.shape[0] + next_points)[:, np.newaxis]
        Y_transformed = self.Yr.fit_transform(
            np.arange(ts.shape[0])[:, np.newaxis])

        pred_path = np.zeros(
            ts.shape[0] * 3).reshape(-1, 3)
        pred_path[:, 0] = xr.predict(
            np.arange(ts.shape[0])[:, np.newaxis])
        pred_path[:, 1] = yr.predict(
            Y_transformed)
        pred_path[:, 2] = zr.predict(
            np.arange(ts.shape[0])[:, np.newaxis])
        return pred_path

    def find_interception_point(self, p_0: np.array, v_0: np.array) -> np.array:
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

    def draw_points_to_frame(self, points: np.array, frame: np.array,
                             rvec: np.array = np.identity(3),
                             tvec: np.array = np.zeros(3)) -> None:
        '''Draws points to the frame

        Blue Circles represent the ball's position when sampled.
        Green lines represent the ball's path between points.

        Args:
            points: An ndarray of 3D points to draw to the frame
            frame: An ndarray of the image frame
        '''
        # Our calculations must be done with unflipped x and y axes
        points[:, 0] *= -1.0
        points[:, 1] *= -1.0
        pts, jac = cv2.projectPoints(
            points, rvec, tvec, self.camera_matrix, self.dist)
        cv2.polylines(frame, np.int32([pts]), 0, (0, 255, 0))
        # Draw each point
        for pt in pts:
            x, y = pt[0]
            cv2.circle(frame, (int(x), int(y)), 2, (255, 0, 0), 2)

    def main(self):
        '''Main loop for running the system'''
        while True:
            try:
                frame = self.vs.read(shape=(640, 480))
            except StopIteration:
                break

            # Undistort the frame before computing any measurements
            # frame = cv2.GaussianBlur(frame, (3, 3), 0)
            frame = cv2.medianBlur(frame, 5)
            center, radius = self.BC.find_center_and_radius(frame)

            if center is not None and radius is not None:
                dist_hat = self.find_dist(radius)  # Predicted distance
                p_t = self.find_ball_global_position(
                    np.array([center]), dist_hat)
                # OpenCV flips x and y axes
                p_t[:, 0] *= -1.0
                p_t[:, 1] *= -1.0

                if len(self.pos_history) > 1:
                    v_t = (p_t - self.pos_history[-1])

                if self.pos_history.shape[0] == 0:
                    self.pos_history = p_t
                else:
                    self.pos_history = np.vstack((self.pos_history, p_t))

                logging.info(self.pos_history)

                if self.pos_history.shape[0] > self.min_samples:
                    self.predicted_path = self.predict_path(self.pos_history)
                # interception = self.find_interception_point(p_0, v_0)
                if self.debug:
                    if len(self.pos_history) > self.min_samples:
                        points = self.predict_path(self.pos_history)
                        self.draw_points_to_frame(
                            points, frame)
                    cv2.circle(img=frame, center=center, radius=int(
                        radius), color=(0, 255, 0), thickness=2)
                    cv2.circle(img=frame, center=center, radius=2,
                               color=(255, 0, 0), thickness=2)
                    screenDebug(
                        frame, f"radius: {radius:.4f} px", f"Distance:{self.find_dist(radius):.4f} mm")
                    self.out.write(frame)
            if self.debug:
                cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        self.out.release()
        self.vs.release()
        cv2.destroyAllWindows()
        ax.scatter(
            self.pos_history[:, 0], self.pos_history[:, 1], self.pos_history[:, 2], c=np.arange(self.pos_history.shape[0]))
        points = self.predict_path(self.pos_history)
        ax.plot(points[:, 0], points[:, 1], points[:, 2])
        if self.debug:
            ax1.plot(self.pos_history[:, 0])
            ax1.plot(points[:, 0])
            ax2.plot(self.pos_history[:, 1])
            ax2.plot(points[:, 1])
            ax3.plot(self.pos_history[:, 2])
            ax3.plot(points[:, 2])

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
