import numpy as np
import argparse
from trajectory_predictor import TrajectoryPredictor
import cv2
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from time import sleep

fig, (ax,ax1,ax2) = plt.subplots(1,3)

class TrajectoryMetrics(object):
    def __init__(self, args):
        self.TP = TrajectoryPredictor(args)
        self.max_value = args["max"]
        

    def project_sample_point(self, points, intrinsic, distortion = np.array([0,0,0,0], dtype=np.float32)):
        rvec = tvec = np.array([[0,0,0]], dtype=np.float32)
        result, jac = cv2.projectPoints(points, rvec, tvec, intrinsic, distortion)
        result = np.squeeze(result, axis=1)
        return result
    
    def get_random_coords(self, number_of_points, max_val):
        return np.squeeze(np.array([[random.random() * max_val for _ in range(3)] for _ in range(number_of_points)], dtype=np.float32))
    
    def unproject_points_accuracy(self):
        sample_points = get_random_coords(10, self.max_value)
        projected_sample_point = self.project_sample_point(sample_points, self.TP.camera_matrix)
        unprojected_sample_point = self.TP.find_ball_global_position(projected_sample_point, np.linalg.norm(sample_points, axis=1))
        error = np.linalg.norm(sample_points - unprojected_sample_point, axis=1)
        accuracy = 1.0 - (error / np.linalg.norm(sample_points,axis=1))
        return np.average(accuracy)
    
    def trajectory_estimation(self):
        a = np.array([0,-9.8,0], dtype=np.float32)
        p_0 = self.get_random_coords(1, self.max_value)
        v_0 = self.get_random_coords(1,self.max_value)
        timestamps = np.linspace(0,50)
        path = np.array([p_0 + v_0 * t + 0.5 * a * t**2 for t in timestamps], dtype=np.float32)
        p_0hat, v_0hat = self.TP.find_initial_conditions(path, timestamps)
        v_1_avg = (path[1] - path[0]) / timestamps[1]
        
        p_error = np.linalg.norm(p_0 - p_0hat)
        v_error = np.linalg.norm(v_0 - v_0hat)
        return 1.0 - p_error / np.linalg.norm(p_0), 1.0 - v_error / np.linalg.norm(v_0)
    
    def main(self):
        print(f"Average Accuracy of finding global positions: {self.unproject_points_accuracy()}")
        print(f"Accuracy of solving for trajectory initial conditions: {self.trajectory_estimation()}")

def configure_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--max", help="The max value of points to project", default=10)
    return vars(ap.parse_args())

if __name__ == "__main__":
    args = configure_args()
    TM = TrajectoryMetrics(args)
    TM.main()