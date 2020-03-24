import cv2
import numpy as np
from ball_classifier import BallClassifier
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import logging
# logging.getLogger().setLevel(logging.DEBUG)
import os

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.elev = 0
ax.azim = 60
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.ion()

class TrajectoryPredictor:
    def __init__(self, args):
        if args.get("video", False):
            self.vs = cv2.VideoCapture(args["video"])
        else:
            self.vs = cv2.VideoCapture(0)
        
        self.BC = BallClassifier(args)
        self.record = []

        # Attributes of the ball
        self.pb_0 = np.array([0,0,0], dtype=np.float32)
        self.v_0 = np.array([0,0,0], dtype=np.float32)

        # Assume acceleration is uniform and is Earth's gravitational constant (We're on Earth friends... or are we?)
        self.a = np.array([0,0,-9.8], dtype=np.float32)
        self.camera_matrix = self.calibrate_camera()
        fovx, fovy, self.fL, pP, aR = cv2.calibrationMatrixValues(self.camera_matrix, (1280, 720), 1, 1)
    
    def get_dist(self, radius) -> float:
        '''
            Gets the distance in inches from the object to the camera
            :param radius: The radius of the ball (in px)
            :param ref_dist: Conditions that are true, calculated with the camera's focal length
            :return: The distance from the object to the camera
        '''
        focal_length = self.fL # Accurate to 10 mm
        # focal_length = self.camera_matrix[0, 0] * 25.4/96
        # logging.info(f'Focal Length: {self.fL}')
        dist = focal_length * 19.939 * 720 / radius
        logging.info(f'''Focal Length: {focal_length}
            Estimated Distance: {dist}''')
        return dist
    
    def calibrate_camera(self):
        try:
            return np.loadtxt('calibrate_images/out.txt', dtype=np.float32)
            # f = open('calibrate_images/params.json')
            # saved_params = json.loads(f.read())
            # self.camera_matrix = saved_params["camera_matrix"]
        except OSError:
            # termination criteria
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

            # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
            pattern_size = (9,6)
            objp = np.zeros((np.prod(pattern_size),3), np.float32)
            objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)
            objp[:,:2] = np.indices(pattern_size).T.reshape(-1,2)
            objp *= 2.1

            # Arrays to store object points and image points from all the images.
            objpoints = [] # 3d point in real world space
            imgpoints = [] # 2d points in image plane.

            images = ["calibrate_images/" + name for name in os.listdir("./calibrate_images/") if name.endswith(".jpg") and int(name[:name.index('.jpg')])]
            # logging.info(images)

            for fname in images:
                img = cv2.imread(fname)
                gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

                # Find the chess board corners
                ret, corners = cv2.findChessboardCorners(gray, pattern_size)

                # If found, add object points, image points (after refining them)
                if ret:
                    corners = cv2.cornerSubPix(gray,corners,(5,5),(-1,-1), criteria)
                    imgpoints.append(corners.reshape(-1, 2))
                    objpoints.append(objp)

            rms, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
            logging.info(f'''
            ---------------------------
                RMS:
                {rms}
                Camera Matrix:
                {mtx}
                Dist:
                {dist.ravel()}
                Rvecs:
                {rvecs}
                Tvecs:
                {tvecs}
            ---------------------------
            ''')
            np.savetxt('out.txt', mtx)
            # img = cv2.imread('calibrate_images/59.jpg')
            # h,  w = img.shape[:2]
            # newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
            # # undistort
            # dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

            # # crop the image
            # x,y,w,h = roi
            # dst = dst[y:y+h, x:x+w]
            # cv2.imwrite('calibresult.png',dst)
            
            # TODO: ERROR LOGGING in separate class
            # mean_error = 0
            # for i in range(len(objpoints)):
            #     imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            #     error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            #     mean_error += error
            # mean_error /= len(objpoints)
            # logging.info(f"Reprojection Error: {mean_error}")
            return mtx

    def find_ball_global_position(self, point, Z):
        undistorted = np.squeeze(cv2.undistortPoints(np.array(point, dtype=np.float32), self.camera_matrix, Z, P = self.camera_matrix))

        f_x = self.camera_matrix[0, 0]
        f_y = self.camera_matrix[1, 1]
        c_x = self.camera_matrix[0, 2]
        c_y = self.camera_matrix[1, 2]
        return np.array([((undistorted[0] - c_x) / f_x) * Z, ((undistorted[1] - c_y) / f_y) * Z, Z], dtype=np.float32)

    def main(self):
        while True:
            ret, frame = self.vs.read()
            if not ret:
                break
            center, radius = self.BC.find_center_and_radius(frame)
            
            if center is not None and radius is not None:
                dist_hat = self.get_dist(radius) # Predicted distance
                p_t = self.find_ball_global_position(center, dist_hat)
                
                cv2.circle(img=frame,center=center, radius= int(radius), color= (0,255,0), thickness=2)
                cv2.circle(img=frame,center=center, radius=2, color= (255,0,0), thickness=2)
                self.screenDebug(frame, f"radius(px): {radius:.4f}", f"Distance(mm):{self.get_dist(radius):.4f}")
                
                self.record.append(p_t)
                logging.info(self.record)
                record = np.array(self.record, dtype=np.float32)
                logging.info(record)

                ax.scatter(xs=record[:,0], ys=record[:,1], zs =record[:,2])#, marker=".", s=20, c="goldenrod", alpha=0.6)
                fig.canvas.draw()
                plt.show()
            cv2.imshow('frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    def screenDebug(self, frame, *messages):
        '''
        logging.infos a given string to the window supplied by frame
        :param frame: The window on which the message will be displayed
        :param message: The string to show on the given frame
        '''
        height, width, channels = frame.shape
        font                    = cv2.FONT_HERSHEY_SIMPLEX
        defwidth                = 10
        defheight               = height - 20
        fontScale               = 1
        fontColor               = (255,255,255)
        thickness               = 1
        for index, message in enumerate(messages):
            cv2.putText(frame, message, (defwidth, defheight - index * 30), font, fontScale, fontColor, thickness, cv2.LINE_AA)

def configure_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", help="path to the (optional) video file")
    ap.add_argument("-b", "--buffer", type=int, default=64, help="max buffer size")
    return vars(ap.parse_args())

def project_sample_point(points, intrinsic, distortion = np.array([0,0,0,0], dtype=np.float32)):
        rvec = np.array([[0,0,0]], dtype=np.float32)
        logging.info(f"Rot. Mat for projection{cv2.Rodrigues(rvec)[0]}")
        tvec = np.array([0,0,0], dtype=np.float32)
        result, jac = cv2.projectPoints(points, rvec, tvec, intrinsic, distortion)
        logging.info(f"Projected Sample Point has shape {np.squeeze(result, axis=1).shape}")
        logging.info(f"Projected Sample Point: {np.squeeze(result, axis = 1)}")
        return np.squeeze(result, axis=1)

if __name__ == "__main__":
    args = configure_args()
    tp = TrajectoryPredictor(args)
    tp.main()