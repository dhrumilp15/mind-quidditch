import numpy as np
import cv2
import logging
import os

def calibrate_camera():
    try:
        mtx = np.load('calibrate_images/mtx.npy')
        dist = np.load('calibrate_images/dist.npy')
    except (OSError, FileNotFoundError) as error:
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        pattern_size = (9,6)
        objp = np.zeros((np.prod(pattern_size),3), np.float32)
        # objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)
        objp[:,:2] = np.indices(pattern_size).T.reshape(-1,2)
        objp *= 20

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.
        count = 0
        images = ["calibrate_images/" + name for name in os.listdir("./calibrate_images/") if name.endswith(".jpg")]
        # logging.info(images)

        for fname in images:
            img = cv2.imread(fname)
            dsize = (img.shape[1] // 2, img.shape[0] // 2)
            img = cv2.resize(img, dsize)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, pattern_size)

            # If found, add object points, image points (after refining them)
            if ret:
                count += 1
                corners = cv2.cornerSubPix(gray,corners,(5,5),(-1,-1), criteria)
                imgpoints.append(corners.reshape(-1, 2))
                objpoints.append(objp)
            else:
                print(f'Couldn\'t find corners for {fname}!')
        print(f"Found Chessboard Corners for {count} images")
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
        logging.info(f'''
        ---------------------------
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
        np.save('calibrate_images/mtx.npy', mtx)
        np.save('calibrate_images/dist.npy', dist)
        np.save('calibrate_images/rvecs.npy', rvecs)
        np.save('calibrate_images/tvecs.npy', tvecs)
        
        # TODO: ERROR LOGGING in separate class
        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            mean_error += error
        mean_error /= len(objpoints)
        print(f'Reprojection Error: {mean_error}')
    return mtx, dist