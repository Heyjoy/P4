import numpy as np
import pickle
import cv2
import glob
from pathlib import Path

def calibration():
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob('./camera_cal/cali*.jpg')
    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, (8,6), corners, ret)
        #Do camera calibration given object points and image points

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,(1280,720),None,None)

    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    pickle.dump( dist_pickle, open( "camera_cal/wide_dist_pickle.p", "wb" ) )

def undistort(img):
    camera_pfile = Path("./camera_cal/wide_dist_pickle.p")
    if camera_pfile.is_file():
        dist_pickle = pickle.load( open( "./camera_cal/wide_dist_pickle.p", "rb" ))
        dst = cv2.undistort(img, dist_pickle["mtx"], dist_pickle["dist"], None, dist_pickle["mtx"])
        dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
        #cv2.imwrite('output_images/real_undist.jpg',dst)
        return dst
    else:
        print("not exist this file, now runing cameraCalibartion()!")
        print("afterwards runing this funtion again!")
        calibration()
        return None
