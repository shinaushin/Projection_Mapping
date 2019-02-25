import numpy as np
import cv2
import cv2.aruco as aruco
import pyrealsense2 as rs

import Realsense
import marker_setup


cam = RealSense()

tf_dict = setup(cam)

# User input of how long to do calibration for
num_pts = raw_input("How many pivot calibration points do you want to use?")

aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters_create()

i = 0
A = []
b = []
while (i < num_pts): # Time passed is less than user input
    input("Move the tool to new pose, and press enter to record.");

    cam.detect_markers_realsense()

    i = i + 1;

# Solve least-squares Ax = b
x = np.linalg.lstsq(A,b) # x[0:2] = p_t, x[3:5] = p_pivot

