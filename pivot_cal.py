import numpy as np
import cv2
import cv2.aruco as aruco
import pyrealsense2 as rs

from Realsense import RealSense
from  marker_setup import setup


cam = RealSense()

tf_dict = setup(cam)

# User input of how long to do calibration for
num_pts = raw_input("How many pivot calibration points do you want to use?")
comm_marker_id = raw_input("What is ID of the top marker?")

aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters_create()

i = 0
A = []
b = []
while (i < num_pts):
    input("Move the tool to new pose, and press enter to record.");

    ids, rvecs, tvecs = cam.detect_markers_realsense()
    if len(ids) > 0:
        if comm_marker_id in ids:
            comm_index = ids.index(comm_marker_id)
            R_j, _ = cv2.Rodrigues(rvecs[comm_index])
            t_j = tvecs[comm_index]
        else:
            R_i_comm, _ = cv2.Rodrigues(tf_dict[ids[0]][0:2])
            t_i_comm = tf_dict[ids[0]][3:5]
            R_comm_i = R_i_comm.transpose()
            t_comm_i = np.matmul(-R_comm_i, t_i_comm)

            R_i, _ = cv2.Rodrigues(rvecs[0])
            t_i = tvecs[0]

            R_j = np.matmul(R_comm_i, R_i)
            t_j = np.matmul(R_comm_i, t_i) + t_comm_i

        item = R_j[0,:].extend([-1, 0, 0])
        A.append(item)
        item = R_j[1,:].extend([0, -1, 0])
        A.append(item)
        item = R_j[2,:].extend([0, 0, -1])
        A.append(item)

        b.extend(-t_j)
        i = i + 1
    else:
        print("No markers detected. Try again.")

# A = [ R_j | -I ]
# b = [ -p_j ]
# Solve least-squares Ax = b
x = np.linalg.lstsq(A,b) # x[0:2] = p_t, x[3:5] = p_pivot

