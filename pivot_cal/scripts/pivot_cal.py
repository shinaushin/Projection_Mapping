# pivot_cal.py
# author: Austin Shin

import sys
sys.path.append('../../')

import cv2
import cv2.aruco as aruco
import math
import numpy as np
import pickle

import pyrealsense2 as rs
from Realsense import RealSense

from rot_mat_euler_angles_conversion import rotToEuler

# saved as [0.00812451 -0.00558087 -0.21730515] N = 10

def calibrate(cam, align, marker_IDs, num_markers, comm_marker_id, tf_dict, num_pts):
    """


    Args:


    Returns:

    """
    tolerance = 4 # degs; set by user

    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters_create()

    k = 0
    A = []
    b = []
    print('Move tool to new pose, and press p to record')
    while (k < num_pts):
        frames = cam.pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        frame = color_image

        cv2.imshow('frame', frame)
        userinput = cv2.waitKey(10)
        if userinput & 0xFF == ord('p'): # user wants to record point
            font = cv2.FONT_HERSHEY_SIMPLEX
            corners, ids, rvecs, tvecs = cam.detect_markers_realsense(frame)
            if comm_marker_id in ids:
                index = np.argwhere(ids==comm_marker_id)
                ids = np.delete(ids, index)

            shouldRecord = True
            if ids is not None and len(ids) > 1:
                ids.shape = (ids.size)
                ids = ids.tolist()
                ideal_angle = 360 / (num_markers-1)
                for i in range(len(ids)-1):
                    print(ids)
                    marker1 = rvecs[i] # tf_dict[ids[i]]
                    marker1_rot, _ = cv2.Rodrigues(marker1[0])
                    j = i + 1
                    marker2 = rvecs[j] # tf_dict[ids[j]]
                    marker2_rot, _ = cv2.Rodrigues(marker2[0])
                    marker1_rot_T = marker1_rot.transpose()
                    rot_btw_1_2 = np.matmul(marker1_rot_T, marker2_rot)
                    angles = rotToEuler(rot_btw_1_2)
                    y_angle = np.absolute(angles[1])*180/3.1415
                    print(y_angle)
                    if (np.absolute(ids[i] - ids[j]) > 1 and \
                            np.absolute(ids[i] - ids[j]) < num_markers-2) or \
                            y_angle < ideal_angle - tolerance or \
                            y_angle > ideal_angle + tolerance:
                        shouldRecord = False
                        print("Bad orientation found")
                        break

                if shouldRecord:
                    if comm_marker_id in ids: # if comm_marker_id detected, use point
                        comm_index = ids.index(comm_marker_id)
                        R_j, _ = cv2.Rodrigues(rvecs[comm_index])
                        t_j = tvecs[comm_index]
                        t_j.shape = (3)
                    else:
                        # transformation from marker_i frame to common_marker_id frame
                        R_i_comm, _ = cv2.Rodrigues(tf_dict[ids[0]][0])
                        t_i_comm = tf_dict[ids[0]][1]
 
                        # transformation from comm_marker_id frame to marker_i
                        R_comm_i = R_i_comm.transpose()
                        t_comm_i = np.matmul(-R_comm_i, t_i_comm)
                        t_comm_i.shape = (3,1)
  
                        # transformation from marker_i to camera frame
                        R_i, _ = cv2.Rodrigues(rvecs[0])
                        t_i = tvecs[0]
                        t_i.shape = (3,1)

                        # combine transformation tgoether to get transformation
                        # from comm_marker_id to camera frame
                        R_j = np.matmul(R_i, R_comm_i)
                        t_j = np.matmul(R_i, t_comm_i) + t_i
                        t_j.shape = (3)

                    # set up for pivot cal least squares equation
                    # A = [ R_j | -I ]
                    # b = [ -p_j ]
                    item1 = np.append(R_j[0,:], [-1, 0, 0])
                    item1 = item1.tolist()
                    A.append(item1)
                    item2 = np.append(R_j[1,:], [0, -1, 0])
                    item2 = item2.tolist()
                    A.append(item2)
                    item3 = np.append(R_j[2,:], [0, 0, -1])
                    item3 = item3.tolist()
                    A.append(item3)

                    b.extend(-t_j)
                    k = k + 1
                    print('Recorded')
            else:
                print("Not enough markers detected. Try again.")

    # Solve least-squares Ax = b
    A = np.asanyarray(A)
    b = np.asanyarray(b)
    x = np.linalg.lstsq(A,b, rcond=None)[0] # x[0:2] = p_t, x[3:5] = p_pivot
    print(x[0:3])

    return x[0:3]

def main():
    """
    Receives user input and begins calibration

    Args:
        None

    Returns:
        None
    """
    cam = RealSense()
    profile = cam.pipeline.start(cam.config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    align_to = rs.stream.color
    align = rs.align(align_to)

    marker_IDs = raw_input("What are the IDs of the markers of the tool, starting in order with the ones on the side and finally the one on the top all separated by a single space? ")
    marker_IDs = [int(x) for x in marker_IDs.split()]
    num_markers = len(marker_IDs)
    comm_marker_id = marker_IDs[-1]

    with open('../pickles/markertool'+str(num_markers)+'.pickle', 'rb') as f:
        tf_dict = pickle.load(f)

    # User input of how long to do calibration for
    num_pts = raw_input("How many pivot calibration points do you want to use? ")
    num_pts = int(num_pts)

    x = calibrate(cam, align, marker_IDs, num_markers, comm_marker_id, tf_dict,
        num_pts)

    with open('../pickles/test_pivot_cal_markertool'+str(num_markers)+'.pickle', 'wb') as f:
        pickle.dump(x, f)

if __name__ == "__main__":
    main()

