import numpy as np
import cv2
import cv2.aruco as aruco
import pyrealsense2 as rs
import pickle
from Realsense import RealSense


cam = RealSense()
profile = cam.pipeline.start(cam.config)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
align_to = rs.stream.color
align = rs.align(align_to)

num_markers = raw_input('How many markers does the tool have? ')
with open('markertool'+str(num_markers)+'.pickle', 'rb') as f:
    tf_dict = pickle.load(f)
# print(tf_dict)

# User input of how long to do calibration for
num_pts = raw_input("How many pivot calibration points do you want to use? ")
num_pts = int(num_pts)
comm_marker_id = raw_input("What is ID of the top marker? ")
comm_marker_id = int(comm_marker_id)

aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters_create()

i = 0
A = []
b = []
print('Move tool to new pose, and press p to record')
while (i < num_pts):
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
        ids.shape = (ids.size)
        # print(corners)
        print(ids)
        ids = ids.tolist()
        if len(ids) > 0:
            if comm_marker_id in ids: # if comm_marker_id is detected, use that point
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

                # combine transformation tgoether to get transformation from comm_marker_id to camera frame
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
            i = i + 1
            print('Recorded')
        else:
            print("No markers detected. Try again.")

# Solve least-squares Ax = b
A = np.asanyarray(A)
b = np.asanyarray(b)
# print(A)
# print(b)
x = np.linalg.lstsq(A,b, rcond=None) # x[0:2] = p_t, x[3:5] = p_pivot
print(x)

