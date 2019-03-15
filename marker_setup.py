import numpy as np
import cv2
import cv2.aruco as aruco
import pyrealsense2 as rs
from Realsense import RealSense
import pickle

cam = RealSense()
profile = cam.pipeline.start(cam.config)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
align_to = rs.stream.color
align = rs.align(align_to)

num_markers = raw_input("How many markers does the tool have? ")
tf_dict = {}

comm_marker_id = raw_input("What is the ID of the top marker? ")
    
num_markers = int(num_markers)
comm_marker_id = int(comm_marker_id)      

# TODO: how to filter out bad pose estimations    
while (len(tf_dict) < num_markers-1):
    frames = cam.pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    color_frame = aligned_frames.get_color_frame()
    color_image = np.asanyarray(color_frame.get_data())
    frame = color_image
    font = cv2.FONT_HERSHEY_SIMPLEX
    corners, ids, rvecs, tvecs = cam.detect_markers_realsense(frame)

    # find transformation between comm_marker_id and other ids
    if np.all(ids != None) and len(ids) > 1 and comm_marker_id in ids:
        for i in range(0, ids.size):
            aruco.drawAxis(frame, cam.newcameramtx, cam.dist, rvecs[i], tvecs[i], 0.1)  # Draw axis
        aruco.drawDetectedMarkers(frame, corners) #Draw a square around the markers
        ids.shape = len(ids)
        ids = ids.tolist()
        # print(rvecs)
        # print(tvecs)            
        comm_index = ids.index(comm_marker_id) # index of where comm_marker_id is in ids list
        R_comm, _ = cv2.Rodrigues(rvecs[comm_index][0][0])
        t_comm = tvecs[comm_index][0][0]
        t_comm.shape = (3,1)
        
        # rotation and translation to transform camera coordinate frame to comm_marker_id coordinate frame
        R_comm_T = R_comm.transpose()
        t_comm_inv = np.matmul(-R_comm_T, t_comm)

        for i in range(len(ids)):
            if i != comm_index and ids[i] not in tf_dict.keys():
                # transformation from marker_i to camera coordinate frame
                R_i, _ = cv2.Rodrigues(rvecs[i][0][0])
                t_i = tvecs[i][0][0]
                t_i.shape = (3,1)

                R_tf = np.matmul(R_comm_T, R_i) # transformation of marker_i to comm_marker_id frame
                print(R_tf)
                t_tf = np.matmul(R_comm_T, t_i) + t_comm_inv
                r_tf, _ = cv2.Rodrigues(R_tf)
                r_tf.shape = (3)
                t_tf.shape = (3)

                # store transformation in dictionary
                tf_dict[ids[i]] = []
                tf_dict[ids[i]].append( r_tf )
                tf_dict[ids[i]].append( t_tf )

    cv2.imshow('frame', frame)
    cv2.waitKey(10)

print("Done")
print(tf_dict)
    
# store dictionary in pickle
with open('markertool'+str(num_markers)+'.pickle', 'wb') as f:
    pickle.dump(tf_dict, f)
