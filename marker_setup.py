import numpy as np
import cv2
import cv2.aruco as aruco
import pyrealsense2 as rs
import Realsense


def setup(cam, startMarker):
    num_markers = raw_input("How many markers does the tool have?")
    tf_dict = {}

    comm_marker_id = raw_input("What is the ID of the top marker?")
          
    while (len(tf_dict) < num_markers-1):
        ids, rvecs, tvecs = cam.detect_markers_realsense()
        ids, rvecs, tvecs = [x,y,z for x,y,z in sorted(zip(ids, rvecs, tvecs))]
        if len(ids) > 1 and comm_marker_id in ids:
            comm_index = ids.index(comm_marker_id)
            R_comm, _ = cv2.Rodrigues(rvecs[comm_index]
            t_comm = tvecs[comm_index] 
            R_comm_T = R_comm.transpose()
            t_comm_inv = np.matmul(-R_comm_T, t_comm)

            for i in range(len(ids)-1):
                if i != comm_index and ids[i] not in tf_dict.keys():
                    R_i, _ = cv2.Rodrigues(rvecs[i])
                    t_i = tvecs[i]
                    R_tf = np.matmul(R_i, R_comm_T)
                    t_tf = np.matmul(R_i, t_comm_inv) + t_i
                    r_tf, _ = cv2.Rodrigues(R_tf)
                    tf_dict[ids[i]].append( r_tf )
                    tf_dict[ids[i]].append( t_tf )
    
    return tf_dict

