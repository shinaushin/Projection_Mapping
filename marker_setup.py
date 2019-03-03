import numpy as np
import cv2
import cv2.aruco as aruco
import pyrealsense2 as rs
import Realsense


def tf_btw_markers(ref, orig, tf_dict):


def setup(cam):
    num_markers = raw_input("How many markers does the tool have?")
    tf_dict = {}
      
    # Detect all markers and calculate transformation between each one
    while (len(tf_dict) < num_markers-1):
        ids, rvecs, tvecs = cam.detect_markers_realsense()
        ids, rvecs, tvecs = [x,y,z for x,y,z in sorted(zip(ids, rvecs, tvecs))]
        for i in range(len(ids)-1):
            if ( ids[i+1] - ids[i] == 1 and (ids[i], ids[i+1]) not in tf_dict.keys() ):
                tf_dict[(ids[i], ids[i+1])] = []
                R_i, _ = cv2.Rodrigues(rvecs[i])
                R_i_T = R_i.transpose()
                R_i_1, _ = cv2.Rodrigues(rvecs[i+1])
                R_tf = np.matmul(R_i_T, R_i_1)
                r_tf, _ = cv2.Rodrigues(R_tf)                

                neg_R_i_T = -1 * R_i_T
                t_i_inv = matmul(neg_R_i_T, tvecs[i])
                t_tf = np.matmul(R_i_T, tvecs[i+1]) + t_i_inv

                tf_dict[(ids[i], ids[i+1])].append( r_tf )
                tf_dict[(ids[i], ids[i+1])].append( t_tf )

    # Calculate common coordinate frame for all markers
    # Create another dictionary of tfs from each marker to common coordinate frame
    #    orientation of common coordinate frame can be same as first marker
    


