import sys
sys.path.append('../../')

import numpy as np
import cv2
import cv2.aruco as aruco
import pyrealsense2 as rs
from Realsense import RealSense
import pickle
import math
from rot_mat_euler_angles_conversion import rotToEuler


def setup(cam, align, marker_IDs, num_markers, comm_marker_id):
    tolerance = 4

    tf_dict = {}
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
                    t_tf = np.matmul(R_comm_T, t_i) + t_comm_inv
                    r_tf, _ = cv2.Rodrigues(R_tf)
                    r_tf.shape = (3)
                    t_tf.shape = (3)

                    # store transformation in dictionary
                    tf_dict[ids[i]] = []
                    tf_dict[ids[i]].append( r_tf )
                    tf_dict[ids[i]].append( t_tf )

        if len(tf_dict) == num_markers-1:
            # print(tf_dict.keys())
            ideal_angle = 360 / (num_markers-1)
            list_angles = []
            for i in range(num_markers-1):
                marker1 = tf_dict[marker_IDs[i]]
                marker1_rot, _ = cv2.Rodrigues(marker1[0])
                if i == num_markers-2:
                    j = 0
                else:
                    j = i + 1
                marker2 = tf_dict[marker_IDs[j]]
                marker2_rot, _ = cv2.Rodrigues(marker2[0])
                marker1_rot_T = marker1_rot.transpose()
                rot_btw_1_2 = np.matmul(marker1_rot_T, marker2_rot)
                angles = rotToEuler(rot_btw_1_2)
                y_angle = angles[1] * 180 / 3.1415 # np.absolute(angles[1])*180/3.1415
                list_angles.append(y_angle)
        
            print(list_angles)
            for i in range(len(list_angles)):
                if list_angles[i] < ideal_angle - tolerance or list_angles[i] > ideal_angle + tolerance:
                    if marker_IDs[i] in tf_dict:
                        tf_dict.pop(marker_IDs[i])
                    
                    if i == comm_marker_id - 1:
                        j = 0
                    else:
                        j = i + 1
 
                    if marker_IDs[j] in tf_dict:
                        tf_dict.pop(marker_IDs[j])

        cv2.imshow('frame', frame)
        cv2.waitKey(10)

    print("Done")
    print(tf_dict)

    return tf_dict, list_angles

def main():
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

    dictionary, _ = setup(cam, align,  marker_IDs, num_markers, comm_marker_id)
    
    # store dictionary in pickle
    with open('../pickles/stationary_marker'+str(num_markers)+'.pickle', 'wb') as f:
        pickle.dump(dictionary, f)

if __name__ == "__main__":
    main()

