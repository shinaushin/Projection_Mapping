import sys
sys.path.append('../')

import pickle
import numpy as np
import cv2
import pyrealsense2 as rs
from Realsense import RealSense as Realsense
import math
from rot_mat_euler_angles_conversion import rotToEuler
import cv2.aruco as aruco

from functools import partial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pycpd import rigid_registration
import time

from stl import mesh
from record_points import find_best_marker

def main():
    cam = Realsense()
    profile = cam.pipeline.start(cam.config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    align_to = rs.stream.color
    align = rs.align(align_to)

    target_obj = 'skull'
    with open('../pts_stat_marker_frame_'+target_obj+'.pickle', 'rb') as f:
        pts = pickle.load(f)
        pts = np.asarray(pts, dtype=np.float32)
        pts.shape = (10,3)
        # print(pts)

    # make sure this is same marker tool used in record_points.py
    num_markers = 7
    with open('../stationary_marker'+str(num_markers)+'.pickle', 'rb') as f:
        tf_dict = pickle.load(f)     
    print(tf_dict)

    # load target obj
    my_mesh = mesh.Mesh.from_file('data/CRANIAL HEADS_Head_1_001_centered.stl')

    all_pts = []
    for vector in my_mesh.vectors:
        all_pts.extend(vector / 1000)

    all_pts = np.array(all_pts)
    num = all_pts.size/3
    indices = np.random.choice(num, 1200)
    sub_pts = all_pts[indices,:]
    sub_pts = np.asarray(sub_pts, dtype=np.float32)
                                                                                       
    try:
        while (True):
            frames = cam.pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            color_image = np.asanyarray(color_frame.get_data())
            frame = color_image
            font = cv2.FONT_HERSHEY_SIMPLEX

            cv2.imshow('frame', frame)
            userinput = cv2.waitKey(10)

            corners, ids, rvecs, tvecs = cam.detect_markers_realsense(frame)

            if np.all(ids != None) and len(ids) >= 2:
                for i in range(0, ids.size):
                    aruco.drawAxis(frame, cam.newcameramtx, cam.dist, rvecs[i], tvecs[i], 0.1)
                aruco.drawDetectedMarkers(frame, corners)
                ids.shape = len(ids)
                ids = ids.tolist()

                arr = [x for x in sorted(zip(ids,rvecs,tvecs))]
                # print(arr)
                ids_ord = [item[0] for item in arr]
                rvecs_ord = [item[1] for item in arr]
                tvecs_ord = [item[2] for item in arr]

                best_index = find_best_marker(ids_ord, rvecs_ord, tvecs_ord, num_markers)
                print(best_index)
                pts_cam_frame = []
                if best_index != -1 and ids_ord[best_index] != 13: # no way to filter bad orientation for top marker
                    marker = ids_ord[best_index]
                    r_tf, t_tf = tf_dict[marker]
                    t_tf.shape = (3,1)
                    rot1, _ = cv2.Rodrigues(r_tf)
                    rot1_T = rot1.transpose()
                    inv_t1 = np.matmul(-rot1_T, t_tf)
                    
                    rot2, _ = cv2.Rodrigues(rvecs_ord[best_index][0][0])
                    t2 = tvecs_ord[best_index][0][0]
                    t2.shape = (3,1)
                    
                    for pt in pts:
                        pt.shape = (3,1)
                        new_pt = np.matmul(rot1_T, pt) + inv_t1
                        pts_cam_frame.append( np.matmul(rot2, new_pt) + t2 )

                    # do registration
                    print("Doing registration")
                    pts_cam_frame = np.asarray(pts_cam_frame, dtype=np.float32)
                    pts_cam_frame.shape = (10,3)
                    reg = rigid_registration(**{ 'X': pts_cam_frame, 'Y':sub_pts})
                    TY, (s_reg, R_reg, t_reg) = reg.register()
                    print("Rotation:")
                    print(reg.R)
                    print("Translation:")
                    print(reg.t)
                    print("\n")

    finally:
        cam.pipeline.stop()

if __name__ == "__main__":
    main()

