# record_points.py
# author: Austin Shin

import sys
sys.path.append('../')

import cv2
import cv2.aruco as aruco
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pickle

import pyrealsense2 as rs
from Realsense import RealSense as Realsense

from rot_mat_euler_angles_conversion import rotToEuler

def find_best_marker(ids_ord, rvecs_ord, tvecs_ord, num_markers):
    """
    Finds most accurately detected marker

    Args:
        ids_ord: ordered id list of detected markers
        rvecs_ord: rotation vectors in order of detected marker ids
        tvecs_ord: translation vectors in order of detected marker ids
        num_markers: number of markers on digitizer

    Returns:
        best marker ID
    """
    tolerance = 4 # degs; set by user
    ideal_angle = 360 / (num_markers-1)

    idx = -1 
    for i in range(len(ids_ord)-1):
        if ids_ord[i+1] - ids_ord[i] > 1:
             idx = i

    if idx != -1:
        iters = len(ids_ord)-idx+1
        last = len(ids_ord)-1
        for i in range(iters):
            ids_ord.insert(0, ids_ord.pop(last))
            rvecs_ord.insert(0, rvecs_ord.pop(last))
            tvecs_ord.insert(0, tvecs_ord.pop(last))

    # print(ids_ord)
    acceptable_id_idx = []
    for i in range(len(ids_ord)-1):
        marker1_id = ids_ord[i]
        marker2_id = ids_ord[i+1]
        if marker1_id == num_markers-2:
            marker2_id = marker2_id + num_markers-1

        if marker2_id - marker1_id == 1:
            marker1 = rvecs_ord[i]
            marker1_rot, _ = cv2.Rodrigues(marker1[0])
            marker2 = rvecs_ord[i+1]
            marker2_rot, _ = cv2.Rodrigues(marker2[0])
            marker1_rot_T = marker1_rot.transpose()
            rot_btw_1_2 = np.matmul(marker1_rot_T, marker2_rot)
            angles = rotToEuler(rot_btw_1_2)
            angles = angles * 180 / 3.1415
            # print(angles)
            y_angle = angles[1]
            if y_angle < ideal_angle + tolerance and \
                    y_angle > ideal_angle - tolerance:
                if i not in acceptable_id_idx:
                    acceptable_id_idx.append(i)
                if i+1 not in acceptable_id_idx:
                    acceptable_id_idx.append(i+1)
            else:
                if i in acceptable_id_idx:
                    idx = acceptable_id_idx.index(i)
                    acceptable_id_idx.pop(idx)
                if i+1 in acceptable_id_idx:
                    idx = acceptable_id_idx.index(i+1)
                    acceptable_id_idx.pop(idx)

    # print(acceptable_id_idx)
    length = len(acceptable_id_idx)
    if length > 0:
        index = 0
        if length > 1:
            total = 0
            for i in range(length):
                rot1, _ = cv2.Rodrigues(rvecs_ord[acceptable_id_idx[i]])
                angle_set = np.absolute(rotToEuler(rot1))
                temp = np.sum(angle_set)
                if temp > total:
                    index = i
                    total = temp
        return acceptable_id_idx[index]
    return -1

def calc_pt(corners, ids, rvecs, tvecs, frame, cam, num_markers,
        num_stat_markers, tf_dict, stat_tf_dict, tip):
    """
    Transforms point on digitizer to be in frame of base marker on static
    marker group

    Args:
        corners: corners of detected markers
        ids: ids of detected markers
        rvecs: rotation vectors of digitizer markers
        tvecs: translation vectors of digitizer markers
        frame: image
        cam: Realsense parameter
        num_markers: number of markers on digitizer
        num_stat_markers: number of markers on static marker tool
        tf_dict: transformation dictionary between adjacent markers for digitizer
        stat_tf_dict: transformation dictionary between adjacent markers for
            static marker tool
        tip: tip point relative to digitizer coordinate frame

    Returns:
        point in coordinate frame of base marker of static marker tool
    """
    if np.all(ids != None) and len(ids) >= 4:
        for i in range(0, ids.size):
            aruco.drawAxis(frame, cam.newcameramtx, cam.dist, rvecs[i], tvecs[i], 0.1)
        aruco.drawDetectedMarkers(frame, corners)
        ids.shape = len(ids)
        ids = ids.tolist()

        arr = [x for x in sorted(zip(ids,rvecs,tvecs))]
        # print(arr)
        tot_ids_ord = [item[0] for item in arr]
        tot_rvecs_ord = [item[1] for item in arr]
        tot_tvecs_ord = [item[2] for item in arr]

        threshold = -1
        for i in range(len(tot_ids_ord)):
            if tot_ids_ord[i] >= num_markers:
                threshold = i
                break

        if threshold == -1:
            print("Stationary marker on target object is not in frame, could not record point")
            return []
        elif threshold == 0:
            print("Marker tool is not in frame, could not calculate point")
            return []
        else:
            ids_ord = tot_ids_ord[:threshold]
            rvecs_ord = tot_rvecs_ord[:threshold]
            tvecs_ord = tot_tvecs_ord[:threshold]

            stat_ids_ord = tot_ids_ord[threshold:]
            stat_rvecs_ord = tot_rvecs_ord[threshold:]
            stat_tvecs_ord = tot_tvecs_ord[threshold:]

            if len(ids_ord) < 2:
                print("Not enough markers detected on marker tool")
                return []
            elif len(stat_ids_ord) < 2:
                print("Not enough markers detected on stationary marker on target object")
                return []
            else:
                best_index = find_best_marker(ids_ord, rvecs_ord, tvecs_ord,
                    num_markers)
                best_stat_index = find_best_marker(stat_ids_ord, stat_rvecs_ord,
                    stat_tvecs_ord, num_stat_markers)
        
                if best_index != -1 and best_stat_index != -1:
                    # transform end tip from marker tool frame to camera frame
                    marker = ids_ord[best_index]
                    r_tf, t_tf = tf_dict[marker]
                    t_tf.shape = (3,1)
                    rot, _ = cv2.Rodrigues(r_tf)
                    rot_T = rot.transpose()
                    inv_t = np.matmul(-rot_T, t_tf)
                    pt_marker_i = np.matmul(rot_T, tip) + inv_t
                    # pt_marker_i.shape = (3,1)

                    # transform from marker i frame to camera frame
                    rot, _ = cv2.Rodrigues(rvecs_ord[best_index][0][0])
                    t = tvecs_ord[best_index][0][0]
                    t.shape = (3,1)
                    pt_cam = np.matmul(rot, pt_marker_i) + t 

                    # transform point from camera frame to stationary marker frame
                    rot, _ = cv2.Rodrigues(stat_rvecs_ord[best_stat_index][0][0])
                    t = stat_tvecs_ord[best_stat_index][0][0]
                    t.shape = (3,1)
                    rot_T = rot.transpose()
                    inv_T = np.matmul(-rot_T, t)

                    pt_stat_marker_i = np.matmul(rot_T, pt_cam) + inv_T

                    stat_marker = stat_ids_ord[best_stat_index]
                    r_tf, t_tf = stat_tf_dict[stat_marker]
                    t_tf.shape = (3,1)
                    rot, _ = cv2.Rodrigues(r_tf)
                    pt_stat_comm_marker = np.matmul(rot, pt_stat_marker_i) + \
                        t_tf

                    return pt_stat_comm_marker
                else:
                    print("Bad orientations calculated")
                    return []
    else:
        print("Not enough markers detected")
        return []

def plot_pts(corners, ids, rvecs, tvecs, frame, cam, num_markers, tf_dict, tip):
    """
    Calculates non-base marker on digitizer to be with respect to camera frame

    Args:
        corners: corners of detected markers on digitizer
        ids: detected marker IDs
        rvecs: rotation vectors of detected markers
        tvecs: translation vectors of detected markers
        frame: image
        cam: Realsense parameter
        num_markers: number of markers on digitizer
        tf_dict: transformation dictionary for adjacent markers on digitizer
        tip: point of tip relative to digitizer frame

    Returns:
        point with respect to camera frame
    """
    if np.all(ids != None) and len(ids) >= 2:
        for i in range(0, ids.size):
            aruco.drawAxis(frame, cam.newcameramtx, cam.dist, rvecs[i],
                tvecs[i], 0.1)
        aruco.drawDetectedMarkers(frame, corners)
        ids.shape = len(ids)
        ids = ids.tolist()

        arr = [x for x in sorted(zip(ids,rvecs,tvecs))]
        # print(arr)
        ids_ord = [item[0] for item in arr]
        rvecs_ord = [item[1] for item in arr]
        tvecs_ord = [item[2] for item in arr]

        if len(ids_ord) < 2:
            print("Not enough markers detected on marker tool")
            return []
        else:
            best_index = find_best_marker(ids_ord, rvecs_ord, tvecs_ord,
                num_markers)

            if best_index != -1:
                # transform end tip from marker tool frame to camera frame
                marker = ids_ord[best_index]
                r_tf, t_tf = tf_dict[marker]
                t_tf.shape = (3,1)
                rot, _ = cv2.Rodrigues(r_tf)
                rot_T = rot.transpose()
                inv_t = np.matmul(-rot_T, t_tf)
                pt_marker_i = np.matmul(rot_T, tip) + inv_t
                # pt_marker_i.shape = (3,1)

                # transform from marker i frame to camera frame
                rot, _ = cv2.Rodrigues(rvecs_ord[best_index][0][0])
                t = tvecs_ord[best_index][0][0]
                t.shape = (3,1)
                pt_cam = np.matmul(rot, pt_marker_i) + t
             
                return pt_cam
            else:
                print("Bad orientations calculated")
                return []
    else:
        print("Not enough markers detected")
        return []
             

def main(onlyDisplay):
    """
    Begins point recording process

    Args:
        onlyDisplay: if not 0, then stores data as pickle files

    Returns:
        None
    """
    num_markers = raw_input("How many markers are there on the tool you are using? ")
    num_markers = int(num_markers)
    marker_IDs = range(num_markers)

    with open('../pivot_cal_markertool'+str(num_markers)+'.pickle', 'rb') as f:
        x = pickle.load(f)

    with open('../markertool'+str(num_markers)+'.pickle', 'rb') as f:
        tf_dict = pickle.load(f)
    # print(tf_dict)

    tip = np.array(x)
    tip.shape = (3,1)
    cam = Realsense()

    profile = cam.pipeline.start(cam.config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    align_to = rs.stream.color
    align = rs.align(align_to)

    num_pts = raw_input("How many points do you want to record? ")
    num_pts = int(num_pts)

    if onlyDisplay == "0":
        stationary_marker_IDs = raw_input("What are the IDs of the marker fixed rigidly on the target object, starting in order with the ones on the side and finally the one on the top all separated by a single space? ")
        stationary_marker_IDs = [int(x) for x in stationary_marker_IDs.split()]
        num_stat_markers = len(stationary_marker_IDs)
        stat_ideal_angle = 360 / (num_stat_markers-1)

        with open('../stationary_marker'+str(num_stat_markers)+'.pickle', 'rb') as f:
            stat_tf_dict = pickle.load(f)

    recorded = 0
    tolerance = 4
    pts = []
    try:
        while (recorded < num_pts):
            frames = cam.pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            color_image = np.asanyarray(color_frame.get_data())
            frame = color_image
            font = cv2.FONT_HERSHEY_SIMPLEX

            cv2.imshow('frame', frame)
            userinput = cv2.waitKey(10)
            if userinput & 0xFF == ord('p'):
                corners, ids, rvecs, tvecs = cam.detect_markers_realsense(frame)

                if onlyDisplay == "0":
                    pt_stat_comm_marker = calc_pt(corners, ids, rvecs, tvecs,
                        frame, cam, num_markers, num_stat_markers, tf_dict,
                        stat_tf_dict, tip)

                    if len(pt_stat_comm_marker) != -0:
                        pts.append(pt_stat_comm_marker)
                        recorded = recorded + 1
                        print("Recorded: " + str(recorded))
                else:
                    pt = plot_pts(corners, ids, rvecs, tvecs, frame, cam,
                        num_markers, tf_dict, tip)
                    if len(pt) != 0:
                        pts.append(pt)
                        recorded = recorded + 1
                        print("Recorded: " + str(recorded))

    finally:
        cam.pipeline.stop()

    # plot pts
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    pts = np.array(pts)
    ax.scatter(pts[:,0], pts[:,1], pts[:,2])
    print(pts)
    # plt.show()

    return pts

if __name__ == "__main__":
    onlyDisplay = sys.argv[1]
    pts = main(onlyDisplay)

    if onlyDisplay == "0":
        target_obj = 'skull1'
        with open('../pts_stat_marker_frame_'+target_obj+'.pickle', 'wb') as f:
            pickle.dump(pts, f)

    plt.show()
