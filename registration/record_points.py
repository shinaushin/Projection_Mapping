import pickle
import numpy as np
import cv2
import pyrealsense2 as rs
from Realsense import RealSense
import math
import rot_mat_euler_angles_conversion import rotToEuler
import cv2.aruco as aruco

def main():
    num_markers = raw_input("How many markers are there on the tool you are using? ")
    with open('markertool'+str(num_markers)+'.pickle', 'rb') as f:
        x = pickle.load(f)

    cam = Realsense()

    profile = cam.pipeline.start(cam.config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    align_to = rs.stream.color
    align = rs.align(align_to)

    num_pts = raw_input("How many points do you want to record? ")

    marker_IDs = raw_input("What are the IDs of the marker fixed rigidly on the target object, starting in order with the ones on the side and finally the one on the top all separated by a single space? ")
    marker_IDs = [int(x) for x in marker_IDs.split()]

    recorded = 0
    tolerance = 4
    try:
        while (recorded < num_pts):
            frames = cam.pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            color_Frame = aligned_frames.get_color_frame()
            color_image = np.asanyarray(color_frame.get_data())
            frame = color_image
            font = cv2.FONT_HERSHEY_SIMPLEX
            corners, ids, rvecs, tvecs = cam.detect_markers_realsense(frame)

            if np.all(ids != None) and len(ids) > 1:
                for i in range(0, ids.size):
                    aruco.drawAxis(frame, cam.newcameramtx, cam.dist, rvecs[i], tvecs[i], 0.1)
                aruco.drawDetectedMarkers(frame, corners)
                ids.shape = len(ids)
                ids = ids.tolist()

                ids_ord, rvecs_ord, tvecs_ord = [x,y,z for x,y,z in sorted(zip(ids,rvecs,tvecs))]
                acceptable_id_idx = []
                for i in range(len(ids)):
                    j = i + 1
                    if i == len(ids)-1:
                        j = 0
                    


if __name__ == "__main__":
    main()

