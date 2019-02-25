# marker is 7.2 cm to the left of left middle checkerboard corner

import numpy as np
import cv2
import cv2.aruco as aruco
import pyrealsense2 as rs

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
profile = pipeline.start(config)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
align_to = rs.stream.color
align = rs.align(align_to)

mtx = np.asaray( [644.31, 0, 643.644], [0, 644.31, 352.594], [0, 0, 1] ] )
dist = np.asarray( [0.0, 0.0, 0.0, 0.0, 0.0] )
h = 720
w = 1280
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((3*4,3), np.float32)
objp[:,:2] = np.mgrid[0:4,0:3].T.reshape(-1,2)
axis = np.float32([[1,0,0], [0,1,0], [0,0,-1]]).reshape(-1,3)

# Data Collecting Procedure:
# from -90 to 90 degrees, increment 5 or 10 degrees
# collect 30 frames of data once user presses button, then stop
# let user change to new angle, collect data once user presses button
# keep track of how many frames we are not able to detect marker
try:
    while (True):
        frames = pipeline.wait_for_frames()
        aligned_Frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        frame = color_image
        frame = cv2.undistort(frame, mtx, dist, None, newcameramtx)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (4,3), None)

        if ret == True:
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            ret, rvec, tvec = cv2.solvePnP(objp, corners2, newcameramtx, dist)

            # transform (2,1) corner to camera coordinate system
            # transform marker center to coordinate system of (2,1) corner -- 7.2 cm to left

            # print location of marker center and orientation of marker axes (comp of rots)

            # find marker using aruco package
            # compare 3d position and orientation, calculate error and std dev
 
