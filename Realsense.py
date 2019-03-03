import pyrealsense2 as rs
import numpy as np
import cv2.aruco as aruco
import cv2

class RealSense:
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.h = 720
        self.w = 1280
        self.config.enable_stream(rs.stream.depth, self.w, self.h, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, self.w, self.h, rs.format.bgr8, 30)

        # Default intrinsic camera parameters
        # self.mtx = np.asarray( [ [644.31, 0, 643.644], [0, 644.31, 352.594], [0, 0, 1] ] )
        # self.dist = np.asarray( [0, 0, 0, 0, 0] )

        # OpenCV calibrated intrinsic parameters - 20 images, RMS: 0.566338583233
        self.mtx = np.asarray( [ [885.88621413, 0, 627.67489027], [0, 889.29330139, 330.51422313], [0, 0, 1] ] )
        self.dist = np.asarray( [0.09543156, -0.3296279, -0.01296923, -0.00432939, 0.18224466] )

        self.newcameramtx, _ = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (self.w,self.h), 1, (self.w,self.h))
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


    def detect_markers_realsense(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
        parameters = aruco.DetectorParameters_create()
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        rvecs = None
        tvecs = None 
        if np.all(ids != None):
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, 0.055, self.newcameramtx, self.dist)

        return corners, ids, rvecs, tvecs

    def access_intrinsics(self):
        cfg = self.pipeline.start()
        profile = cfg.get_stream(rs.stream.depth)
        intr = profile.as_video_stream_profile().get_intrinsics()
        print(intr)

        self.pipeline.stop()

