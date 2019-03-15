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
        self.config.enable_stream(rs.stream.infrared, 1, self.w, self.h, rs.format.y8, 30)
        self.config.enable_stream(rs.stream.infrared, 2, self.w, self.h, rs.format.y8, 30)

        # OpenCV calibrated intrinsic parameters - 20 images, RMS: 0.566338583233
        self.mtx = np.asarray( [ [885.88621413, 0, 627.67489027], [0, 889.29330139, 330.51422313], [0, 0, 1] ] )
        self.dist = np.asarray( [0.09543156, -0.3296279, -0.01296923, -0.00432939, 0.18224466] )
        self.newcameramtx, _ = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (self.w,self.h), 1, (self.w,self.h))

        # OpenCV calibrated intrinsic parameters - 20 images, RMS ~= 0.6        
        self.IRmtx = np.asarray( [ [662.55704262, 0, 652.61212726], [0, 666.17830383, 291.70354751], [0, 0, 1] ] )
        self.IRdist = np.asarray( [-0.11005826, 0.23544585, -0.0173338, -0.00744232, -0.18301952] )
        self.newIRmtx, _ = cv2.getOptimalNewCameraMatrix(self.IRmtx, self.IRdist, (self.w, self.h), 1, (self.w, self.h))

        self.IR_to_RGB = np.asanyarray( [ [0.999894, 0.0145499, -0.000164819, 0.014785], [-0.0145488, 0.99988, 0.0053798, 0.000417606], [0.000243075, -0.00537683, 0.999986, 0.000176072], [0, 0, 0, 1] ] )

        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


    def detect_markers_realsense(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
        parameters = aruco.DetectorParameters_create()

        # detect markers
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        rvecs = []
        tvecs = []
        if np.all(ids != None):
            for i in range(len(ids)):
                # subpixel refinement
                corners2 = cv2.cornerSubPix(gray, np.array(corners[i]), (11,11), (-1,-1), self.criteria)

                # estimate pose of each marker, prone to bad estimations
                rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners2, 0.055, self.newcameramtx, self.dist)
                rvecs.append(rvec)
                tvecs.append(tvec)
        # print(rvecs)
        # print(tvecs)
        return corners, ids, rvecs, tvecs

    def access_intr_and_extr(self):
        cfg = self.pipeline.start()
        color = cfg.get_stream(rs.stream.color)
        intr = color.as_video_stream_profile().get_intrinsics()
        print(intr)
        
        depth = cfg.get_stream(rs.stream.depth)
        intr2 = depth.as_video_stream_profile().get_intrinsics()
        print(intr2)

        extr = depth.as_video_stream_profile().get_extrinsics_to(color.as_video_stream_profile())
        print(extr)

        self.pipeline.stop()

