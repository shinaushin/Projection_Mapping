import pyrealsense2 as rs
import numpy as np
import cv2.aruco as aruco
import cv2

class RealSense:
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        self.mtx = np.asarray( [ [644.31, 0, 643.644], [0, 644.31, 352.594], [0, 0, 1] ] )
        self.dist = np.asarray( [0, 0, 0, 0, 0] )

    def detect_markers_realsense(self):
        profile = self.pipeline.start(self.config)
        depth_sensor = self.profile.get_device().first_depth_sensor()
        depth_scale = self.depth_sensor.get_depth_scale()
        align_to = rs.stream.color
        align = align(self.align_to)
        
        try:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            color_image = np.asanyarray(color_frame.get_data())
            frame = color_image

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
            corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
 
            font = cv2.FONT_HERSHEY_SIMPLEX
            if np.all(ids != None):
                rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, 0.055, self.mtx, self.dist)

            return ids, rvecs, tvecs

        finally:
            self.pipeline.stop()

