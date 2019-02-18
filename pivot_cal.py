import numpy as np
import cv2
import cv2.aruco as aruco
import pyrealsense2 as rs


pipeline = rs.pipeline()

config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color(640, 480, rs.format.bgr8, 30);

profile = pipeline.start(config)

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

align_to = rs.streamcolor
align = rs.align(align_to)

# Intrinsic camera parameters of RealSense camera
mtx = [ [644.31, 0, 643.644], [0, 644.31, 352.594], [0, 0, 1]]
dist = [0, 0, 0, 0, 0]

try:
    # User input of how long to do calibration for

    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters_create()

    while (): # Time passed is less than user input
        ret, frame = cap.read()
        # operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #lists of ids and the corners beloning to each id
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        font = cv2.FONT_HERSHEY_SIMPLEX #font for displaying text (below)
        if np.all(ids != None):
            rvec, tvec ,_ = aruco.estimatePoseSingleMarkers(corners, 0.05, mtx, dist) #Estimate pose of each marker and return the values rvet and tvec---different from camera coefficients
            #(rvec-tvec).any() # get rid of that nasty numpy value array error

            # ...

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

finally:
    pipeline.stop()
