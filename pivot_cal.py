import numpy as np
import cv2
import cv2.aruco as aruco
import pyrealsense2 as rs

pipeline = rs.pipeline()

config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30);

profile = pipeline.start(config)

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

align_to = rs.stream.color
align = rs.align(align_to)

# Intrinsic camera parameters of RealSense camera
mtx = np.asarray( [ [644.31, 0, 643.644], [0, 644.31, 352.594], [0, 0, 1] ] )
dist = np.asarray( [0, 0, 0, 0, 0] )

try:
    # User input of how long to do calibration for
    num_pts = raw_input("How many pivot calibration points do you want to use?")

    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters_create()

    i = 0
    A = []
    b = []
    while (i < num_pts): # Time passed is less than user input
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        frame = color_image

        # operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #lists of ids and the corners beloning to each id
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        font = cv2.FONT_HERSHEY_SIMPLEX #font for displaying text (below)
        if np.all(ids != None):
            rvec, tvec ,_ = aruco.estimatePoseSingleMarkers(corners, 0.05, mtx, dist) #Estimate pose of each marker and return the values rvet and tvec---different from camera coefficients
            #(rvec-tvec).any() # get rid of that nasty numpy value array error

            rotMat, _  = cv2.Rodrigues(rvec)
            # ...

        i = i + 1;

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    # Solve least-squares Ax = b
    x = np.linalg.lstsq(A,b) # x[0:2] = p_t, x[3:5] = p_pivot

finally:
    pipeline.stop()
