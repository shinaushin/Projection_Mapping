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

# Intrinsic camera parameters of RealSense camera
mtx = np.asarray( [ [644.31, 0, 643.644], [0, 644.31, 352.594], [0, 0, 1] ] )
dist = np.asarray( [0.0, 0.0, 0.0, 0.0, 0.0] )

try:
    while (True):
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())        
        frame = color_image

        # operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
        parameters = aruco.DetectorParameters_create()

        #lists of ids and the corners belonging to each id
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)


        font = cv2.FONT_HERSHEY_SIMPLEX #font for displaying text (below)


        if np.all(ids != None):

            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, 0.05, mtx, dist) #Estimate pose of each marker and return the values rvet and tvec---different from camera coefficients
            #(rvec-tvec).any() # get rid of that nasty numpy value array error

            for i in range(0, ids.size):
                aruco.drawAxis(frame, mtx, dist, rvecs[i], tvecs[i], 0.1)  # Draw Axis
            aruco.drawDetectedMarkers(frame, corners) #Draw A square around the markers


            ###### DRAW ID #####
            strg = ''
            for i in range(0, ids.size):
                strg += str(ids[i][0])+', '

            cv2.putText(frame, "Id: " + strg, (0,25), font, 1, (0,255,0),2,cv2.LINE_AA)


	    ###### Output marker  positions in camera frame ######
 	    # output tvec
            y0 = 60
            dy = 40
            for i in range(0, ids.size):
                y = y0 + i*dy
                cv2.putText(frame, str(tvecs[i][0]), (0, y), font, 1, (0,255,0), 2, cv2.LINE_AA)

        else:
            ##### DRAW "NO IDS" #####
            cv2.putText(frame, "No Ids", (0,64), font, 1, (0,255,0),2,cv2.LINE_AA)

        # Display the resulting frame
        cv2.imshow('frame',frame)
        if cv2.waitKey(10) == ord('p'):
            while True:
                if cv2.waitKey(0):
                    break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

finally:
    pipeline.stop()
