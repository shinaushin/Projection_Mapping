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
    while (True):
        ret, frame = cap.read()
        # operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
        parameters = aruco.DetectorParameters_create()

        #lists of ids and the corners beloning to each id
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)


        font = cv2.FONT_HERSHEY_SIMPLEX #font for displaying text (below)


        if np.all(ids != None):

            rvec, tvec ,_ = aruco.estimatePoseSingleMarkers(corners, 0.05, mtx, dist) #Estimate pose of each marker and return the values rvet and tvec---different from camera coefficients
            #(rvec-tvec).any() # get rid of that nasty numpy value array error

            for i in range(0, ids.size):
                aruco.drawAxis(frame, mtx, dist, rvec[i], tvec[i], 0.1)  # Draw Axis
            aruco.drawDetectedMarkers(frame, corners) #Draw A square around the markers


            ###### DRAW ID #####
            strg = ''
            for i in range(0, ids.size):
                strg += str(ids[i][0])+', '

            cv2.putText(frame, "Id: " + strg, (0,64), font, 1, (0,255,0),2,cv2.LINE_AA)


	    ###### Output marker  positions in camera frame ######
	    # output rvec and tvec
  	   

        else:
            ##### DRAW "NO IDS" #####
            cv2.putText(frame, "No Ids", (0,64), font, 1, (0,255,0),2,cv2.LINE_AA)

        # Display the resulting frame
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

finally:
    pipeline.stop()