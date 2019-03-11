import numpy as np
import cv2
import cv2.aruco as aruco
import pyrealsense2 as rs
from Realsense import RealSense as Realsense


cam = Realsense()
cam.access_intr_and_extr()
profile = cam.pipeline.start(cam.config)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
align_to = rs.stream.color
align = rs.align(align_to)

try:
    while (True):
        frames = cam.pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())        
        frame = color_image
        font = cv2.FONT_HERSHEY_SIMPLEX
        corners, ids, rvecs, tvecs = cam.detect_markers_realsense(frame)
        if np.all(ids != None):

            for i in range(0, ids.size):
                aruco.drawAxis(frame, cam.newcameramtx, cam.dist, rvecs[i], tvecs[i], 0.1)  # Draw axis
            aruco.drawDetectedMarkers(frame, corners) #Draw a square around the markers

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
    cv2.destroyAllWindows()

finally:
    cam.pipeline.stop()
