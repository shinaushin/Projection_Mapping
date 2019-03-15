import numpy as np
import cv2
import cv2.aruco as aruco
import pyrealsense2 as rs
from Realsense import RealSense as Realsense


def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (0,0,255), 5)
    return img

cam = Realsense()
# cam.access_intr_and_extr()
profile = cam.pipeline.start(cam.config)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
align_to = rs.stream.color
align = rs.align(align_to)

objp = np.zeros((3*4,3), np.float32)
objp[:,:2] = np.mgrid[0:4,0:3].T.reshape(-1,2)
axis = np.float32([[1,0,0], [0,1,0], [0,0,-1]]).reshape(-1,3)
# print(objp)

try:
    while (True):
        # detect ArUco markers in RGB images
        frames = cam.pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())        
        frame = color_image
        font = cv2.FONT_HERSHEY_SIMPLEX
        corners, ids, rvecs, tvecs = cam.detect_markers_realsense(frame)
        
        if np.all(ids != None): # if markers are detected
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

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (4,3), None)
        if ret == True:
            corners2 = cv2.cornerSubPix(gray, corners,(11,11), (-1,-1), cam.criteria)
            corners2 = corners2[::-1]
            # print(corners2)
            # print(objp)
            frame = cv2.drawChessboardCorners(frame, (4,3), corners2, ret)
            # Find the rotation and translation vectors.
            _, rvecs, tvecs = cv2.solvePnP(objp, corners2, cam.newcameramtx, cam.dist)
            rot, _ = cv2.Rodrigues(rvecs)
            # print(rot)
            # project 3D points to image plane
            imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, cam.newcameramtx, cam.dist)
            frame = draw(frame, corners2, imgpts)

        # Display the resulting frame
        cv2.imshow('frame',frame)
        cv2.waitKey(5)

    # When everything done, release the capture
    cv2.destroyAllWindows()

finally:
    cam.pipeline.stop()
