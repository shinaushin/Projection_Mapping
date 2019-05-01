import numpy as np
import cv2
import cv2.aruco as aruco
import pyrealsense2 as rs
from Realsense import RealSense

def main():
    cam = RealSense()

    profile = cam.pipeline.start(cam.config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    align_to = rs.stream.color
    align = rs.align(align_to)

    # Define object points of chessboard corners
    # do not need to scale object points for camera calibration
    objp = np.zeros((3*4,3), np.float32)
    objp[:,:2] = np.mgrid[0:4,0:3].T.reshape(-1,2)
    axis = np.float32([[1,0,0], [0,1,0], [0,0,-1]]).reshape(-1,3)
    objp = np.flip(objp, 0)
    # print(objp)

    detectNum = 0 # how many images have detected chessboard corners
    datapts = 20 # collect 20 images in total
    objpoints = [] # chessboard coordinate system
    imgpoints = [] # camera image coordinate system
    try:
        while (detectNum < datapts):        
            frames = cam.pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            color_image = np.asanyarray(color_frame.get_data())
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

            ret, corners = cv2.findChessboardCorners(gray, (4,3), None)
            if ret == True: # if corners detected
                corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), cam.criteria) # subpixel refinement
                img = cv2.drawChessboardCorners(color_image, (4,3), corners, ret)
                cv2.imshow('img', img)
                retval = cv2.waitKey(10)

                if (retval == ord('p')): # if user presses p, add corners
                    imgpoints.append(corners2)
                    objpoints.append(objp)
                    detectNum = detectNum + 1
                    print("Data point collected")
        
        print("Starting calibration") 
        ret, mtx ,dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        print(ret)
        print(mtx)
        print(dist)
    finally:
        cam.pipeline.stop()
 
if __name__ == "__main__":
    main()
