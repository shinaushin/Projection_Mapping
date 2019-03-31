import pickle
import numpy as np
import cv2
import cv2.aruco as aruco
import pyrealsense2 as rs
import math
from Realsense import RealSense as Realsense
from rot_mat_euler_angles_conversion import rotToEuler, eulerToRot

def eval_acc():
    cam = Realsense()

    # object points from chessboard coordinate frame
    objp = np.zeros((3*4,3), np.float32)
    objp[:,:2] = np.mgrid[0:4,0:3].T.reshape(-1,2)
    axis = np.float32([[1,0,0], [0,1,0], [0,0,-1]]).reshape(-1,3)
    objp = objp * 2.6 / 100.0
            
    datapts = 30
    finalDeg = 150
    deg = 30

    pos_truth = {}
    rot_vec = {}
    pos_calc = {}
    rot_theta = {}

    profile = cam.pipeline.start(cam.config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    align_to = rs.stream.color
    align = rs.align(align_to)

    # Data Collecting Procedure:
    # from 30 to 150 degrees, increment by 10 degrees
    # collect 30 frames of data once user presses button, then stop
    # let user change to new angle, collect data once user presses button
    try:
        while (deg <= finalDeg):
            print(str(deg) + "\n")
            raw_input('When ready to start taking data, press Enter: ')

            groundTruth_pos = []
            calculated_pos = []
            vec_item = []
            theta_item = []
            detectNum = 0
            while (detectNum < datapts):
                frames = cam.pipeline.wait_for_frames()
                aligned_frames = align.process(frames)
                color_frame = aligned_frames.get_color_frame()
                color_image = np.asanyarray(color_frame.get_data())
                frame = color_image
                frame = cv2.undistort(frame, cam.mtx, cam.dist, None, cam.newcameramtx) # undistort picture based on intrinsics
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # change to grayscale
                ret, corners = cv2.findChessboardCorners(gray, (4,3), None) # corner detection
               
                depth = frames.get_depth_frame()
                if ret == True:
                    corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), cam.criteria) # subpixel refinement
                    corners2 = corners2[::-1] # flip order of corners detected to match coordinate frame ot checkerboard with coordinate frame of camera
                    img = cv2.drawChessboardCorners(frame, (4,3), corners2, ret)
                    _, rvec, tvec, _ = cv2.solvePnPRansac(objp, corners2, cam.newcameramtx, cam.dist) # find pose of chessboard
                    
                    ## transform center of left marker to camera coordinate system
                    rot, _ = cv2.Rodrigues(rvec) # transform from checkerboard origin to camera

                    pt = np.array([-0.073025, 0.0714375, 0]) # relative to origin of chessboard
                    tvec.shape = (1,3)
                    markerPos = rot.dot(pt) + tvec # now in camera coordinate frame
                    groundTruth_pos.append(markerPos[0])
                    
                    # ZYX format
                    markerRot = np.dot(rot, eulerToRot(-math.pi/2, 0, math.pi) ) # now in camera coordinate frame
                    print(rotToEuler(markerRot))
                    markerRot, _ = cv2.Rodrigues(markerRot)
  
                    # find marker using aruco package
                    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
                    parameters = aruco.DetectorParameters_create()

                    # lists of ids and the corners belonging to each id
                    corners, ids, rvecs, tvecs = cam.detect_markers_realsense(frame)

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    if np.all(ids != None):
                        listCorners = corners[0][0]
                        center = np.mean(listCorners, axis=0)
                        dist = depth.get_distance(center[0], center[1])
                        # print(dist)
                        # rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, 0.05, 
                        #                       cam.newcameramtx, cam.dist)
                        # print(tvecs)

                        # for i in range(0, ids.size):
                        #     aruco.drawAxis(img, cam.newcameramtx, cam.dist, rvecs[i], tvecs[i], 0.1)

                        corns = []
                        listCorners = []
                        listCorners.extend(corners[0][0])
                        listCorners.extend(corners[1][0])
                        for corner in listCorners:
                            corns.append(np.array(corner).tolist())
                        corns = np.array(corns)
                        
                        # user-measured corners of two markers on calibration board
                        # two markers = 8 points, more stability in pose estimation
                        objpoints = [ [-0.025, 0.025, 0], [0.025, 0.025, 0], [0.025, -0.025, 0], [-0.025, -0.025, 0], 
                                      [0.1194625, -0.1607375, 0], [0.1694625, -0.1607375, 0], [0.1694625, -0.2107375, 0], [0.1194625, -0.2107375, 0] ]
                        objpoints = np.array(objpoints)

                        _, rvec, tvec, _ = cv2.solvePnPRansac(objpoints, corns, cam.newcameramtx, cam.dist) # pose of left marker
                        aruco.drawAxis(frame, cam.newcameramtx, cam.dist, rvec, tvec, 0.1)
                    else:
                        cv2.putText(img, "No Ids", (0,64), font, 1, (0,255,0),2,cv2.LINE_AA)    
                    
                    cv2.imshow('img', img)
                    cv2.waitKey(10)
                
                    if np.all(ids != None):
                        groundTruth_pos.append(markerPos)

                        x = tvec[0][0]
                        y = tvec[1][0]
                        vec = [x, y, 0]

                        # transform x,y coordinate of left marker to IR camera coordinate
                        # because depth map is in coordinate system of IR camera
                        inv_R = cam.IR_to_RGB[0:3,0:3].transpose()
                        inv_t = np.matmul(-inv_R, cam.IR_to_RGB[0:3,3])
                        vec_IR = np.matmul(inv_R, vec) + inv_t
                        x_IR = vec_IR[0]
                        y_IR = vec_IR[1]
                        # print(dist, x_IR, y_IR)
                        z_IR = np.sqrt( dist**2 - x_IR**2 - y_IR**2  ) # calculate z value using depth,x,y values
                        vec = [0, 0, z_IR, 1] # transform z coordinate back to RGB camera coordinate frame
                        vec_RGB = np.matmul(cam.IR_to_RGB, vec)
                        if dist != 0: # dist can be 0 because depth map has holes
                            tvec[2][0] = vec_RGB[2]

                        tvec = [ tvec[0][0], tvec[1][0], tvec[2][0] ]
                        # print(tvec)
                        calculated_pos.append(tvec)

                        thresh = 0.115 # user-defined threshold to account for noise when looking at marker straight on
                        if rvec[0][0] * markerRot[0][0] < 0 or rvec[1][0] * markerRot[1][0] < 0 or rvec[2][0] * markerRot[2][0] < 0: # if any component differs in sign
                             print(rvec[:,0] - markerRot[:,0])
                             # if value differs by more than threshold, it is not noise and is bad PNP solution
                             if abs(rvec[0][0] - markerRot[0][0]) > thresh or abs(rvec[1][0] - markerRot[1][0]) > thresh or abs(rvec[2][0] - markerRot[2][0]) > thresh:
                              continue # do not append because bad PNP solution
                        
                        rotmat, _ = cv2.Rodrigues(rvec)
                        markerRotMat, _ = cv2.Rodrigues(markerRot)
                        print(rotToEuler(rotmat))
                        print(rotToEuler(markerRotMat))
                        print(rotToEuler(rotmat) - rotToEuler(markerRotMat))
                        nearIdentity = np.matmul(rotmat.transpose(), markerRotMat)
                        print(nearIdentity)
                        nearIdentity_vec, _ = cv2.Rodrigues(nearIdentity)
                        # print(nearIdentity_vec)
                        vec_item.append(nearIdentity_vec)
                        theta = math.acos( (np.trace(nearIdentity) - 1.0) / 2.0) * 180 / 3.1415 # in degrees
                        print(theta)
                        theta_item.append(theta)
                            
                        detectNum = detectNum + 1
                        print(detectNum)
                     
            print("\n")
            pos_truth[deg] = groundTruth_pos
            pos_calc[deg] = calculated_pos
            rot_vec[deg] = vec_item
            rot_theta[deg] = theta_item

            deg = deg + 10
            cv2.destroyAllWindows()

        with open('multi_marker2.pickle', 'wb') as f:
            pickle.dump([pos_truth, pos_calc, rot_vec, rot_theta], f)
    
    finally:
        cam.pipeline.stop()

eval_acc()
