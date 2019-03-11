import pickle
import numpy as np
import cv2
import cv2.aruco as aruco
import pyrealsense2 as rs
import math
from Realsense import RealSense as Realsense

# rotating about x first
def rotToEuler(rot):
    sy = math.sqrt(rot[0,0] * rot[0,0] + rot[1,0] * rot[1,0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(rot[2,1], rot[2,2])
        y = math.atan2(-rot[2,0], sy)
        z = math.atan2(rot[1,0], rot[0,0])
    else:
        x = math.atan2(-rot[1,2], rot[1,1])
        y = math.atan2(-rot[2,0], sy)
        z = 0

    return np.array([x,y,z])

def eulerToRot(x,y,z):
    R_x = np.array( [ [1, 0, 0],
                      [0, math.cos(x), -math.sin(x)],
                      [0, math.sin(x), math.cos(x)] ])
    R_y = np.array( [ [math.cos(y), 0, math.sin(y) ],
                      [0, 1, 0],
                      [-math.sin(y), 0, math.cos(y) ] ])
    R_z = np.array( [ [math.cos(z), -math.sin(z), 0],
                      [math.sin(z), math.cos(z), 0],
                      [0, 0, 1] ] )
    return np.dot(R_x, np.dot(R_y, R_z))

def eval_acc():
    cam = Realsense()

    objp = np.zeros((3*4,3), np.float32)
    objp[:,:2] = np.mgrid[0:4,0:3].T.reshape(-1,2)
    axis = np.float32([[1,0,0], [0,1,0], [0,0,-1]]).reshape(-1,3)
    objp = np.flip(objp, 0)
    objp = objp * 2.6 / 100.0
#     print(objp)
    
    datapts = 30
    finalDeg = 170
    deg = 10

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
    # from -90 to 90 degrees, increment 5 or 10 degrees
    # collect 30 frames of data once user presses button, then stop
    # let user change to new angle, collect data once user presses button
    # keep track of how many frames we are not able to detect marker
    try:
        while (deg < finalDeg):
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
                frame = cv2.undistort(frame, cam.mtx, cam.dist, None, cam.newcameramtx)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(gray, (4,3), None)

                if ret == True:
                    corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), cam.criteria)
                    img = cv2.drawChessboardCorners(frame, (4,3), corners2, ret)
 
                    _, rvec, tvec = cv2.solvePnP(objp, corners2, cam.newcameramtx, cam.dist)

                    # transform (0,0.026) corner to camera coordinate system
                    # transform marker center to coordinate system of (0,0.026) corner -- 7.2 cm to left
                    rot, _ = cv2.Rodrigues(rvec)
                    # x,y,z = rotToEuler(rot)

                    pt = np.array([-0.072, 0.026, 0])
                    tvec.shape = (1,3)
                    markerPos = rot.dot(pt) + tvec 

                    # rotate about X-axis by 180 degrees, then Z-axis by -90 degrees
                    markerRot = np.dot( rot, eulerToRot(math.pi, 0, math.pi/2) )
                    markerRot, _ = cv2.Rodrigues(markerRot)

                    # find marker using aruco package
                    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
                    parameters = aruco.DetectorParameters_create()

                    # lists of ids and the corners belonging to each id
                    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

                    font = cv2.FONT_HERSHEY_SIMPLEX #font for displaying text (below)
                    if np.all(ids != None):
                        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, 0.05, 
                                              cam.newcameramtx, cam.dist)
            
                        for i in range(0, ids.size):
                            aruco.drawAxis(img, cam.newcameramtx, cam.dist, rvecs[i], tvecs[i], 0.1)
                    else:
                        cv2.putText(img, "No Ids", (0,64), font, 1, (0,255,0),2,cv2.LINE_AA)    
                    
                    cv2.imshow('img', img)
                    cv2.waitKey(1)
                    
                    if np.all(ids != None):
                        rvec = rvecs[0][0]
                        if ((rvec[0] * markerRot[0]) > 0 and (rvec[1] * markerRot[1]) > 0 and (rvec[2] * markerRot[2]) > 0) or ((rvec[0] * markerRot[0] < 0 and (rvec[1] * markerRot[1]) < 0) and (rvec[2] * markerRot[2] < 0)):
                            groundTruth_pos.append(markerPos)
                            calculated_pos.append(tvecs[0])

                            if (rvec[0] * markerRot[0] < 0):
                                rvec = -rvec
                            
                            rotmat, _ = cv2.Rodrigues(rvec)
                            markerRotMat, _ = cv2.Rodrigues(markerRot)
                            nearIdentity = np.matmul(rotmat.transpose(), markerRotMat)
                            # print(nearIdentity)
                            nearIdentity_vec, _ = cv2.Rodrigues(nearIdentity)
                            # print(nearIdentity_vec)
                            vec_item.append(nearIdentity_vec)
                            theta = math.acos( (np.trace(nearIdentity) - 1.0) / 2.0) * 180 / 3.1415
                            print(theta)
                            theta_item.append(theta)
                            
                            detectNum = detectNum + 1
                            print(detectNum)
                        else:
                            print("unstable")
                     
            print("\n")
            pos_truth[deg] = groundTruth_pos
            pos_calc[deg] = calculated_pos
            rot_vec[deg] = vec_item
            rot_theta[deg] = theta_item

            deg = deg + 10
            cv2.destroyAllWindows()

        with open('acc_eval2.pickle', 'wb') as f:
            pickle.dump([pos_truth, pos_calc, rot_vec, rot_theta], f)
    
    finally:
        cam.pipeline.stop()

eval_acc()
