import pickle
import numpy as np
import cv2
import cv2.aruco as aruco
import pyrealsense2 as rs
import math
from Realsense import RealSense as Realsense

"""
From IR camera coordinate to RGB camera coordinate:
TF = [	0.999894	0.0145499	-0.000164819	0.014785
	-0.0145488	0.99988		0.0053798	0.000417606
	0.000243075	-0.00537683	0.999986	0.000176072
	0		0		0		1		]
"""

# rotating about x firstr
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
    rot_IR_RGB = np.asanyarray( [ [0.999894, 0.0145499, -0.000164819], [-0.0145488, 0.99988, 0.0053798], [0.000243075, -0.00537683, 0.999986] ] )
    rotvec_IR_RGB, _ = cv2.Rodrigues(rot_IR_RGB)
    rotvec_IR_RGB.shape = (3)

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
    rot_truth = {}
    pos_calc = {}
    rot_calc = {}

    profile = cam.pipeline.start(cam.config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    align_to = rs.stream.color
    align = rs.align(align_to)

    # Data Collecting Procedure:
    # from 10 to 160, increment 10 degrees
    # collect 30 frames of data once user presses button, then stop
    # let user change to new angle, collect data once user presses button
    # keep track of how many frames we are not able to detect marker
    try:
        while (deg < finalDeg):
            print(str(deg) + "\n")
            raw_input('When ready to start taking data, press Enter: ')

            groundTruth_pos = []
            calculated_pos = []
            groundTruth_rot = []
            calculated_rot = []
            detectNum = 0
            while (detectNum < datapts):
                frames = cam.pipeline.wait_for_frames()
                aligned_frames = align.process(frames)
                color_frame = aligned_frames.get_color_frame()
                color_image = np.asanyarray(color_frame.get_data())
                color_image = cv2.undistort(color_image, cam.mtx, cam.dist, None, cam.newcameramtx)
                gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
                ret_col, corners_col = cv2.findChessboardCorners(gray, (4,3), None)

                ir1_frame = frames.get_infrared_frame(1)
                ir1_image = np.asanyarray(ir1_frame.get_data())
                ret_IR, corners_IR = cv2.findChessboardCorners(ir1_image, (4,3), None)

                if ret_col == True and ret_IR == True:
                    corners2_col = cv2.cornerSubPix(gray, corners_col, (11,11), (-1,-1), cam.criteria)
                    color_img = cv2.drawChessboardCorners(color_image, (4,3), corners2_col, ret_col)
 
                    _, rvec_col, tvec_col = cv2.solvePnP(objp, corners2_col, cam.newcameramtx, cam.dist)

                    # transform (0,0.026) corner to camera coordinate system
                    # transform marker center to coordinate system of (0,0.026) corner -- 7.2 cm to left
                    rot_col, _ = cv2.Rodrigues(rvec_col)
                    pt = np.array([-0.072, 0.026, 0])
                    tvec_col.shape = (3)
                    markerPos_col = rot_col.dot(pt) + tvec_col 

                    # rotate about X-axis by 180 degrees, then Z-axis by -90 degrees
                    markerRot_col = np.dot( rot_col, eulerToRot(math.pi, 0, math.pi/2) )
                    markerVec_col, _ = cv2.Rodrigues(markerRot_col)
                    markerVec_col.shape = (3)

                    # same process with IR image
                    corners2_IR = cv2.cornerSubPix(ir1_image, corners_IR, (11,11), (-1,-1), cam.criteria)
                    IR_img = cv2.drawChessboardCorners(ir1_image, (4,3), corners2_IR, ret_IR)
 
                    _, rvec_IR, tvec_IR = cv2.solvePnP(objp, corners2_IR, cam.newIRmtx, cam.IRdist)

                    rot_IR, _ = cv2.Rodrigues(rvec_IR)
                    tvec_IR.shape = (3)
                    markerPos_IR = rot_IR.dot(pt) + tvec_IR
 
                    markerRot_IR = np.dot(rot_IR, eulerToRot(math.pi, 0, math.pi/2) )
                    markerVec_IR, _ = cv2.Rodrigues(markerRot_IR)
                    markerVec_IR.shape = (3)

                    # find marker using aruco package
                    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
                    parameters = aruco.DetectorParameters_create()

                    # lists of ids and the corners belonging to each id
                    corners_markers, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

                    font = cv2.FONT_HERSHEY_SIMPLEX #font for displaying text (below)
                    if np.all(ids != None):
                        rvecs_markers, tvecs_markers, _ = aruco.estimatePoseSingleMarkers(corners_markers, 0.05, 
                                              cam.newcameramtx, cam.dist)
            
                        for i in range(0, ids.size):
                            aruco.drawAxis(color_img, cam.newcameramtx, cam.dist, rvecs_markers[i], tvecs_markers[i], 0.1)
                    else:
                        cv2.putText(color_img, "No Ids", (0,64), font, 1, (0,255,0),2,cv2.LINE_AA)    
                    
                    cv2.imshow('img', color_img)
                    cv2.waitKey(1)
                    
                    if np.all(ids != None):
                        rvec = rvecs_markers[0][0]
                        markerRot, _ = cv2.Rodrigues(rvec)
                        markerRot_IR_T = markerRot_IR.transpose()
                        markerRot_total = np.matmul(markerRot_IR_T, markerRot)
                        markerVec_total, _ = cv2.Rodrigues(markerRot_total)
                        markerVec_total.shape = (3)
                        diff = min( np.linalg.norm(np.subtract(markerVec_total, rotvec_IR_RGB)), 
                                    np.linalg.norm(np.subtract(-markerVec_total, rotvec_IR_RGB)) )
                        # print(diff)

                        nearIdentity = markerRot_total.transpose() * rot_IR_RGB
                        nearIdentity_vec, _ = cv2.Rodrigues(nearIdentity)
                        nearIdentity_vec.shape = (3)
                        print(nearIdentity)
                        theta = math.acos( (np.trace(nearIdentity) - 1.0) / 2.0)
                        print(theta*180/3.1415)                       
 
                        detectNum = detectNum + 1
                        print(detectNum)
                     
            print("\n")
            pos_truth[deg] = groundTruth_pos
            pos_calc[deg] = calculated_pos
            rot_truth[deg] = groundTruth_rot
            rot_calc[deg] = calculated_rot

            deg = deg + 10
            cv2.destroyAllWindows()

        with open('acc_eval2.pickle', 'wb') as f:
            pickle.dump([pos_truth, pos_calc, rot_truth, rot_calc], f)
    
    finally:
        cam.pipeline.stop()

eval_acc()
