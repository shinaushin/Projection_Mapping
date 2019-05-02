# Projection_Mapping

Development of a projection mapping prototype that will project important patient data such as CT scans or MRI scans onto patient bodies in realtime. 

## Documentation

Unless otherwise stated, all use cases for python files are simply: python <file_name.py>

Also, always run files while in the directory that the file is in because Python imports depend on the directory in which you run the file.

### Archive/

Not significant

Old files that are not used any more

### heart_data/

Files related to testing registration using heart model in lab

Old files that are not used any more

### IR_calibration.py

Python script used to calibrate infrared camera and will print intrinsics of camera (camera matrix and distortion coefficients)

This script will work for any 4x3 checkerboard (size can be changed in code) and will collect 20 different images (number of images used can also be changed in code) before conducted the calibration using OpenCV. This script does not save the images.

### marker_detection_realsense.py

Detects markers in images from Realsense camera

Draws marker coordinate frame on it

Writes marker 3D coordinate in camera coordinate frame on image

Draws checkerboard corners detected on image (4x3 checkerboard -- can be changed in code)

### marker_tool_setup/

#### pickles/markertool*.pickle

pickle is a data format used by Python to easily serialize and deserialize information and is the main way I use to save data that I want to use for later at any point in time.

For each marker tool, we specify the frame of the top marker as the overall marker tool coordinate frame.

This pickle file contains the transformation of every other marker to that top marker

#### scripts/marker_setup.py

python script to calculate transformation from each marker on the side of the marker tool to the top marker.

Saves all transformations into a pickle file.

Verifies that transformations are reasonable by making the angle of rotation between any two consecutive markers is within a certain threshold that can be set by the user.

For example, suppose the marker tool used had 7 markers. This means there is 1 marker on top and 6 markers on the side. The tool is therefore in a hexagonal shape. Therefore the angle of rotation for any two consecutive markers should be approximately 60 degrees.

#### scripts/marker_setup_test.py

python script to evaluation accuracy of marker_setup.py

Creates box plot and mean/stddev plot of error in saved transformations

User can define how many marker setup procedures to conduct before doing the error analysis

Saves plot data in a pickle file

### OpenCV_pose_acc_eval/

#### pickles/acc_eval*.pickle OR multi_marker.pickle

Contain data from the marker pose estimation accuracy evaluations conducted to visualize the position and orientation error

#### plots/marker_det_plots_bad_rot_metric/

plots of results of marker pose estimation using faulty metric for orientation error

#### plots/marker_det_plots_one_marker/

plots of results of marker pose estimation using only one marker with corrected error metrics

#### plots/marker_det_plots_two_markers/

plots of results of marker pose estimation using two markers with corrected error metrics

#### scripts/data_viz*.py
Use case: python data_viz*.py "<name of pickle to deserialize>" "<name of folder to put plots in>"
  
Deserializes pickle containing all data collected from pose_est_accuracy_eval.py

Plots box plots and mean/stddev plots for x,y,z,mag error in position and quaternion,theta error in orientation. Assumes that calibration is done in increments of 10 deg from 30 to 150 degrees (can be changed in code)

#### scripts/pose_est_accuracy_eval.py

Calibration procedure is done in increments of 10 deg from 30 to 150 deg. There are two markers (marker1 and marker2) set coplanar to a 4x3 checkerboard. The transformation among the markers and the checkerboard are all known. The "ground truth" transformation between the camera and marker1 is calculated through the checkerboard because we assume that the error in detecting the checkerboard is very small from the camera calibration. The mesaured/observed transformation from the camera frame to the marker1 frame is computed through solvepnp using 8 points (4 corners of marker1 and 4 additional corners from marker2 transformed to frame of marker1).

### pivot_cal/

#### scripts/pivot_cal.py

Performs pivot calibration of user-specified marker tool

Outputs position of tip of marker tool relative to origin of marker tool in marker tool coordinate frame

Saves that position in pickle file

Dimensions: hexagon - 211 mm, square - 210 mm

#### scripts/pivot_cal_test.py

Same as marker_setup_test.py except it is testing the pivot_cal.py

### pts_stat_marker_frame_skull.pickle

### Realsense.py

Defines Realsense class

Performs some of the initialization needed to stream rgb and depth images

Contains important info such as height/width of image, camera matrix, distortion coefficients, IR camera matrix, IR distortion coefficients, refined RGB + IR camera matrices, extrinsics of TF from IR to RGB camera frames

Detects marker corners and calcualtes transformation from marker to camera frame using Realsense RGB images

Can access Intel-saved intrinsics and extrinsics

### registration/

#### record_points.py

Python script to record points on object touched with marker for fixed set of stationary markers
If markers set stationary around object are ever moved, new points need to be recorded

TODO: It is easy to record where the points on the object are using the marker tool. However the issue is how do we record the location of these points relative to the stationary markers? Currently, the markers on the panels are simply single markers on each side. This will most likely be prone to pose ambiguity, and in situations where we can only detect one marker, relying on that  marker could lead to very inaccurate results. A way to minimize this would be to incorporate multiple markers. However as the setup is right now, each marker panel is set arbitrarily in the environment, and each one will be prone to pose ambiguity. There is nooinformation we know as ground truth between any two markers. All research on improving pose estimation using multiple markers that you know assume that you know the relative transformations between each marker.

One possible solution is to put two markers within the 50x50 mm on each side of the marker panel. Because the two markers will be posted on the same marker panel and printed on the same paper, we will know the ground truth relative position of all the marker corners. We can then use solvepnp on 8 points as opposed to the minimum of 4. However, even with 8 points, the system is still prone to pose ambiguity but most likely much less often. One thing to further analyze is how often a bad pose is produced using solvepnp on 8 points. If a bad pose is produced not very often, a sliding window may be sufficient to minimize the effect of the bad pose. In addition, if multiple of these marker panels are detected, we can average the calculated point on the object to minimize the error.

Another possible solution could be to use the marker panels with single markers on them and ignore all markers whose orientation  in all 3 axes is less than some threshold angle (or prefer markers whose orientation differ heavily from the camera frame -- weighted approach). If all three axes have very little rotation, then that means the camera is looking straight on at a marker, and this marker is heavily prone to pose ambiguity.

Another enhancement / solution is through using IPPE (Infinitesimal Plane-based Pose Estimation). This python package will output the two ambiguous pose estimations for the marker. And with the transformation matrix for each pose, we can compare the 3D position error of the four corners with the depth map position of the corners. Whichever one has smaller corner position error contains the correct transformation matrix.

4/1 Brainstorm: print smaller version of 6-sided marker tool to mount on patient skull. Even when patient head is moving, we wil have fixed reference frame between marker and patient skull. Another idea is to print workstation (eg poster) with multiple markers on it. It is good in that you will know relative transformation between markers but it is bad because you will have to re-register everytime the object is moved.

#### reg_test_Open3D.py

Test file for Open3D package rigid registration method

#### reg_test_pycpd.py

Test file for Coherent Point Drift algorithm written in Python

### RGB_camera_calibration.py

Same thing as IR_calibration.py except for RGB camera.

Uses same checkerboard and same number of images for OpenCV calibration (numbers can be changed in code)

### rot_mat_euler_angles_conversion.py

contains methods for converting rotation matrix to ZYX euler angles and vice versa

### stationary_marker*.pickle
