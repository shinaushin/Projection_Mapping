# Projection_Mapping

Development of a projection mapping prototype that will project important patient data such as CT scans or MRI scans onto patient bodies in realtime. 

## Documentation

Unless otherwise stated, all use cases for python files are simply: python <file_name.py> 

### acc_eval*.pickle / multi_marker.pickle

pickle is a data format used by Python to easily serialize and deserialize information and is the main way I use to save data that I want to use for later at any point in time
These pickle files contain data from the marker pose estimation accuracy evaluations conducted to visualize the position and orientation error

### Archive/

Not significant
Old files that are not used any more

### camera_calibration.py

Python script used to calibrate camera and will print intrinsics of camera (camera matrix and distortion coefficients)
Although there were default parameters that were stored in the Realsense camera, they were not at all accurate for unknown reasons. This script will work for any 4x3 checkerboard and will collect 20 different images before conducted the calibration using OpenCV. This script does not save the images.

### data/

#### heart-scan3d.txt
OBJ file converted to txt file

#### heart_scan_processed.txt
txt file parsed from heart-scan3d.txt so that we only keep vertices

#### heart_scan_scaled.txt
original heart scaled by 2. For the purpose of testing different registration packages.

#### scale_data.py
reads in txt file with vertex data and multiplies everything by a scalar number

### data_viz*.py
Use case: python data_viz*.py <name of pickle to deserialize> <name of folder to put plots in>
  
Deserializes pickle containing all data collected from pose_est_accuracy_eval.py
Plots box plots and mean/stddev plots for x,y,z,mag error in position and quaternion,theta error in orientation

### IR_calibration.py

Same thing as camera_calibration.py except for IR camera.
Uses same checkerboard and same number of images for OpenCV calibration

### marker_detection_realsense.py

Detects markers in images from Realsense camera
Draws marker coordinate frame on it
Writes marker 3D coordinate in camera coordinate frame on image
Draws checkerboard corners detected on image (4x3 checkerboard)

### marker_det_plots_bad_rot_metric/

plots of results of marker pose estimation using faulty metric for orientation error

### marker_det_plots_one_marker/

plots of results of marker pose estimation using only one marker with corrected error metrics

### marker_det_plots_two_markers/

plots of results of marker pose estimation using two markers with corrected error metrics

### marker_setup.py

python script to calculate transformation from each marker on the side of the marker tool to the top marker.
Saves all transformations into a pickle file.
Verifies that transformations are reasonable by making the angle of rotation between any two consecutive markers is within a certain threshold that can be set by the user.
For example, suppose the marker tool used had 7 markers. This means there is 1 marker on top and 6 markers on the side. The tool is therefore in a hexagonal shape. Therefore the angle of rotation for any two consecutive markers should be approximately 60 degrees.

### marker_setup_test.py

python script to evaluation accuracy of marker_setup.py
Creates box plot and mean/stddev plot of error in saved transformations
User can define how many marker setup procedures to conduct before doing the error analysis
Saves plot data in a pickle file

### markertool*.pickle

For each marker tool, we specify the frame of the top marker as the overall marker tool coordinate frame.
This pickle file contains the transformation of every other marker to that top marker

### pivot_cal.py

Performs pivot calibration of user-specified marker tool
Outputs position of tip of marker tool relative to origin of marker tool in marker tool coordinate frame
Saves that position in pickle file

### pivot_cal_test.py

Same as marker_setup_test.py except it is testing the pivot_cal.py

### pose_est_accuracy_eval.py



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

### rot_mat_euler_angles_conversion.py

contains methods for converting rotation matrix to ZYX euler angles and vice versa

### test/

Not significant.
Folder that contains plots created when testing data viz files.
