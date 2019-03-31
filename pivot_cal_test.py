from pivot_cal import calibrate
import math
import numpy as np
import cv2
import cv2.aruco as aruco
import pyrealsense2 as rs
import pickle
from Realsense import RealSense
from rot_mat_euler_angles_conversion import rotToEuler
import matplotlib.pyplot as plt
import sys


cam = RealSense()
profile = cam.pipeline.start(cam.config)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
align_to = rs.stream.color
align = rs.align(align_to)

marker_IDs = raw_input("What are the IDs of the markers of the tool, starting in order with the ones on the side and finally the one on the top all separated by a single space? ")
marker_IDs = [int(x) for x in marker_IDs.split()]
num_markers = len(marker_IDs)
if num_markers == 7:
    ground_truth = .211
else:
    ground_truth = .210
comm_marker_id = marker_IDs[-1]

with open('markertool'+str(num_markers)+'.pickle', 'rb') as f:
    tf_dict = pickle.load(f)

# User input of how long to do calibration for
num_pts = raw_input("How many pivot calibration points do you want to use? ")
num_pts = int(num_pts)

data_pts = 30
for i in range(data_pts):
    x = calibrate(marker_IDs, num_markers, comm_marker_id, tf_dict, num_pts)
    x[2] = x[2] + ground_truth
    pivot_val[i,:] = x

with open('pivot_cal_test_markertool'+str(num_markers)+'.pickle', 'wb') as f:
    pickle.dump(pivot_val, f)

# box plot
bar_width = 0.35
n_groups = 13
fig, ax = plt.subplots()
index = np.arange(n_groups)
ax.set_ylabel('Error (m)')
title = "Error in Tool Tip Position Relative to Base"
ax.set_title(title)
ax.set_xticks(index + bar_width/2)
ax.legend()
fig.tight_layout()
plt.boxplot(pivot_val)
plt.xticks([1, 2, 3], ['X', 'Y', 'Z'])

# mean and stddev plot
mean_err = np.mean(pivot_val, axis=0)
std_dev_err = np.std(pivot_val, axis=0)
x = np.array(['X', 'Y', 'Z'])
fig, ax = plt.subplots()
ax.set_ylabel('Error (m)')

title = "Error in Tool Tip Position Relative to Base"
ax.set_title(title)
plt.errorbar(x, mean_err, std_dev_err, linestyle='None', marker='o')

