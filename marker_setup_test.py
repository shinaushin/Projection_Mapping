import numpy as np
import cv2
import cv2.aruco as aruco
import pyrealsense2 as rs
from Realsense import RealSense
import pickle
import math
from rot_mat_euler_angles_conversion import rotToEuler
from marker_setup import setup
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
comm_marker_id = marker_IDs[-1]

if num_markers == 7:
    ground_truth = 60
else:
    ground_truth = 90

data_pts = 30
angle_list = []
for i in range(data_pts):
    _, angles = setup(marker_IDs, num_markers, comm_marker_id)
    angle_list[i,:] = [angle - ground_truth for angle in angles]

with open('marker_setup_test_markertool'+str(num_markers)+'.pickle', 'wb') as f:
    pickle.dump(angle_list, f)

# box plot
bar_width = 0.35
n_groups = 13
fig, ax = plt.subplots()
index = np.arange(n_groups)
ax.set_ylabel('Error (deg)')
title = "Error in Orientation Between Consecutive Markers"
ax.set_title(title)
ax.set_xticks(index + bar_width/2)
ax.legend()
fig.tight_layout()
plt.boxplot(angle_list)

if num_markers == 7:
    plt.xticks([1, 2, 3, 4, 5, 6], [r'$\alpha_{12}', r'$\alpha_{23}', r'$\alpha_{34}', r'$\alpha_{45}', r'$\alpha_{56}', r'$\alpha_{61}'])
else:
    plt.xticks([1, 2, 3, 4], [r'$\alpha_{12}', r'$\alpha_{23}', r'$\alpha_{34}', r'$\alpha_{41}'])

# mean and stddev plot
mean_err = np.mean(angle_list, axis=0)
std_dev_err = np.std(angle_list, axis=0)
if num_markers == 7:
    x = np.array([r'$\alpha_{12}', r'$\alpha_{23}', r'$\alpha_{34}', r'$\alpha_{45}', r'$\alpha_{56}', r'$\alpha_{61}'])
else:
    x = np.array([r'$\alpha_{12}', r'$\alpha_{23}', r'$\alpha_{34}', r'$\alpha_{41}'])

fig, ax = plt.subplots()
ax.set_ylabel('Error (deg)')

title = "Error in Orientation Between Consecutive Markers"
ax.set_title(title)
plt.errorbar(x, mean_err, std_dev_err, linestyle='None', marker='o')

