import pickle
import numpy as np
import cv2
import pyrealsense2 as rs
from Realsense import RealSense
import math
import rot_mat_euler_angles_conversion import rotToEuler
import cv2.aruco as aruco

num_markers = raw_input("How many markers are there on the tool you are using? ")
with open('markertool'+str(num_markers)+'.pickle', 'rb') as f:
    x = pickle.load(f)

cam = Realsense()

profile = cam.pipeline.start(cam.config)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
align_to = rs.stream.color
align = rs.align(align_to)

num_pts = raw_input("How many points do you want to record? ")

stationary_markers = raw_input("What are the IDs of the markers stationed around the object? ")
stationary_markers = [int(x) for x in stationary_markers.split()]

recorded = 0
try:
    while (recorded < num_pts):
        

