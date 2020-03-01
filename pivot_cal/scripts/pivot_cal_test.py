# pivot_cal_test.py
# author: Austin Shin

import sys
sys.path.append('../../')

import cv2
import cv2.aruco as aruco
import math
import matplotlib.pyplot as plt
import numpy as np
import pickle

import pyrealsense2 as rs

from Realsense import RealSense

from pivot_cal import calibrate
from rot_mat_euler_angles_conversion import rotToEuler

def main():
    """
    Perform pivot calibration

    Args:
        None

    Returns:
        None
    """

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
        ground_truth = .211 # meters
    else:
        ground_truth = .210 # meters
    comm_marker_id = marker_IDs[-1]

    with open('markertool'+str(num_markers)+'.pickle', 'rb') as f:
        tf_dict = pickle.load(f)

    # User input of how long to do calibration for
    num_pts = raw_input("How many pivot calibration points do you want to use? ")
    num_pts = int(num_pts)

    data_pts = 10
    pivot_val = []
    for i in range(data_pts):
        x = calibrate(cam, align, marker_IDs, num_markers, comm_marker_id,
            tf_dict, num_pts)
        x[2] = x[2] + ground_truth
        pivot_val.append(x)

    with open('../pickles/pivot_cal_test_markertool'+str(num_markers)+'.pickle', 'wb') as f:
        pickle.dump(pivot_val, f)


def plot(num_markers):
    """
    Plot accuracy statistics

    Args:
        num_markers: number of markers on digitizer

    Returns:
        None
    """
    with open('../pickles/pivot_cal_test_markertool'+str(num_markers)+'.pickle', 'rb') as f:
        pivot_val = pickle.load(f)

    # box plot
    bar_width = 0.35
    n_groups = 3
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    ax.set_ylabel('Error (m)')
    title = "Error in Tool Tip Position Relative to Base"
    ax.set_title(title)
    ax.set_xticks(index + bar_width/2)
    ax.legend()
    fig.tight_layout()
    plt.boxplot(np.array(pivot_val).T.tolist())
    plt.xticks([1, 2, 3], ['X', 'Y', 'Z'])

    filename = '../plots/tool'+str(num_markers)+'_boxplot.jpg'
    plt.savefig(filename)

    # mean and stddev plot
    mean_err = np.mean(pivot_val, axis=0)
    std_dev_err = np.std(pivot_val, axis=0)

    fig, ax = plt.subplots()
    ax.set_xlim(0,4)
    ax.set_ylabel('Error (m)')

    title = "Error in Tool Tip Position Relative to Base"
    ax.set_title(title)
    plt.errorbar(np.arange(1,4), mean_err, std_dev_err, linestyle='None',
        marker='o')

    plt.xticks([1, 2, 3], ['X', 'Y', 'Z'])

    filename = '../plots/tool'+str(num_markers)+'_mean_stddev.jpg'
    plt.savefig(filename)

    plt.show()

if __name__ == "__main__":
    # main()
    plot(7)
