# data_viz_rot.py
# author: Austin Shin

import sys
sys.path.append('../../')

import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
import pickle

deg_inc = 10 # set by user

# convert euler angles to rotation matrix
def eulerToRot(x,y,z):
    """
    Convert euler angles to rotation matrix

    Args:
        x: rotation about x
        y: rotation about y
        z: rotation about z

    Returns:
        rotation matrix
    """
    R_x = np.array( [ [1, 0, 0],
                      [0, math.cos(x), -math.sin(x)],
                      [0, math.sin(x), math.cos(x)] ])
    R_y = np.array( [ [math.cos(y), 0, math.sin(y) ],
                      [0, 1, 0],
                      [-math.sin(y), 0, math.cos(y) ] ])
    R_z = np.array( [ [math.cos(z), -math.sin(z), 0],
                      [math.sin(z), math.cos(z), 0],
                      [0, 0, 1] ] )
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R

# mainly converts error stored in dictionaries into ordered lists
def preprocess_for_plot(err):
    """
    Converts error stored in dictionaries into lists based on order of angles

    Args:
        err: dictionary containing error values

    Returns:
        ordered list of error values
    """
    all_err = []
    mean_err = []
    stddev_err = []
    # angles used for keys based on specific experimental setup
    keys = [30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
    for key in keys:
        all_err.append(err[key])
        mean_err.append(np.mean(err[key]))
        stddev_err.append(np.std(err[key]))        

    return all_err, mean_err, stddev_err

def getTitle(mode):
    """
    Generates appropriate title for plot

    Args:
        mode: indicates total magnitude or individual axis

    Returns:
        appropriate plot title
    """
    title = ''
    if mode == 0:
        title = 'Accuracy of ArUco Marker 3D Orientation Estimates: Axis of Rotation'
    elif mode == 1:
        title = 'Accuracy of ArUco Marker 3D Orientation Estimates: Magnitude of Rotation'

    return title

def boxplot_err(err, foldername, mode, axis=0):
    """
    Creates box plot for error

    Args:
        err: error data
        foldername: folder where to store plots
        mode: quaternion or theta (rotation magnitude)
        axis: x/y/z component

    Returns:
        None
    """
    bar_width = 0.35
    n_groups = 13
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    ax.set_xlabel('Degrees')
    if mode == 0:
        if axis == 0:
            ax.set_ylabel('Quaternion Distance')
        elif axis == 1:
            ax.set_ylabel('Quaternion X-Component Difference')
        elif axis == 2:
            ax.set_ylabel('Quaternion Y-Component Difference')
        else:
            ax.set_ylabel('Quaternion Z-Component Difference')
    else:
        ax.set_ylabel('Theta (deg)')
    title = getTitle(mode)
    ax.set_title(title)
    ax.set_xticks(index + bar_width/2)
    ax.legend()
    fig.tight_layout()
    plt.boxplot(err)
    # depends on specific experimental setup
    ax.set_xticklabels(('30', '40', '50', '60', '70', '80', '90',
                        '100', '110', '120', '130', '140', '150'))

    if mode == 0:
        if axis == 0:
            filename = foldername + '/quat_dist_boxplot.jpg'
        elif axis == 1:
            filename = foldername + '/quat_x_boxplot.jpg'
        elif axis == 2:
            filename = foldername + '/quat_y_boxplot.jpg'
        else:
            filename = foldername + '/quat_z_boxplot.jpg'
    else:
        filename = foldername + '/theta_boxplot.jpg'
    plt.savefig(filename)

def mean_stddev(mean_err, std_dev_err, foldername,  mode, axis=0):
    """
    Plots mean and standard deviation of error

    Args:
        mean_err: average error
        std_dev_err: standard deviation error
        foldername: folder where to store plots
        mode: quaternion or theta (angle magnitude)
        axis: x/y/z component

    Returns:
        None
    """
    # x = np.array([60, 70, 80, 90, 100, 110, 120])
    x = np.array([30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150])
    fig, ax = plt.subplots()
    ax.set_xlim(20, 160)
    ax.set_xlabel('Degrees')
    if mode == 0:
        if axis == 0:
            ax.set_ylabel('Quaternion Distance')
        elif axis == 1:
            ax.set_ylabel('Quaternion X-Component Difference')
        elif axis == 2:
            ax.set_ylabel('Quaternion Y-Component Difference')
        else:
            ax.set_ylabel('Quaternion Z-Component Difference')
    else:
        ax.set_ylabel('Theta (deg)')

    title = getTitle(mode)
    ax.set_title(title)
    plt.errorbar(x, mean_err, std_dev_err, linestyle='None', marker='o')
    
    if mode == 0:
        if axis == 0:
            filename = foldername + '/quat_dist_mean_stddev.jpg'
        elif axis == 1:
            filename = foldername + '/quat_x_mean_stddev.jpg'
        elif axis == 2:
            filename = foldername + '/quat_y_mean_stddev.jpg'
        else:
            filename = foldername + '/quat_z_mean_stddev.jpg'
    else:
        filename = foldername + '/theta_mean_stddev.jpg'
    plt.savefig(filename)

# picklename: name of pickle (without extension) that you are reading data from
# foldername: path of folder you want to save your plots in
#	      can be absolute path or relative path from where you execute file
def main():
    """
    Executes error visualization of rotation data
    picklename: name of pickle file (without extension) you are reading data
                from
    foldername: path of folder you want to save your plots in
	            can be absolute path or relative path to where you execute file

    Args:
        None

    Returns:
        None
    """
    picklename = sys.argv[1]
    foldername = sys.argv[2]

    with open(picklename + '.pickle', 'rb') as f:
        _, _, vec_dist_err, mag_err = pickle.load(f)

    vec_err = {}

    # creates plots for RMS distance in err quaternion
    for key in vec_dist_err:
        vec_dist_err[key] = np.reshape(vec_dist_err[key], (30,3))
        vec_err[key] = np.linalg.norm(vec_dist_err[key], axis=1)
    vd_all, vd_mean, vd_stddev = preprocess_for_plot(vec_err)
    boxplot_err(vd_all, foldername, 0)
    mean_stddev(vd_mean, vd_stddev, foldername, 0)

    # creates plots for x component of err quaternion
    for key in vec_dist_err:
        item = vec_dist_err[key]
        vec_err[key] = item[:,0]
    vd_all, vd_mean, vd_stddev = preprocess_for_plot(vec_err)
    boxplot_err(vd_all, foldername, 0, 1)
    mean_stddev(vd_mean, vd_stddev, foldername, 0, 1)

    # creates plots for y component of err quaternion
    for key in vec_dist_err:
        vec_err[key] = vec_dist_err[key][:,1]
    vd_all, vd_mean, vd_stddev = preprocess_for_plot(vec_err)
    boxplot_err(vd_all, foldername, 0, 2)
    mean_stddev(vd_mean, vd_stddev, foldername, 0, 2)

    # creates plots for z component of err quaternion
    for key in vec_dist_err:
        vec_err[key] = vec_dist_err[key][:,2]
    vd_all, vd_mean, vd_stddev = preprocess_for_plot(vec_err)
    boxplot_err(vd_all, foldername, 0, 3)
    mean_stddev(vd_mean, vd_stddev, foldername, 0, 3)

    # creates plots for theta of err rotation matrix
    theta_all, theta_mean, theta_stddev = preprocess_for_plot(mag_err)
    boxplot_err(theta_all, foldername,  1)
    mean_stddev(theta_mean, theta_stddev, foldername, 1)

if __name__ == "__main__":
    main()
