import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math

deg_inc = 10

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
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R

def consolidate_data(rot_truth, rot_calc):
    vec_dist_err = {}
    mag_err = {}
    for key in rot_truth:
        item1 = rot_truth[key]
        item2 = rot_calc[key]
        vec_dist_item = []
        theta_item = []
        for i in range(len(item1)):
            rot_mat1 = eulerToRot(item1[i][0], item1[i][1], item1[i][2])
            rot_mat2 = eulerToRot(item2[i][0], item2[i][1], item2[i][2])
            rot_mat1_T = np.transpose(rot_mat1)
            rot = np.matmul(rot_mat1_T, rot_mat2)
            theta = math.acos( (np.trace(rot) - 1.0) / 2.0)
            rot_vec1,_ = cv2.Rodrigues(rot_mat1)
            rot_vec2,_ = cv2.Rodrigues(rot_mat2)
            if np.linalg.norm(np.subtract(rot_vec1, rot_vec2)) > np.linalg.norm(np.subtract(rot_vec1, -rot_vec2)):
                vec_dist_item.append(np.reshape(np.subtract(rot_vec1, -rot_vec2), 3))
            else:
                vec_dist_item.append(np.reshape(np.subtract(rot_vec1, rot_vec2), 3))
            theta_item.append(theta*180/3.1415)
        vec_dist_err[key] = vec_dist_item
        mag_err[key] = theta_item

    return vec_dist_err, mag_err

def preprocess_for_plot(err):
    all_err = []
    mean_err = []
    stddev_err = []
    keys = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160]
#     keys = [60, 70, 80, 90, 100, 110, 120]
    for key in keys:
        all_err.append(err[key])
        mean_err.append(np.mean(err[key]))
        stddev_err.append(np.std(err[key]))        

    return all_err, mean_err, stddev_err

def getTitle(mode):
    title = ''
    if mode == 0:
        title = 'Accuracy of ArUco Marker 3D Orientation Estimates: Axis of Rotation'
    elif mode == 1:
        title = 'Accuracy of ArUco Marker 3D Orientation Estimates: Magnitude of Rotation'

    return title

def boxplot_err(err, mode, axis=0):
    bar_width = 0.35
    n_groups = 7 # 15
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
    # ax.set_xticklabels(('60', '70', '80', '90', '100', '110', '120'))
    ax.set_xticklabels(('10', '20', '30', '40', '50', '60', '70', '80', '90',
                        '100', '110', '120', '130', '140', '150', '160'))
    ax.legend()
    fig.tight_layout()
    plt.boxplot(err)

    if mode == 0:
        if axis == 0:
            filename = 'marker_acc_plots4/quat_dist_boxplot.jpg'
        elif axis == 1:
            filename = 'marker_acc_plots4/quat_x_boxplot.jpg'
        elif axis == 2:
            filename = 'marker_acc_plots4/quat_y_boxplot.jpg'
        else:
            filename = 'marker_acc_plots4/quat_z_boxplot.jpg'
    else:
        filename = 'marker_acc_plots4/theta_boxplot.jpg'
    plt.savefig(filename)

def mean_stddev(mean_err, std_dev_err, mode, axis=0):
    # x = np.array([60, 70, 80, 90, 100, 110, 120])
    x = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160])
    fig, ax = plt.subplots()
    ax.set_xlim(0, 170)
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
            filename = 'marker_acc_plots4/quat_dist_mean_stddev.jpg'
        elif axis == 1:
            filename = 'marker_acc_plots4/quat_x_mean_stddev.jpg'
        elif axis == 2:
            filename = 'marker_acc_plots4/quat_y_mean_stddev.jpg'
        else:
            filename = 'marker_acc_plots4/quat_z_mean_stddev.jpg'
    else:
        filename = 'marker_acc_plots4/theta_mean_stddev.jpg'
    plt.savefig(filename)


with open('acc_eval2.pickle', 'rb') as f:
    _, _, vec_dist_err, mag_err = pickle.load(f)

vec_err = {}
for key in vec_dist_err:
    vec_dist_err[key] = np.reshape(vec_dist_err[key], (30,3))
    vec_err[key] = np.linalg.norm(vec_dist_err[key], axis=1)
vd_all, vd_mean, vd_stddev = preprocess_for_plot(vec_err)
boxplot_err(vd_all, 0)
mean_stddev(vd_mean, vd_stddev, 0)

for key in vec_dist_err:
    item = vec_dist_err[key]
    # print(item)
    vec_err[key] = item[:,0]
vd_all, vd_mean, vd_stddev = preprocess_for_plot(vec_err)
boxplot_err(vd_all, 0, 1)
mean_stddev(vd_mean, vd_stddev, 0, 1)

for key in vec_dist_err:
    vec_err[key] = vec_dist_err[key][:,1]
vd_all, vd_mean, vd_stddev = preprocess_for_plot(vec_err)
boxplot_err(vd_all, 0, 2)
mean_stddev(vd_mean, vd_stddev, 0, 2)

for key in vec_dist_err:
    vec_err[key] = vec_dist_err[key][:,2]
vd_all, vd_mean, vd_stddev = preprocess_for_plot(vec_err)
boxplot_err(vd_all, 0, 3)
mean_stddev(vd_mean, vd_stddev, 0, 3)

theta_all, theta_mean, theta_stddev = preprocess_for_plot(mag_err)
boxplot_err(theta_all, 1)
mean_stddev(theta_mean, theta_stddev, 1)
