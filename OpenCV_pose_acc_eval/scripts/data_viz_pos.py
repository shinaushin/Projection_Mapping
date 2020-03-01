# data_viz_pos.py
# author: Austin Shin

import sys
sys.path.append('../../')

import matplotlib.pyplot as plt
import numpy as np
import pickle

deg_inc = 10 # set by user

def consolidate_data(pos_truth, pos_calc, axis):
    """
    Calculates error statistics based on position data

    Args:
        pos_truth: ground truth
        pos_calc: measured position data
        axis: 

    Returns:
        xyz error or mag error
        avg error
        standard deviation of error
    """
    deg = 30
    err = []
    mean_pos_err = []
    std_dev_pos_err = []
    while (deg <= 150):
        diff = np.subtract(pos_truth[deg], pos_calc[deg]) # error in pos vectors
        if axis == 0: # norm of error vector
            mag_err = np.linalg.norm(diff, axis=1)
        else:
            mag_err = diff[:,axis-1] # error in x,y,or z component

        err.append(mag_err)
        mean_pos_err.append(np.mean(mag_err))    
        std_dev_pos_err.append(np.std(mag_err))
        deg = deg + deg_inc

    return err, mean_pos_err, std_dev_pos_err
    
def getTitle(axis):
    """
    Generates appropriate title for plot

    Args:
        axis: indicates, x/y/z/total mag error

    Returns:
        plot title
    """
    title = ''
    if axis == 0:
        title = 'Accuracy of ArUco Marker 3D Position Estimates: Total Magnitude'
    elif axis == 1:
        title = 'Accuracy of ArUco Marker 3D Position Estimates: X axis'
    elif axis == 2:
        title = 'Accuracy of ArUco Marker 3D Position Estimates: Y axis'
    else:
        title = 'Accuracy of ArUco Marker 3D Position Estimates: Z axis'

    return title

# creates box plot
def boxplot_err(pos_truth, pos_calc, foldername, axis=0):
    """
    Creates box plot of error

    Args:
        pos_truth: ground truth
        pos_calc: measured position data
        foldername: where to save box plot
        axis: indicates x/y/z/total mag

    Returns:
        None
    """
    err, _, _ = consolidate_data(pos_truth, pos_calc, axis)
    
    bar_width = 0.35
    n_groups = 13
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    ax.set_xlabel('Degrees')
    ax.set_ylabel('Mean Error (m)')
    title = getTitle(axis)
    ax.set_title(title)
    ax.set_xticks(index + bar_width/2)
    ax.legend()
    fig.tight_layout()
    plt.boxplot(err)
    # relies on specific experiment setup - 30-150 deg in increments of 10
    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
       ['30', '40', '50' ,'60', '70', '80', '90', '100', '110', '120', '130',
       '140', '150'])
  
    filename = foldername + '/Position' + str(axis) + '_boxplot.jpg' 
    plt.savefig(filename)

def mean_stddev(pos_truth, pos_calc, foldername, axis=0):
    """
    Plots mean and standard deviation

    Args:
        pos_truth: ground truth
        pos_calc: measured position data
        foldername: where to save box plot
        axis: indicates x/y/z/total mag

    Returns:
        None
    """
    _, mean_err, std_dev_err = consolidate_data(pos_truth, pos_calc, axis)
    
    # relies on specific experiment setup - 30-150 deg in increments of 10
    x = np.array([30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150])
    fig, ax = plt.subplots()
    ax.set_xlim(20, 160)
    ax.set_xlabel('Degrees')
    ax.set_ylabel('Mean Error (m)')

    title = getTitle(axis)
    ax.set_title(title)
    plt.errorbar(x, mean_err, std_dev_err, linestyle='None', marker='o')
    
    filename = foldername + '/Position' + str(axis) + '_mean_stddev.jpg'
    plt.savefig(filename)

def reorganize(data):
    """
    Reformats data

    Args:
        data: position data

    Returns:
        reordered data
    """
    ordered = {}
    for key in data:
        dataArr = []
        for item in data[key]:
            dataArr.append(item[0])
        ordered[key] = dataArr
    return ordered

def main():
    """
    Executes error visualization of position data
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

    with open('../pickles/' + picklename + '.pickle', 'rb') as f:
        pos_truth, pos_calc, rot_truth, rot_calc = pickle.load(f)

    # print(pos_truth)
    # print(pos_calc)
    pos_truth = reorganize(pos_truth)
    pos_calc = reorganize(pos_calc)

    boxplot_err(pos_truth, pos_calc, foldername)
    boxplot_err(pos_truth, pos_calc, foldername, 1)
    boxplot_err(pos_truth, pos_calc, foldername, 2)
    boxplot_err(pos_truth, pos_calc, foldername, 3)

    mean_stddev(pos_truth, pos_calc, foldername)
    mean_stddev(pos_truth, pos_calc, foldername, 1)
    mean_stddev(pos_truth, pos_calc, foldername, 2)
    mean_stddev(pos_truth, pos_calc, foldername, 3)

    # plt.show()

if __name__ == "__main__":
    main()

