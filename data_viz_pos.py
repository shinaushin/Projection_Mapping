import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys

deg_inc = 10

# returns all data needed for plots
def consolidate_data(pos_truth, pos_calc, axis):
    deg = 30
    err = []
    mean_pos_err = []
    std_dev_pos_err = []
    while (deg <= 150):
        diff = np.subtract(pos_truth[deg], pos_calc[deg]) # error in position vectors
        if axis == 0: # norm of error vector
            mag_err = np.linalg.norm(diff, axis=1)
        else:
            mag_err = diff[:,axis-1] # error in x,y,or z component

        err.append(mag_err)
        mean_pos_err.append(np.mean(mag_err))    
        std_dev_pos_err.append(np.std(mag_err))
        deg = deg + deg_inc

    return err, mean_pos_err, std_dev_pos_err
    
# returns appropriate title for plot
def getTitle(axis):
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
    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
       ['30', '40', '50' ,'60', '70', '80', '90', '100', '110', '120', '130', '140', '150'])
  
    filename = foldername + '/Position' + str(axis) + '_boxplot.jpg' 
    plt.savefig(filename)

# plots mean and standard deviation
def mean_stddev(pos_truth, pos_calc, foldername, axis=0):
    _, mean_err, std_dev_err = consolidate_data(pos_truth, pos_calc, axis)
    
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
    ordered = {}
    for key in data:
        dataArr = []
        for item in data[key]:
            dataArr.append(item[0])
        ordered[key] = dataArr
    return ordered


def main():
    picklename = sys.argv[1]
    foldername = sys.argv[2]

    with open(picklename + '.pickle', 'rb') as f:
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

