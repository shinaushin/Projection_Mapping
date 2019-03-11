import pickle
import numpy as np
import matplotlib.pyplot as plt

deg_inc = 10

def consolidate_data(pos_truth, pos_calc, axis):
    deg = 10
    err = []
    mean_pos_err = []
    std_dev_pos_err = []
    while (deg < 170):
        diff = np.subtract(pos_truth[deg], pos_calc[deg])
        if axis == 0:
            mag_err = np.linalg.norm(diff, axis=1)
        else:
            mag_err = diff[:,axis-1]

        err.append(mag_err)
        mean_pos_err.append(np.mean(mag_err))    
        std_dev_pos_err.append(np.std(mag_err))
        deg = deg + deg_inc

    return err, mean_pos_err, std_dev_pos_err
    
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

def boxplot_err(pos_truth, pos_calc, unit, axis=0):
    err, _, _ = consolidate_data(pos_truth, pos_calc, axis)
    
    bar_width = 0.35
    n_groups = 15
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    ax.set_xlabel('Degrees')
    ax.set_ylabel('Mean Error (' + unit + ')')
    title = getTitle(axis)
    ax.set_title(title)
    ax.set_xticks(index + bar_width/2)
    # ax.set_xticklabels(('10', '20', '30', '40', '50', '60', '70', '80', '90',
    #                    '100', '110', '120', '130', '140', '150', '160'))
    ax.legend()
    fig.tight_layout()
    plt.boxplot(err)
    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
       ['10', '20', '30', '40', '50' ,'60', '70', '80', '90', '100', '110', '120', '130', '140', '150', '160'])
  
    filename = 'marker_acc_plots3/Position' + str(axis) + '_boxplot.jpg' 
    plt.savefig(filename)

def mean_stddev(pos_truth, pos_calc, unit, axis=0):
    _, mean_pos_err, std_dev_pos_err = consolidate_data(pos_truth, pos_calc, axis)
    plot(mean_pos_err, std_dev_pos_err, unit, axis)

def plot(mean_err, std_dev_err, unit, axis):
    x = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160])
    fig, ax = plt.subplots()
    ax.set_xlim(0, 170)
    ax.set_xlabel('Degrees')
    ax.set_ylabel('Mean Error (' + unit + ')')

    title = getTitle(axis)
    ax.set_title(title)
    plt.errorbar(x, mean_err, std_dev_err, linestyle='None', marker='o')
    
    filename = 'marker_acc_plots3/Position' + str(axis) + '_mean_stddev.jpg'
    plt.savefig(filename)

def reorganize(data):
    ordered = {}
    for key in data:
        dataArr = []
        for item in data[key]:
            dataArr.append(item[0])
        ordered[key] = dataArr
    return ordered


with open('acc_eval2.pickle', 'rb') as f:
    pos_truth, pos_calc, rot_truth, rot_calc = pickle.load(f)

pos_truth = reorganize(pos_truth)
pos_calc = reorganize(pos_calc)

boxplot_err(pos_truth, pos_calc, 'm')
boxplot_err(pos_truth, pos_calc, 'm', 1)
boxplot_err(pos_truth, pos_calc, 'm', 2)
boxplot_err(pos_truth, pos_calc, 'm', 3)

mean_stddev(pos_truth, pos_calc, 'm')
mean_stddev(pos_truth, pos_calc, 'm', 1)
mean_stddev(pos_truth, pos_calc, 'm', 2)
mean_stddev(pos_truth, pos_calc, 'm', 3)

# plt.show()

