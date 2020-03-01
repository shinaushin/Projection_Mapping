# reg_test_pycpd.py
# author: Austin Shin

from functools import partial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import time

from pycpd import rigid_registration

def visualize(iteration, error, X, Y, ax):
    """


    Args:


    Returns:

    """
    plt.cla()
    ax.scatter(X[:,0],  X[:,1], X[:,2], color='red', label='Target')
    ax.scatter(Y[:,0],  Y[:,1], Y[:,2], color='blue', label='Source')
    ax.text2D(0.87, 0.92, 'Iteration: {:d}\nError: {:06.4f}'.format(iteration, error), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize='x-large')
    ax.legend(loc='upper left', fontsize='x-large')
    plt.draw()
    plt.pause(0.001)

# great visualization, works well with small batches of data
# need to find appropriate way of downsampling data
# can define threshold error in pycpd source files
def main():
    """


    Args:


    Returns:
    
    """
    target = np.loadtxt('../data/heart_scan_processed.txt')
    source = np.loadtxt('../data/heart_scan_processed.txt')    
    
    X = source[780:800,:]
    pts = X.transpose()
    theta = np.radians(30)
    c,s = np.cos(theta), np.sin(theta)
    R = np.array( ( (c, -s, 0), (s, c, 0), (0, 0, 1) ) )
    pts = np.matmul(R, pts)
    source = pts.transpose()    

    target = target[700:900,:]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    callback = partial(visualize, ax=ax)

    reg = rigid_registration(**{ 'X': source, 'Y': target })
    TY, (s_reg, R_reg, t_reg) = reg.register(callback)
    print(s_reg, R_reg, t_reg)
    plt.show()

if __name__ == '__main__':
    main()
