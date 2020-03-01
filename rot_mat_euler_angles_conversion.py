# rot_mat_euler_angles_conversion.py
# author: Austin Shin

import math
import numpy as np

def rotToEuler(rot):
    """
    Converts rotation matrix to euler angles.
    Convention - rotating about x, y, then z of original set of axes

    Args:
        rot: rotation matrix

    Returns:
        array of xyz euler angles
    """
    sy = math.sqrt(rot[0,0] * rot[0,0] + rot[1,0] * rot[1,0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(rot[2,1], rot[2,2])
        y = math.atan2(-rot[2,0], sy)
        z = math.atan2(rot[1,0], rot[0,0])
    else:
        x = math.atan2(-rot[1,2], rot[1,1])
        y = math.atan2(-rot[2,0], sy)
        z = 0

    return np.array([x,y,z])

def eulerToRot(z,y,x):
    """
        Convert euler angles to rotation matrix
        Convention - rotating about x, y, then z of original set of axes

    Args:
        z,y,x: euler angles

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
    return np.dot(R_z, np.dot(R_y, R_x))
