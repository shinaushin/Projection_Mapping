# reg_test_OpenCV.py
# @author Austin Shin

# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/Python/Tutorial/Basic/icp_registration.py

from open3d import *
import copy
import pyrealsense2 as rs
import numpy as np
import cv2

def draw_registration_result(source, target, transformation):
    """
    Draws transformed source point cloud and target point cloud

    Args:
        source: measured point cloud
        target: desired point cloud
        transformation: best fit transformation for source to match target
    
    Returns:
        None
    """
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    draw_geometries([source_temp, target_temp])

def preprocess_point_cloud(pcd, voxel_size):
    """
    Downsampling process adapted from open3d

    Args:
        pcd: point cloud data
        voxel_size: downsampling kernel size
    
    Returns:
        downsampled point cloud, FPFH point cloud
    """
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = voxel_down_sample(pcd, voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    estimate_normals(pcd_down, KDTreeSearchParamHybrid(
            radius = radius_normal, max_nn = 30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = compute_fpfh_feature(pcd_down,
            KDTreeSearchParamHybrid(radius = radius_feature, max_nn = 100))
    return pcd_down, pcd_fpfh

def prepare_dataset(voxel_size, source, target):
    """
    Executes preprocessing of dataset

    Args:
        voxel_size: downsampling kernel size
        source: measured point cloud
        target: desired point cloud
    
    Returns:
        original source, target point clouds
        downsampled point cloud for source and target
        FPFH point cloud for source and target
    """
    print(":: Draw initial pose.")
    draw_registration_result(source, target, np.identity(4))

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh


# Test case - two txt with heart scan data, one is scaled to be twice as big and
# rotated 90 degrees
# Need to find more appropriate way to downsample point cloud data / Open3D way
# doesn't work
# Works with translation too
if __name__ == "__main__":
    source = read_point_cloud("../data/heart_scan_processed.txt", format="xyz")
    source.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])

    test = read_point_cloud("../data/heart_scan_scaled.txt", format="xyz")
    test.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
    test.transform([[0,-1,0,100],[1,0,0,100],[0,0,1,100],[0,0,0,1]])

#    voxel_size = 0.005
#    source, target, source_down, target_down, source_fpfh, target_fpfh =
#       prepare_dataset(voxel_size, source, test)
    
    draw_registration_result(source, target, np.identity(4))

    draw_geometries([source])
    draw_geometries([target])

    source_arr = np.asmatrix(source.points)
    target_arr = np.asmatrix(target.points)
    if (len(target_arr) > len(source_arr)):
        target_arr = target_arr[:len(source_arr),:] 
    
    retval, out, inliers = cv2.estimateAffine3D(source_arr, target_arr,
        confidence=0.96)
    out = np.vstack([out, [0.0, 0.0, 0.0, 1.0]])
    print(out)
    draw_registration_result(source, target, out)
