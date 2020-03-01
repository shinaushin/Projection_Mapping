# reg_test_Open3D.py
# author: Austin Shin

# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/Python/Tutorial/Basic/icp_registration.py

import copy
import numpy as np

from open3d import *
import pyrealsense2 as rs

def draw_registration_result(source, target, transformation):
    """
    Visualize registration result

    Args:
        source: measured data
        target: ground truth data
        transformation: registration transformation

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
    Preprocesses point cloud data

    Args:
        pcd: point cloud data
        voxel_size: downsampling kernel size

    Returns:
        downsampled point cloud, feature histogram descriptors
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
    Transforms source point cloud and preprocesses data

    Args:
        voxel_size: downsampling kernel size
        source: measured point cloud
        target: ground truth point cloud

    Returns:
        source point cloud
        target point cloud
        downsampled source
        downsampled target
        feature histogram descriptors for source
        feature histogram descriptors for target
    """
    print(":: Load two point clouds and disturb initial pose.")
    # disturbance (aka transformation) determined by user
    trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0],
                            [1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0]])
    source.transform(trans_init)
    draw_registration_result(source, target, np.identity(4))

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh

def execute_global_registration(
        source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    """
    Conducts registration using Open3D

    Args:
        source_down: downsampled source point cloud
        target_down: downsampled target point cloud
        source_fpfh: feature histogram descriptors for source
        target_fpfh: feature histogram descriptors for target
        voxel_size: downsampling kernel size

    Returns:
        registration result
    """
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh,
            distance_threshold,
            TransformationEstimationPointToPoint(False), 4,
            [CorrespondenceCheckerBasedOnEdgeLength(0.9),
            CorrespondenceCheckerBasedOnDistance(distance_threshold)],
            RANSACConvergenceCriteria(4000000, 500))
    return result

def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size):
    """
    Performs registration refinement based on ICP

    Args:
        source: measured point cloud
        target: ground truth data
        source_fpfh: feature histogram descriptors for source
        target_fpfh: feature histogram descriptors for target
        voxel_size: downsampling kernel size

    Returns:
        registration result
    """
    distance_threshold = voxel_size * 0.4
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    result = registration_icp(source, target, distance_threshold,
            result_ransac.transformation,
            TransformationEstimationPointToPoint())
    return result

# Open3D registration does not do deformable / non-rigid registration
# test case - same heart scan data, one rotated by 180 degrees
if __name__ == "__main__":
    source = read_point_cloud("../data/heart_scan_processed.txt", format="xyz")
    source.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
    draw_geometries([source])

    test = read_point_cloud("../data/heart_scan_processed.txt", format="xyz")
    draw_geometries([test])

    voxel_size = 0.01
    source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(voxel_size, source, test)

    result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
    print(result_ransac)
    draw_registration_result(source_down, target_down, result_ransac.transformation)

    result_icp = refine_registration(source, target, source_fpfh, target_fpfh, voxel_size)
    print(result_icp)
    draw_registration_result(source, target, result_icp.transformation)

