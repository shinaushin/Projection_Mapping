# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/Python/Tutorial/Basic/icp_registration.py

# Registration is BROKEN. Open3D doesn't do non-rigid registration

from open3d import *
import copy
import pyrealsense2 as rs
import numpy as np

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    draw_geometries([source_temp, target_temp])

def preprocess_point_cloud(pcd, voxel_size):
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
    print(":: Load two point clouds and disturb initial pose.")
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
    distance_threshold = voxel_size * 0.4
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    result = registration_icp(source, target, distance_threshold,
            result_ransac.transformation,
            TransformationEstimationPointToPoint())
    return result


try:
    # dev = find_device_that_supports_advanced_mode()
    ctx = rs.context()
    dev = ctx.query_devices()[0]
    advnc_mode = rs.rs400_advanced_mode(dev)
    print("Advanced mode is", "enabled" if advnc_mode.is_enabled() else "disabled")

    # Loop until we successfully enable advanced mode
    while not advnc_mode.is_enabled():
        print("Trying to enable advanced mode...")
        advnc_mode.toggle_advanced_mode(True)
        # At this point the device will disconnect and re-connect.
        print("Sleeping for 5 seconds...")
        time.sleep(5)
        # The 'dev' object will become invalid and we need to initialize it again
        dev = find_device_that_supports_advanced_mode()
        advnc_mode = rs.rs400_advanced_mode(dev)
        print("Advanced mode is", "enabled" if advnc_mode.is_enabled() else "disabled")

    depth_table = advnc_mode.get_depth_table()
    depth_table.depthClampMax = 1000
    depth_table.depthClampMin = 500
    advnc_mode.set_depth_table(depth_table)
except Exception as e:
    print(e)
pass

source = read_point_cloud("../data/heart_scan_processed.txt", format="xyz")
# source.normals = read_point_cloud("heart_scan_normals.txt", format="xyz")
source.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
draw_geometries([source])

test = read_point_cloud("../data/heart_scan_scaled.txt", format="xyz")

"""pipe = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
profile = pipe.start(config)
 
try:
    # pointcloud from realsense camera
    pcd = PointCloud()
    points = rs.points()

    frames = pipe.wait_for_frames()
    depth = frames.get_depth_frame()
    color = frames.get_color_frame()

    color_raw = Image(np.array(color.get_data()))
    depth_raw = Image(np.array(depth.get_data()))
    rgbd_image = create_rgbd_image_from_color_and_depth(color_raw, depth_raw, convert_rgb_to_intensity=False)

    tt = profile.get_stream(rs.stream.depth)
    intr = tt.as_video_stream_profile().get_intrinsics()
    pinhole_camera_intrinsic = PinholeCameraIntrinsic(intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy)
    pcd = create_point_cloud_from_rgbd_image(rgbd_image, pinhole_camera_intrinsic)
    pcd.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])

    # Record points
    # pcd.map_to(color)
    # points = pcd.calculate(depth)
    np_points = np.asarray(pcd.points)
    np.savetxt('pointcloud.txt', np_points)"""

draw_geometries([test])


voxel_size = 0.01
source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(voxel_size, source, test)

result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
print(result_ransac)
draw_registration_result(source_down, target_down, result_ransac.transformation)

result_icp = refine_registration(source, target, source_fpfh, target_fpfh, voxel_size)
print(result_icp)
draw_registration_result(source, target, result_icp.transformation)

"""# Begin Open3D registration
    threshold = 0.02
    trans_init = np.asarray(
            [[1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0], 
            [0.0, 0.0, 0.0, 1.0]])
    draw_registration_result(source, test, trans_init)
    print("Initial alignment")
    evaluation = evaluate_registration(source, test, threshold, trans_init)
    print(evaluation)

    print("Apply point-to-point ICP")
    reg_p2p = registration_icp(source, test, threshold, trans_init, TransformationEstimationPointToPoint())
    print(reg_p2p)
    print("Transformation is:")
    print(reg_p2p.transformation)
    print("")
    draw_registration_result(source, test, reg_p2p.transformation)

     
finally:
    pipe.stop()"""

