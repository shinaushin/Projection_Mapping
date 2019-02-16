# License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#####################################################
##          rs400 advanced mode tutorial           ##
#####################################################

# First import the library
import pyrealsense2 as rs
from open3d import *
import numpy as np
from datetime import datetime

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
    depth_table.depthClampMax = 3000
    depth_table.depthClampMin = 0
    advnc_mode.set_depth_table(depth_table)
except Exception as e:
    print(e)
pass

pipeline = rs.pipeline()

config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8,30)

profile = pipeline.start(config)

try:
    vis = Visualizer()
    vis.create_window("Tests", width=640, height=480)
    pointcloud = PointCloud()
    geometry_added = False

    while True:
        pointcloud.clear()
        dt0 = datetime.now()

        frames = pipeline.wait_for_frames()
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
        print(type(pcd))
        pointcloud += pcd
        
        if not geometry_added:
            vis.add_geometry(pointcloud)
            geometry_added = True

        vis.update_geometry()
        vis.poll_events()
        vis.update_renderer()
        
        process_time = datetime.now() - dt0
        print("FPS: " + str(1/process_time.total_seconds()))
finally:
    pipeline.stop()
