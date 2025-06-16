import open3d as o3d
import numpy as np

def visualize_rgbd(rgb_path, depth_path, intrinsics, depth_scale=6553.5):
    rgb = o3d.io.read_image(rgb_path)
    depth = o3d.io.read_image(depth_path)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb, depth, depth_scale=depth_scale, convert_rgb_to_intensity=False
    )
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        intrinsics["width"], intrinsics["height"], intrinsics["fx"], intrinsics["fy"], intrinsics["cx"], intrinsics["cy"]
    )
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
    o3d.visualization.draw_geometries([pcd])

# 从 MAGiC-SLAM 配置文件获取内参
intrinsics = {
    "width": 1200, "height": 680, "fx": 600.0, "fy": 600.0, "cx": 599.5, "cy": 339.5
}  # 需确认
visualize_rgbd(
    "Data/Office-0/office_0_part1/results/frame000000.jpg",
    "Data/Office-0/office_0_part1/results/depth000000.png",
    intrinsics
)