"""Optional point-cloud inspection with Open3D."""
import numpy as np


def clean_cloud(path, voxel_mm=0.5):
    import open3d as o3d
    points = np.load(path)
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    return cloud.voxel_down_sample(voxel_mm)
