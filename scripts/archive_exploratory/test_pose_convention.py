from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import pandas as pd
from scipy.spatial.transform import Rotation


CAPTURE = Path(
    r"..\cozmo-dataset\raw_dataset\single_room\c00a170fe1"
)


def depth_to_points(
    depth,
    fx,
    fy,
    cx,
    cy,
    depth_scale=1000.0,
    stride=8,
):
    h, w = depth.shape

    # RGB -> depth resolution
    fx *= w / 1920
    fy *= h / 1440
    cx *= w / 1920
    cy *= h / 1440

    sampled = depth[::stride, ::stride]

    v, u = np.indices(sampled.shape)

    u = u * stride
    v = v * stride

    z = sampled.astype(np.float64) / depth_scale

    valid = z > 0

    u = u[valid]
    v = v[valid]
    z = z[valid]

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    return np.column_stack((x, y, z))


def make_cloud(inverse_pose=False):
    depth_dir = CAPTURE / "depth"

    depth_files = sorted(
        depth_dir.glob("*.png")
    )

    odometry = pd.read_csv(
        CAPTURE / "odometry.csv"
    )

    odometry.columns = (
        odometry.columns.str.strip()
    )

    points = []

    # Sparse diagnostic reconstruction
    frame_stride = 30
    pixel_stride = 8

    for i in range(
        0,
        len(depth_files),
        frame_stride,
    ):
        depth = cv2.imread(
            str(depth_files[i]),
            cv2.IMREAD_UNCHANGED,
        )

        row = odometry.iloc[i]

        camera_points = depth_to_points(
            depth,
            float(row["fx"]),
            float(row["fy"]),
            float(row["cx"]),
            float(row["cy"]),
            stride=pixel_stride,
        )

        q = [
            float(row["qx"]),
            float(row["qy"]),
            float(row["qz"]),
            float(row["qw"]),
        ]

        R = Rotation.from_quat(q).as_matrix()

        t = np.array([
            float(row["x"]),
            float(row["y"]),
            float(row["z"]),
        ])

        if inverse_pose:
            # P_world = R.T @ (P_camera - t)
            world = (
                camera_points - t
            ) @ R
        else:
            # P_world = R @ P_camera + t
            world = (
                camera_points @ R.T
                + t
            )

        points.append(world)

    points = np.vstack(points)

    cloud = o3d.geometry.PointCloud()

    cloud.points = (
        o3d.utility.Vector3dVector(points)
    )

    cloud = cloud.voxel_down_sample(
        voxel_size=0.03
    )

    return cloud


def main():

    print("Testing pose convention A...")
    cloud_a = make_cloud(
        inverse_pose=False
    )

    print(
        "A bounds:",
        np.asarray(cloud_a.points).min(axis=0),
        "→",
        np.asarray(cloud_a.points).max(axis=0),
    )

    print("\nTesting pose convention B...")
    cloud_b = make_cloud(
        inverse_pose=True
    )

    print(
        "B bounds:",
        np.asarray(cloud_b.points).min(axis=0),
        "→",
        np.asarray(cloud_b.points).max(axis=0),
    )

    output_a = (
        Path("outputs/single_room")
        / "pose_a.ply"
    )

    output_b = (
        Path("outputs/single_room")
        / "pose_b.ply"
    )

    output_a.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    o3d.io.write_point_cloud(
        str(output_a),
        cloud_a,
    )

    o3d.io.write_point_cloud(
        str(output_b),
        cloud_b,
    )

    print("\nSaved:")
    print(output_a)
    print(output_b)


if __name__ == "__main__":
    main()
