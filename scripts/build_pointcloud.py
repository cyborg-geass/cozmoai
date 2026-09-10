from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import pandas as pd

from cozmo_ai.geometry.backprojection import depth_to_points
from cozmo_ai.geometry.transforms import (
    quaternion_to_rotation_matrix,
    transform_points,
)


def load_odometry(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def build_point_cloud(
    capture_path: Path,
    output_path: Path,
    frame_stride: int = 10,
    pixel_stride: int = 4,
    depth_scale: float = 1000.0,
):
    depth_dir = capture_path / "depth"
    odometry_path = capture_path / "odometry.csv"

    odometry = load_odometry(odometry_path)

    all_points = []

    depth_files = sorted(depth_dir.glob("*.png"))

    print(f"Depth frames found: {len(depth_files)}")
    print(f"Odometry rows:      {len(odometry)}")

    if len(depth_files) != len(odometry):
        raise ValueError(
            "Depth and odometry counts do not match."
        )

    for i in range(
        0,
        min(len(depth_files), len(odometry)),
        frame_stride,
    ):
        depth_path = depth_files[i]

        depth = cv2.imread(
            str(depth_path),
            cv2.IMREAD_UNCHANGED,
        )

        if depth is None:
            print(f"WARNING: failed to read {depth_path}")
            continue

        row = odometry.iloc[i]

        # Odometry intrinsics are expressed at the RGB/native
        # camera resolution. depth_to_points() converts them
        # to the depth image resolution.
        
        fx = float(row["fx"])
        fy = float(row["fy"])
        cx = float(row["cx"])
        cy = float(row["cy"])

        points_camera = depth_to_points(
            depth=depth,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            depth_scale=depth_scale,
            stride=pixel_stride,
        )

        rotation = quaternion_to_rotation_matrix(
            float(row["qx"]),
            float(row["qy"]),
            float(row["qz"]),
            float(row["qw"]),
        )

        translation = np.array(
            [
                float(row["x"]),
                float(row["y"]),
                float(row["z"]),
            ]
        )

        points_world = transform_points(
            points_camera,
            rotation,
            translation,
        )

        all_points.append(points_world)

        if (i // frame_stride) % 10 == 0:
            print(
                f"Processed frame {i}/{len(depth_files)} "
                f"→ {len(points_world):,} points"
            )

    if not all_points:
        raise RuntimeError("No points were generated.")

    points = np.vstack(all_points)

    print()
    print("=" * 70)
    print("POINT CLOUD")
    print("=" * 70)

    print("Total points:", len(points))

    print("\nBounds:")
    print("X:", points[:, 0].min(), "→", points[:, 0].max())
    print("Y:", points[:, 1].min(), "→", points[:, 1].max())
    print("Z:", points[:, 2].min(), "→", points[:, 2].max())

    # Remove obvious numerical outliers.
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]

    cloud = o3d.geometry.PointCloud()

    cloud.points = o3d.utility.Vector3dVector(points)

    print("\nRunning voxel downsampling...")

    cloud = cloud.voxel_down_sample(
        voxel_size=0.02
    )

    print(
        "Points after downsampling:",
        len(cloud.points),
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    o3d.io.write_point_cloud(
        str(output_path),
        cloud,
    )

    print("\nSaved:", output_path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--capture",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--frame-stride",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--pixel-stride",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--depth-scale",
        type=float,
        default=1000.0,
    )

    args = parser.parse_args()

    build_point_cloud(
        capture_path=args.capture,
        output_path=args.output,
        frame_stride=args.frame_stride,
        pixel_stride=args.pixel_stride,
        depth_scale=args.depth_scale,
    )


if __name__ == "__main__":
    main()
