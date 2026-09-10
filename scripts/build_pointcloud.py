from pathlib import Path
import sys

import cv2
import numpy as np
import open3d as o3d

from cozmo_ai.geometry.calibration import CameraCalibration
from cozmo_ai.geometry.backprojection import depth_to_camera_points
from cozmo_ai.geometry.pose import (
    quaternion_to_rotation_matrix,
    camera_to_world,
)
from cozmo_ai.io.odometry import load_odometry
from cozmo_ai.io.calibration import load_camera_calibration


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

FRAME_STRIDE = 8
PIXEL_STRIDE = 4
VOXEL_SIZE = 0.02


def find_depth_files(depth_dir):
    files = sorted(
        list(depth_dir.glob("*.png"))
        + list(depth_dir.glob("*.PNG"))
    )

    if not files:
        raise RuntimeError(
            f"No depth PNG files found in {depth_dir}"
        )

    return files


def main():

    if len(sys.argv) != 3:
        print(
            "Usage:\n"
            "  uv run python scripts/build_pointcloud.py "
            "<capture_dir> <output_ply>"
        )
        sys.exit(1)

    capture_dir = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    if not capture_dir.exists():
        raise FileNotFoundError(
            f"Capture directory does not exist: {capture_dir}"
        )

    depth_dir = capture_dir / "depth"
    odometry_path = capture_dir / "odometry.csv"
    camera_matrix_path = capture_dir / "camera_matrix.csv"

    print("=" * 70)
    print("PRODUCTION POINT CLOUD BUILDER")
    print("=" * 70)

    print("\nCapture:")
    print(capture_dir)

    # --------------------------------------------------------
    # Load depth files
    # --------------------------------------------------------

    depth_files = find_depth_files(depth_dir)

    print("\nDepth frames:", len(depth_files))

    # --------------------------------------------------------
    # Load calibration
    # --------------------------------------------------------

    calibration = load_camera_calibration(
        camera_matrix_path
    )

    print("\nCalibration:")
    print("RGB:", calibration.rgb_width, "x", calibration.rgb_height)
    print(
        "Depth:",
        calibration.depth_width,
        "x",
        calibration.depth_height
    )

    print(
        "RGB intrinsics:",
        calibration.fx_rgb,
        calibration.fy_rgb,
        calibration.cx_rgb,
        calibration.cy_rgb
    )

    print(
        "Depth scale:",
        calibration.depth_scale
    )

    print("\nDepth intrinsics:")
    print(calibration.depth_intrinsics)

    # --------------------------------------------------------
    # Load odometry
    # --------------------------------------------------------

    odometry = load_odometry(
        odometry_path
    )

    print("\nOdometry frames:", len(odometry))

    # --------------------------------------------------------
    # Validate correspondence
    # --------------------------------------------------------

    if len(depth_files) != len(odometry):
        print(
            "\nWARNING:"
            f" depth frames={len(depth_files)}"
            f", odometry rows={len(odometry)}"
        )

    max_frame = min(
        len(depth_files),
        len(odometry)
    )

    # --------------------------------------------------------
    # Process frames
    # --------------------------------------------------------

    all_points = []

    print("\nProcessing:")
    print("Frame stride:", FRAME_STRIDE)
    print("Pixel stride:", PIXEL_STRIDE)

    for frame_idx in range(
        0,
        max_frame,
        FRAME_STRIDE
    ):

        depth_path = depth_files[frame_idx]

        depth = cv2.imread(
            str(depth_path),
            cv2.IMREAD_UNCHANGED
        )

        if depth is None:
            print(
                f"WARNING: could not read {depth_path}"
            )
            continue

        if depth.dtype != np.uint16:
            print(
                f"WARNING: unexpected depth dtype "
                f"{depth.dtype} at frame {frame_idx}"
            )

        try:

            points_camera = depth_to_camera_points(
                depth,
                calibration,
                pixel_stride=PIXEL_STRIDE,
            )

        except Exception as exc:

            print(
                f"WARNING: backprojection failed "
                f"at frame {frame_idx}: {exc}"
            )
            continue

        if len(points_camera) == 0:
            continue

        # ----------------------------------------------------
        # Pose
        # ----------------------------------------------------

        row = odometry.iloc[frame_idx]

        rotation = quaternion_to_rotation_matrix(
            row["qx"],
            row["qy"],
            row["qz"],
            row["qw"],
        )

        translation = np.array(
            [
                row["x"],
                row["y"],
                row["z"],
            ],
            dtype=np.float64,
        )

        points_world = camera_to_world(
            points_camera,
            rotation,
            translation,
        )

        # Remove non-finite values.
        valid = np.isfinite(
            points_world
        ).all(axis=1)

        points_world = points_world[valid]

        if len(points_world):
            all_points.append(
                points_world
            )

        if (
            frame_idx == 0
            or frame_idx % 500 == 0
            or frame_idx + FRAME_STRIDE >= max_frame
        ):
            print(
                f"Processed "
                f"{frame_idx + 1}/{max_frame}"
            )

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    if not all_points:
        raise RuntimeError(
            "No valid points were generated."
        )

    points = np.vstack(
        all_points
    )

    print("\nRaw global point cloud:")

    print(
        "Points:",
        len(points)
    )

    mins = points.min(axis=0)
    maxs = points.max(axis=0)

    print(
        "Bounds:"
    )

    print(
        f"X: {mins[0]:.3f} -> {maxs[0]:.3f}"
    )

    print(
        f"Y: {mins[1]:.3f} -> {maxs[1]:.3f}"
    )

    print(
        f"Z: {mins[2]:.3f} -> {maxs[2]:.3f}"
    )

    # --------------------------------------------------------
    # Open3D voxel downsampling
    # --------------------------------------------------------

    print("\nApplying voxel downsampling...")

    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(
        points
    )

    pcd = pcd.voxel_down_sample(
        voxel_size=VOXEL_SIZE
    )

    print(
        "Downsampled points:",
        len(pcd.points)
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    success = o3d.io.write_point_cloud(
        str(output_path),
        pcd
    )

    if not success:
        raise RuntimeError(
            f"Failed to save point cloud: {output_path}"
        )

    print("\nSaved:")
    print(output_path)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
