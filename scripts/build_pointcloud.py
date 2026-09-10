import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from cozmo_ai.geometry.backprojection import (
    depth_to_camera_points,
)
from cozmo_ai.io.calibration import (
    load_camera_calibration,
)
from cozmo_ai.io.odometry import (
    get_frame_pose,
    load_odometry,
)


# ============================================================
# Configuration
# ============================================================

FRAME_STRIDE = 4
PIXEL_STRIDE = 4


# ============================================================
# Main
# ============================================================

def main():

    if len(sys.argv) != 2:

        print(
            "Usage:\n"
            'uv run python scripts\\build_pointcloud.py '
            '"..\\cozmo-dataset\\raw_dataset\\single_room\\c00a170fe1"'
        )

        sys.exit(1)

    capture_dir = Path(
        sys.argv[1]
    )

    depth_dir = (
        capture_dir / "depth"
    )

    odometry_path = (
        capture_dir / "odometry.csv"
    )

    if not depth_dir.exists():
        raise FileNotFoundError(
            f"Missing depth directory: "
            f"{depth_dir}"
        )

    if not odometry_path.exists():
        raise FileNotFoundError(
            f"Missing odometry file: "
            f"{odometry_path}"
        )

    # --------------------------------------------------------
    # Load calibration and odometry
    # --------------------------------------------------------

    calibration = (
        load_camera_calibration(
            capture_dir
        )
    )

    odometry = load_odometry(
        odometry_path
    )

    depth_files = sorted(
        depth_dir.glob("*.png")
    )

    n = min(
        len(depth_files),
        len(odometry),
    )

    print("=" * 70)
    print("BUILDING GLOBAL POINT CLOUD")
    print("=" * 70)

    print(
        f"\nFrames available: {n}"
    )

    print(
        f"Frame stride:    {FRAME_STRIDE}"
    )

    print(
        f"Pixel stride:    {PIXEL_STRIDE}"
    )

    print(
        f"Depth scale:     "
        f"{calibration.depth_scale}"
    )

    print(
        "\nDepth intrinsics:"
    )

    print(
        calibration.depth_intrinsics
    )

    # --------------------------------------------------------
    # Build global cloud
    # --------------------------------------------------------

    all_points = []

    processed = 0

    for frame_index in range(
        0,
        n,
        FRAME_STRIDE,
    ):

        depth = cv2.imread(
            str(depth_files[frame_index]),
            cv2.IMREAD_UNCHANGED,
        )

        if depth is None:

            print(
                f"WARNING: could not read "
                f"{depth_files[frame_index]}"
            )

            continue

        # --------------------------------------------
        # Depth → camera coordinates
        # --------------------------------------------

        points_camera = (
            depth_to_camera_points(
                depth,
                calibration,
                pixel_stride=PIXEL_STRIDE,
            )
        )

        # --------------------------------------------
        # Camera → world coordinates
        # --------------------------------------------

        rotation, translation = (
            get_frame_pose(
                odometry,
                frame_index,
            )
        )

        points_world = (
            rotation
            @ points_camera.T
        ).T + translation

        valid = np.all(
            np.isfinite(points_world),
            axis=1,
        )

        points_world = (
            points_world[valid]
        )

        all_points.append(
            points_world
        )

        processed += 1

        if (
            processed == 1
            or processed % 50 == 0
            or frame_index + FRAME_STRIDE >= n
        ):

            print(
                f"Processed "
                f"{frame_index + 1}/{n} "
                f"frames"
            )

    if not all_points:

        raise RuntimeError(
            "No valid points were generated."
        )

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    points = np.vstack(
        all_points
    )

    print(
        "\nRaw global point cloud:"
    )

    print(
        f"Points: {len(points):,}"
    )

    print(
        "Bounds:"
    )

    print(
        f"X: {points[:, 0].min():.3f} "
        f"→ {points[:, 0].max():.3f}"
    )

    print(
        f"Y: {points[:, 1].min():.3f} "
        f"→ {points[:, 1].max():.3f}"
    )

    print(
        f"Z: {points[:, 2].min():.3f} "
        f"→ {points[:, 2].max():.3f}"
    )

    # --------------------------------------------------------
    # Open3D point cloud
    # --------------------------------------------------------

    cloud = o3d.geometry.PointCloud()

    cloud.points = (
        o3d.utility.Vector3dVector(
            points
        )
    )

    # --------------------------------------------------------
    # Voxel downsampling
    # --------------------------------------------------------

    print(
        "\nApplying voxel downsampling..."
    )

    downsampled = (
        cloud.voxel_down_sample(
            voxel_size=0.02
        )
    )

    print(
        f"Downsampled points: "
        f"{len(downsampled.points):,}"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_dir = (
        Path("outputs")
        / "single_room"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / "pointcloud_production.ply"
    )

    success = (
        o3d.io.write_point_cloud(
            str(output_path),
            downsampled,
        )
    )

    if not success:
        raise RuntimeError(
            f"Failed to save "
            f"{output_path}"
        )

    print(
        f"\nSaved: {output_path}"
    )


if __name__ == "__main__":
    main()
