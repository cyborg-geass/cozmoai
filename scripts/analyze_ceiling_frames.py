from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from cozmo_ai.io.calibration import load_camera_calibration
from cozmo_ai.io.odometry import load_odometry
from cozmo_ai.geometry.backprojection import depth_to_camera_points
from cozmo_ai.geometry.pose import (
    quaternion_to_rotation_matrix,
    camera_to_world,
)


CAPTURE = Path(
    r"..\cozmo-dataset\raw_dataset\single_scan_with_ceiling\c7d28f72c6"
)

DEPTH_DIR = CAPTURE / "depth"

ODOMETRY = CAPTURE / "odometry.csv"


def find_depth_files(depth_dir):
    files = sorted(depth_dir.glob("*.png"))

    if not files:
        files = sorted(
            p for p in depth_dir.iterdir()
            if p.is_file() and p.suffix.lower() == ".png"
        )

    if not files:
        raise RuntimeError(f"No depth PNG files found in {depth_dir}")

    return files


print("=" * 70)
print("LOCAL CEILING / FLOOR FRAME ANALYSIS")
print("=" * 70)

calibration = load_camera_calibration(CAPTURE)

odom = load_odometry(ODOMETRY)

depth_files = find_depth_files(DEPTH_DIR)

print(f"Depth frames: {len(depth_files)}")
print(f"Odometry rows: {len(odom)}")

# Sample approximately 20 frames throughout the trajectory.
indices = np.linspace(
    0,
    min(len(depth_files), len(odom)) - 1,
    20,
    dtype=int,
)

results = []

for idx in indices:

    depth_path = depth_files[idx]

    depth = np.asarray(
        Image.open(depth_path),
        dtype=np.uint16,
    )

    points_camera = depth_to_camera_points(
        depth,
        calibration,
        pixel_stride=2,
    )

    row = odom.iloc[idx]

    R = quaternion_to_rotation_matrix(
        row["qx"],
        row["qy"],
        row["qz"],
        row["qw"],
    )

    t = np.array(
        [
            row["x"],
            row["y"],
            row["z"],
        ],
        dtype=np.float64,
    )

    points_world = camera_to_world(
        points_camera,
        R,
        t,
    )

    # Robust percentiles of world Y.
    y = points_world[:, 1]

    results.append(
        {
            "frame": idx,
            "camera_x": t[0],
            "camera_y": t[1],
            "camera_z": t[2],
            "depth_points": len(points_world),
            "y_p01": np.percentile(y, 1),
            "y_p05": np.percentile(y, 5),
            "y_p25": np.percentile(y, 25),
            "y_median": np.percentile(y, 50),
            "y_p75": np.percentile(y, 75),
            "y_p95": np.percentile(y, 95),
            "y_p99": np.percentile(y, 99),
        }
    )


df = pd.DataFrame(results)

print()
print("=" * 70)
print("FRAME SUMMARY")
print("=" * 70)

print(
    df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)

print()
print("=" * 70)
print("DONE")
print("=" * 70)
