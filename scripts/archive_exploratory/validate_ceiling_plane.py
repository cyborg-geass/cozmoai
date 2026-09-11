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


# Candidate floor and ceiling planes from the detector.
#
# Floor:
#   -0.00000 x + 1.00000 y - 0.00092 z + 1.49729 = 0
#
# Ceiling candidate:
#    0.88380 x + 0.00783 y + 0.46779 z - 5.26825 = 0
#
# IMPORTANT:
# Plane numbering in the previous output changed because this script
# uses the actual plane equation explicitly.

FLOOR = np.array(
    [-0.00000, 1.00000, -0.00092, 1.49729],
    dtype=np.float64,
)

CEILING = np.array(
    [-0.00238, 0.99999, 0.00251, -1.58137],
    dtype=np.float64,
)


def find_depth_files(depth_dir):
    files = sorted(depth_dir.glob("*.png"))

    if not files:
        files = sorted(
            p for p in depth_dir.iterdir()
            if p.is_file() and p.suffix.lower() == ".png"
        )

    if not files:
        raise RuntimeError(
            f"No depth PNG files found in {depth_dir}"
        )

    return files


def plane_distances(points, plane):
    normal = plane[:3]
    d = plane[3]

    normal = normal / np.linalg.norm(normal)

    return points @ normal + d


print("=" * 70)
print("CEILING PLANE VALIDATION")
print("=" * 70)

calibration = load_camera_calibration(CAPTURE)
odom = load_odometry(ODOMETRY)
depth_files = find_depth_files(DEPTH_DIR)

n = min(len(depth_files), len(odom))

print(f"Depth frames: {len(depth_files)}")
print(f"Odometry rows: {len(odom)}")
print(f"Frames used: {n}")

print()
print("=" * 70)
print("SAMPLING FRAMES")
print("=" * 70)

# 40 frames distributed throughout the capture.
indices = np.linspace(
    0,
    n - 1,
    40,
    dtype=int,
)

rows = []

for idx in indices:

    depth = np.asarray(
        Image.open(depth_files[idx]),
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

    floor_dist = plane_distances(
        points_world,
        FLOOR,
    )

    ceiling_dist = plane_distances(
        points_world,
        CEILING,
    )

    # Points within 2 cm / 5 cm of each plane.
    floor_inliers = np.abs(floor_dist) < 0.02
    ceiling_inliers = np.abs(ceiling_dist) < 0.02

    # Relaxed 5 cm count is also useful.
    ceiling_5cm = np.abs(ceiling_dist) < 0.05

    # For the candidate ceiling, calculate the distribution
    # only among points near the plane.
    if ceiling_inliers.any():
        ceiling_residuals = np.abs(
            ceiling_dist[ceiling_inliers]
        )

        ceiling_median_residual = np.median(
            ceiling_residuals
        )

        ceiling_p95_residual = np.percentile(
            ceiling_residuals,
            95,
        )
    else:
        ceiling_median_residual = np.nan
        ceiling_p95_residual = np.nan

    rows.append(
        {
            "frame": idx,
            "camera_x": t[0],
            "camera_y": t[1],
            "camera_z": t[2],

            "floor_2cm": int(floor_inliers.sum()),

            "ceiling_2cm": int(
                ceiling_inliers.sum()
            ),

            "ceiling_5cm": int(
                ceiling_5cm.sum()
            ),

            "ceiling_2cm_pct":
                100.0 * ceiling_inliers.mean(),

            "ceiling_median_residual":
                ceiling_median_residual,

            "ceiling_p95_residual":
                ceiling_p95_residual,
        }
    )


df = pd.DataFrame(rows)

print()
print(
    df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)

valid = df["ceiling_2cm"] > 50

print(
    f"Frames with >50 ceiling inliers (2 cm): "
    f"{valid.sum()}/{len(df)}"
)

print(
    f"Median ceiling inliers among sampled frames: "
    f"{df['ceiling_2cm'].median():.0f}"
)

print(
    f"Maximum ceiling inliers: "
    f"{df['ceiling_2cm'].max():.0f}"
)

print(
    f"Median ceiling 2cm occupancy: "
    f"{df['ceiling_2cm_pct'].median():.2f}%"
)

valid_residuals = df[
    np.isfinite(df["ceiling_median_residual"])
]

if len(valid_residuals):

    print(
        f"Median inlier residual: "
        f"{valid_residuals['ceiling_median_residual'].median():.4f} m"
    )

    print(
        f"Median P95 residual: "
        f"{valid_residuals['ceiling_p95_residual'].median():.4f} m"
    )

print()
print("=" * 70)
print("DONE")
print("=" * 70)
