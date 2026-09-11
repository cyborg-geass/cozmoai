from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from cozmo_ai.io.calibration import load_camera_calibration
from cozmo_ai.io.odometry import load_odometry
from cozmo_ai.geometry.backprojection import depth_to_camera_points
from cozmo_ai.geometry.pose import camera_to_world


CAPTURE = Path(
    r"..\cozmo-dataset\raw_dataset\single_scan_with_ceiling\c7d28f72c6"
)

DEPTH_DIR = CAPTURE / "depth"
ODOMETRY = CAPTURE / "odometry.csv"


START_FRAME = 5200
END_FRAME = 7100

FRAME_STRIDE = 10
PIXEL_STRIDE = 2


def find_depth_files(depth_dir):
    files = sorted(depth_dir.glob("*.png"))

    if not files:
        files = sorted(
            p for p in depth_dir.iterdir()
            if p.is_file() and p.suffix.lower() == ".png"
        )

    return files


def fit_plane_ransac(points, iterations=1000, threshold=0.02):
    """
    Basic RANSAC plane fitting.
    """

    best_inliers = None
    best_plane = None

    rng = np.random.default_rng(42)

    n = len(points)

    for _ in range(iterations):

        ids = rng.choice(n, 3, replace=False)

        p1, p2, p3 = points[ids]

        v1 = p2 - p1
        v2 = p3 - p1

        normal = np.cross(v1, v2)

        norm = np.linalg.norm(normal)

        if norm < 1e-8:
            continue

        normal /= norm

        # We want a horizontal plane.
        if abs(normal[1]) < 0.98:
            continue

        d = -np.dot(normal, p1)

        distances = np.abs(points @ normal + d)

        inliers = distances < threshold

        count = np.sum(inliers)

        if best_inliers is None or count > np.sum(best_inliers):
            best_inliers = inliers
            best_plane = (normal, d)

    return best_plane, best_inliers


print("=" * 70)
print("LOCAL CEILING PLANE FIT")
print("=" * 70)

calibration = load_camera_calibration(CAPTURE)
odom = load_odometry(ODOMETRY)
depth_files = find_depth_files(DEPTH_DIR)

print(f"Depth frames: {len(depth_files)}")
print(f"Odometry rows: {len(odom)}")

points_all = []

for frame in range(
    START_FRAME,
    min(END_FRAME, len(depth_files)),
    FRAME_STRIDE,
):

    depth = np.asarray(
        Image.open(depth_files[frame]),
        dtype=np.uint16,
    )

    points_camera = depth_to_camera_points(
        depth,
        calibration,
        pixel_stride=PIXEL_STRIDE,
    )

    row = odom.iloc[frame]

    R = Rotation.from_quat(
        [
            row["qx"],
            row["qy"],
            row["qz"],
            row["qw"],
        ]
    ).as_matrix()

    t = np.array(
        [
            row["x"],
            row["y"],
            row["z"],
        ]
    )

    points_world = camera_to_world(
        points_camera,
        R,
        t,
    )

    # Candidate ceiling region.
    #
    # Based on the previously observed plane at Y ≈ 1.58 m.
    #
    # Keep a generous band initially.
    mask = (
        (points_world[:, 1] > 1.35)
        & (points_world[:, 1] < 1.80)
    )

    ceiling_points = points_world[mask]

    if len(ceiling_points):
        points_all.append(ceiling_points)

    if frame % 500 == 0:
        print(
            f"Frame {frame}: "
            f"{len(ceiling_points):,} candidate points"
        )


if not points_all:
    raise RuntimeError("No candidate ceiling points found.")


points = np.vstack(points_all)

print()
print(f"Candidate points: {len(points):,}")

# Limit extreme spatial outliers.
# We are interested in the dominant local room surface.
center = np.median(points, axis=0)

dist_from_center = np.linalg.norm(
    points - center,
    axis=1,
)

points = points[dist_from_center < 8.0]

print(
    f"Points after spatial filtering: "
    f"{len(points):,}"
)


plane, inliers = fit_plane_ransac(
    points,
    iterations=2000,
    threshold=0.02,
)

if plane is None:
    raise RuntimeError(
        "Could not fit horizontal ceiling plane."
    )

normal, d = plane

# Make normal point toward +Y.
if normal[1] < 0:
    normal = -normal
    d = -d

residuals = np.abs(
    points @ normal + d
)

inlier_points = points[inliers]

print()
print("=" * 70)
print("LOCAL CEILING PLANE")
print("=" * 70)

print(
    f"Normal: "
    f"[{normal[0]:.6f}, "
    f"{normal[1]:.6f}, "
    f"{normal[2]:.6f}]"
)

print(f"d: {d:.6f}")

print(
    f"Plane equation: "
    f"{normal[0]:.6f}x + "
    f"{normal[1]:.6f}y + "
    f"{normal[2]:.6f}z + "
    f"{d:.6f} = 0"
)

print(
    f"Inliers: "
    f"{len(inlier_points):,} / {len(points):,}"
)

print(
    f"Inlier ratio: "
    f"{100 * len(inlier_points) / len(points):.2f}%"
)

print(
    f"Y median: "
    f"{np.median(inlier_points[:, 1]):.6f} m"
)

print(
    f"Y P05: "
    f"{np.percentile(inlier_points[:, 1], 5):.6f} m"
)

print(
    f"Y P95: "
    f"{np.percentile(inlier_points[:, 1], 95):.6f} m"
)

print(
    f"Residual median: "
    f"{np.median(residuals[inliers]):.6f} m"
)

print(
    f"Residual P95: "
    f"{np.percentile(residuals[inliers], 95):.6f} m"
)

print()
print("=" * 70)
print("DONE")
print("=" * 70)
