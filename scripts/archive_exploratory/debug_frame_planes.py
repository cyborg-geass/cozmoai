from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

CAPTURE_DIR = Path(
    "../cozmo-dataset/raw_dataset/"
    "single_scan_with_ceiling/c7d28f72c6"
)

OUTPUT_DIR = Path(
    "outputs/single_scan_with_ceiling"
)

FRAMES = [5605, 5610]

PIXEL_STRIDE = 2
CONFIDENCE_THRESHOLD = 2
DEPTH_SCALE = 1000.0

WORLD_VERTICAL = np.array(
    [-0.006948, 0.99996554, 0.004543],
    dtype=np.float64,
)

RANSAC_ITERATIONS = 500
RANSAC_DISTANCE = 0.025
MAX_NORMAL_ANGLE = 5.0
MIN_INLIERS = 200


# ============================================================
# HELPERS
# ============================================================

def normalize(v):
    return v / np.linalg.norm(v)


def load_camera_matrix():

    K = np.loadtxt(
        CAPTURE_DIR / "camera_matrix.csv",
        delimiter=",",
    )

    sx = 256 / 1920
    sy = 192 / 1440

    return np.array([
        [
            K[0, 0] * sx,
            0,
            K[0, 2] * sx,
        ],
        [
            0,
            K[1, 1] * sy,
            K[1, 2] * sy,
        ],
        [0, 0, 1],
    ])


def load_odometry():

    data = np.genfromtxt(
        CAPTURE_DIR / "odometry.csv",
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )

    return {
        n.strip(): data[n]
        for n in data.dtype.names
    }


def get_files(directory):

    return sorted(
        p
        for p in directory.iterdir()
        if p.is_file()
        and p.suffix.lower() == ".png"
    )


def get_pose(frame, odometry):

    idx = np.where(
        odometry["frame"] == frame
    )[0]

    if len(idx) == 0:
        return None

    i = idx[0]

    q = np.array([
        odometry["qx"][i],
        odometry["qy"][i],
        odometry["qz"][i],
        odometry["qw"][i],
    ])

    R = Rotation.from_quat(q).as_matrix()

    t = np.array([
        odometry["x"][i],
        odometry["y"][i],
        odometry["z"][i],
    ])

    return R, t


def depth_to_camera(depth, K):

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    h, w = depth.shape

    v, u = np.indices((h, w))

    u = u[
        ::PIXEL_STRIDE,
        ::PIXEL_STRIDE,
    ]

    v = v[
        ::PIXEL_STRIDE,
        ::PIXEL_STRIDE,
    ]

    d = depth[
        ::PIXEL_STRIDE,
        ::PIXEL_STRIDE,
    ].astype(float)

    z = d / DEPTH_SCALE

    valid = (
        np.isfinite(z)
        & (z > 0)
        & (z < 8)
    )

    u = u[valid]
    v = v[valid]
    z = z[valid]

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    return np.column_stack([
        x,
        y,
        z,
    ])


# ============================================================
# RANSAC
# ============================================================

def angle(n1, n2):

    n1 = normalize(n1)
    n2 = normalize(n2)

    return np.degrees(
        np.arccos(
            np.clip(
                abs(np.dot(n1, n2)),
                -1,
                1,
            )
        )
    )


def ransac_plane(
    points,
    reference_normal,
):

    rng = np.random.default_rng(42)

    reference_normal = normalize(
        reference_normal
    )

    best = None

    for _ in range(
        RANSAC_ITERATIONS
    ):

        ids = rng.choice(
            len(points),
            3,
            replace=False,
        )

        p1, p2, p3 = points[ids]

        n = np.cross(
            p2 - p1,
            p3 - p1,
        )

        norm = np.linalg.norm(n)

        if norm < 1e-8:
            continue

        n /= norm

        if np.dot(
            n,
            reference_normal,
        ) < 0:

            n = -n

        if angle(
            n,
            reference_normal,
        ) > MAX_NORMAL_ANGLE:
            continue

        d = -np.dot(
            n,
            p1,
        )

        residual = np.abs(
            points @ n + d
        )

        mask = (
            residual
            <= RANSAC_DISTANCE
        )

        count = np.sum(mask)

        if count < MIN_INLIERS:
            continue

        if (
            best is None
            or count > best["count"]
        ):

            best = {
                "normal": n,
                "d": d,
                "mask": mask,
                "count": int(count),
                "residual": residual,
            }

    return best


# ============================================================
# FRAME
# ============================================================

def process_frame(
    frame,
    K,
    odometry,
    depth_files,
    confidence_files,
):

    depth = cv2.imread(
        str(depth_files[frame]),
        cv2.IMREAD_UNCHANGED,
    )

    confidence = cv2.imread(
        str(confidence_files[frame]),
        cv2.IMREAD_UNCHANGED,
    )

    depth[
        confidence < CONFIDENCE_THRESHOLD
    ] = 0

    points_camera = depth_to_camera(
        depth,
        K,
    )

    pose = get_pose(
        frame,
        odometry,
    )

    R, t = pose

    # Keep both coordinate systems.
    points_world = (
        points_camera @ R
        + t
    )

    vertical_camera = normalize(
        R @ WORLD_VERTICAL
    )

    # --------------------------------------------------------
    # Floor candidate
    # --------------------------------------------------------

    floor = ransac_plane(
        points_camera,
        vertical_camera,
    )

    if floor is None:
        print("No floor plane.")
        return

    # Remove floor.
    remaining_mask = (
        floor["residual"]
        > 0.08
    )

    remaining = points_camera[
        remaining_mask
    ]

    # --------------------------------------------------------
    # Ceiling candidate
    # --------------------------------------------------------

    ceiling = ransac_plane(
        remaining,
        vertical_camera,
    )

    if ceiling is None:
        print("No ceiling plane.")
        return

    print()
    print("=" * 70)
    print(
        f"FRAME {frame}"
    )
    print("=" * 70)

    print()
    print(
        "Camera points:",
        len(points_camera),
    )

    print()
    print("FLOOR")

    print(
        "normal:",
        floor["normal"],
    )

    print(
        "d:",
        floor["d"],
    )

    print(
        "inliers:",
        floor["count"],
    )

    print(
        "P95:",
        np.percentile(
            floor["residual"][
                floor["mask"]
            ],
            95,
        ) * 100,
        "cm",
    )

    print()
    print("CEILING")

    print(
        "normal:",
        ceiling["normal"],
    )

    print(
        "d:",
        ceiling["d"],
    )

    print(
        "inliers:",
        ceiling["count"],
    )

    print(
        "P95:",
        np.percentile(
            ceiling["residual"][
                ceiling["mask"]
            ],
            95,
        ) * 100,
        "cm",
    )

    print()
    print(
        "Plane normal angle:",
        angle(
            floor["normal"],
            ceiling["normal"],
        ),
    )

    # --------------------------------------------------------
    # Coordinates
    # --------------------------------------------------------

    floor_points = points_camera[
        floor["mask"]
    ]

    ceiling_points = remaining[
        ceiling["mask"]
    ]

    # Camera vertical coordinate.
    floor_h = (
        floor_points
        @ vertical_camera
    )

    ceiling_h = (
        ceiling_points
        @ vertical_camera
    )

    print()
    print(
        "Floor vertical coordinate:",
        np.median(floor_h),
    )

    print(
        "Ceiling vertical coordinate:",
        np.median(ceiling_h),
    )

    print(
        "Estimated separation:",
        abs(
            np.median(ceiling_h)
            - np.median(floor_h)
        ),
        "m",
    )

    # --------------------------------------------------------
    # Visualization
    # --------------------------------------------------------

    fig = plt.figure(
        figsize=(12, 8)
    )

    ax = fig.add_subplot(
        111,
        projection="3d",
    )

    # Downsample for plotting.
    sample = points_camera[
        ::10
    ]

    ax.scatter(
        sample[:, 0],
        sample[:, 2],
        sample[:, 1],
        s=1,
        alpha=0.03,
    )

    # Floor.
    fp = floor_points[
        ::max(
            1,
            len(floor_points) // 3000,
        )
    ]

    ax.scatter(
        fp[:, 0],
        fp[:, 2],
        fp[:, 1],
        s=4,
        label="Detected floor",
    )

    # Ceiling.
    cp = ceiling_points[
        ::max(
            1,
            len(ceiling_points) // 3000,
        )
    ]

    ax.scatter(
        cp[:, 0],
        cp[:, 2],
        cp[:, 1],
        s=4,
        label="Detected ceiling",
    )

    ax.set_xlabel(
        "Camera X (m)"
    )

    ax.set_ylabel(
        "Camera Z (m)"
    )

    ax.set_zlabel(
        "Camera Y (m)"
    )

    ax.set_title(
        f"Frame {frame}: "
        "Detected Horizontal Planes"
    )

    ax.legend()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        OUTPUT_DIR
        / f"debug_planes_{frame}.png"
    )

    plt.savefig(
        path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print()
    print(
        "Saved visualization:",
        path,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    K = load_camera_matrix()

    odometry = load_odometry()

    depth_files = get_files(
        CAPTURE_DIR / "depth"
    )

    confidence_files = get_files(
        CAPTURE_DIR / "confidence"
    )

    for frame in FRAMES:

        process_frame(
            frame,
            K,
            odometry,
            depth_files,
            confidence_files,
        )


if __name__ == "__main__":
    main()
