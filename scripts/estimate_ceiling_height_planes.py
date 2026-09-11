from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


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

START_FRAME = 5200
END_FRAME = 7100

FRAME_STRIDE = 5
PIXEL_STRIDE = 2

CONFIDENCE_THRESHOLD = 2

DEPTH_SCALE = 1000.0

# Horizontal surface tolerance.
PLANE_DISTANCE = 0.025  # 2.5 cm

# RANSAC parameters.
RANSAC_ITERATIONS = 300
RANSAC_DISTANCE = 0.02  # 2 cm

MIN_INLIERS = 5000

# A valid floor/ceiling pair should be roughly horizontal.
MAX_HORIZONTAL_ANGLE = 5.0

# We expect a normal-room-height separation.
MIN_HEIGHT = 2.0
MAX_HEIGHT = 4.0


# ============================================================
# REFERENCE NORMAL
# ============================================================

FLOOR_NORMAL = np.array(
    [-0.006948, 0.99996554, 0.004543],
    dtype=np.float64,
)

CEILING_NORMAL = np.array(
    [0.001841, 0.99999669, -0.001799],
    dtype=np.float64,
)


# ============================================================
# BASIC UTILITIES
# ============================================================

def normalize(v):

    n = np.linalg.norm(v)

    if n < 1e-12:
        raise ValueError("Zero-length vector")

    return v / n


def plane_angle_degrees(n1, n2):

    n1 = normalize(n1)
    n2 = normalize(n2)

    dot = abs(np.dot(n1, n2))

    dot = np.clip(dot, -1.0, 1.0)

    return np.degrees(
        np.arccos(dot)
    )


def load_camera_matrix():

    K = np.loadtxt(
        CAPTURE_DIR / "camera_matrix.csv",
        delimiter=",",
    )

    return K.astype(np.float64)


def load_odometry():

    data = np.genfromtxt(
        CAPTURE_DIR / "odometry.csv",
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )

    columns = {}

    for original in data.dtype.names:

        columns[original.strip()] = data[
            original
        ]

    return columns


# ============================================================
# DEPTH
# ============================================================

def get_files(directory):

    return sorted(
        p
        for p in directory.iterdir()
        if p.is_file()
        and p.suffix.lower() == ".png"
    )


DEPTH_FILES = get_files(
    CAPTURE_DIR / "depth"
)

CONFIDENCE_FILES = get_files(
    CAPTURE_DIR / "confidence"
)


def depth_to_camera(depth, K):

    h, w = depth.shape

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    v, u = np.indices(
        (h, w)
    )

    u = u[::PIXEL_STRIDE, ::PIXEL_STRIDE]
    v = v[::PIXEL_STRIDE, ::PIXEL_STRIDE]

    d = depth[
        ::PIXEL_STRIDE,
        ::PIXEL_STRIDE
    ].astype(np.float64)

    z = d / DEPTH_SCALE

    valid = (
        np.isfinite(z)
        & (z > 0)
        & (z < 10)
    )

    u = u[valid]
    v = v[valid]
    z = z[valid]

    x = (
        (u - cx)
        * z
        / fx
    )

    y = (
        (v - cy)
        * z
        / fy
    )

    return np.column_stack(
        (x, y, z)
    )


# ============================================================
# POSE
# ============================================================

def get_pose(
    frame,
    odometry,
):

    matches = np.where(
        odometry["frame"] == frame
    )[0]

    if len(matches) == 0:
        return None

    i = matches[0]

    q = np.array(
        [
            odometry["qx"][i],
            odometry["qy"][i],
            odometry["qz"][i],
            odometry["qw"][i],
        ],
        dtype=np.float64,
    )

    R = Rotation.from_quat(q).as_matrix()

    t = np.array(
        [
            odometry["x"][i],
            odometry["y"][i],
            odometry["z"][i],
        ],
        dtype=np.float64,
    )

    return R, t


def transform_to_world(
    points,
    R,
    t,
):

    # Validated convention for this capture:
    #
    # P_world = R.T @ P_camera + t
    #
    return points @ R + t


# ============================================================
# FRAME LOADING
# ============================================================

def load_frame(
    frame,
    K,
    odometry,
):

    depth = cv2.imread(
        str(DEPTH_FILES[frame]),
        cv2.IMREAD_UNCHANGED,
    )

    confidence = cv2.imread(
        str(CONFIDENCE_FILES[frame]),
        cv2.IMREAD_UNCHANGED,
    )

    if depth is None:
        return None

    if confidence is None:
        return None

    confidence_mask = (
        confidence >= CONFIDENCE_THRESHOLD
    )

    depth = depth.copy()

    depth[~confidence_mask] = 0

    points_camera = depth_to_camera(
        depth,
        K,
    )

    pose = get_pose(
        frame,
        odometry,
    )

    if pose is None:
        return None

    R, t = pose

    return transform_to_world(
        points_camera,
        R,
        t,
    )


# ============================================================
# COLLECT LOCAL POINT CLOUD
# ============================================================

def collect_points():

    K_rgb = load_camera_matrix()

    sx = 256 / 1920
    sy = 192 / 1440

    K_depth = np.array(
        [
            [
                K_rgb[0, 0] * sx,
                0,
                K_rgb[0, 2] * sx,
            ],
            [
                0,
                K_rgb[1, 1] * sy,
                K_rgb[1, 2] * sy,
            ],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )

    odometry = load_odometry()

    all_points = []

    print(
        f"Collecting frames "
        f"{START_FRAME} -> {END_FRAME}"
    )

    for frame in range(
        START_FRAME,
        END_FRAME,
        FRAME_STRIDE,
    ):

        points = load_frame(
            frame,
            K_depth,
            odometry,
        )

        if points is None:
            continue

        all_points.append(
            points
        )

        if frame % 100 == 0:

            print(
                f"frame={frame:5d} "
                f"points={len(points):6d}"
            )

    if not all_points:

        raise RuntimeError(
            "No points collected."
        )

    points = np.vstack(
        all_points
    )

    return points, K_depth


# ============================================================
# RANSAC PLANE
# ============================================================

def fit_plane_ransac(
    points,
    reference_normal,
):

    if len(points) < 3:
        return None

    reference_normal = normalize(
        reference_normal
    )

    best = None

    rng = np.random.default_rng(42)

    for _ in range(
        RANSAC_ITERATIONS
    ):

        idx = rng.choice(
            len(points),
            size=3,
            replace=False,
        )

        p1, p2, p3 = points[idx]

        v1 = p2 - p1
        v2 = p3 - p1

        normal = np.cross(
            v1,
            v2,
        )

        norm = np.linalg.norm(
            normal
        )

        if norm < 1e-8:
            continue

        normal /= norm

        # Orient toward reference normal.
        if np.dot(
            normal,
            reference_normal,
        ) < 0:

            normal = -normal

        angle = plane_angle_degrees(
            normal,
            reference_normal,
        )

        if angle > MAX_HORIZONTAL_ANGLE:
            continue

        d = -np.dot(
            normal,
            p1,
        )

        distances = np.abs(
            points @ normal + d
        )

        inlier_mask = (
            distances <= RANSAC_DISTANCE
        )

        count = int(
            np.sum(inlier_mask)
        )

        if count < MIN_INLIERS:
            continue

        residuals = distances[
            inlier_mask
        ]

        p95 = np.percentile(
            residuals,
            95,
        )

        candidate = {
            "normal": normal,
            "d": float(d),
            "count": count,
            "p95": float(p95),
            "mask": inlier_mask,
        }

        if best is None:

            best = candidate

        elif (
            count > best["count"]
        ):

            best = candidate

    return best


# ============================================================
# PLANE REFINEMENT
# ============================================================

def refine_plane(
    points,
    plane,
):

    if plane is None:
        return None

    mask = plane["mask"]

    inliers = points[mask]

    if len(inliers) < 3:
        return None

    centroid = np.mean(
        inliers,
        axis=0,
    )

    centered = (
        inliers - centroid
    )

    _, _, vh = np.linalg.svd(
        centered,
        full_matrices=False,
    )

    normal = vh[-1]

    normal = normalize(
        normal
    )

    if np.dot(
        normal,
        plane["normal"],
    ) < 0:

        normal = -normal

    d = -np.dot(
        normal,
        centroid,
    )

    distances = np.abs(
        points @ normal + d
    )

    mask = (
        distances <= PLANE_DISTANCE
    )

    residuals = distances[mask]

    return {
        "normal": normal,
        "d": float(d),
        "count": int(np.sum(mask)),
        "p50": float(
            np.percentile(
                residuals,
                50,
            )
        ),
        "p95": float(
            np.percentile(
                residuals,
                95,
            )
        ),
        "points": points[mask],
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("LOCAL FLOOR / CEILING PLANE ESTIMATION")
    print("=" * 70)

    points, K_depth = collect_points()

    print()
    print(
        "Total collected points:",
        len(points),
    )

    vertical = normalize(
        FLOOR_NORMAL
    )

    print()
    print(
        "Reference floor normal:",
        vertical,
    )

    print(
        "Reference ceiling normal:",
        normalize(
            CEILING_NORMAL
        ),
    )

    print()
    print(
        "Fitting floor plane..."
    )

    # --------------------------------------------------------
    # Floor
    # --------------------------------------------------------

    floor_candidate = fit_plane_ransac(
        points,
        FLOOR_NORMAL,
    )

    if floor_candidate is None:

        raise RuntimeError(
            "Could not fit floor plane."
        )

    floor = refine_plane(
        points,
        floor_candidate,
    )

    print()
    print("FLOOR")
    print(
        "Normal:",
        floor["normal"],
    )

    print(
        "d:",
        floor["d"],
    )

    print(
        "Inliers:",
        floor["count"],
    )

    print(
        "P50 residual:",
        f"{floor['p50'] * 100:.3f} cm",
    )

    print(
        "P95 residual:",
        f"{floor['p95'] * 100:.3f} cm",
    )

    # --------------------------------------------------------
    # Remove floor points
    # --------------------------------------------------------

    floor_distance = np.abs(
        points @ floor["normal"]
        + floor["d"]
    )

    non_floor = (
        floor_distance > 0.05
    )

    remaining = points[
        non_floor
    ]

    print()
    print(
        "Points after floor removal:",
        len(remaining),
    )

    # --------------------------------------------------------
    # Ceiling
    # --------------------------------------------------------

    print()
    print(
        "Fitting ceiling plane..."
    )

    ceiling_candidate = fit_plane_ransac(
        remaining,
        CEILING_NORMAL,
    )

    if ceiling_candidate is None:

        raise RuntimeError(
            "Could not fit ceiling plane."
        )

    ceiling = refine_plane(
        remaining,
        ceiling_candidate,
    )

    print()
    print("CEILING")
    print(
        "Normal:",
        ceiling["normal"],
    )

    print(
        "d:",
        ceiling["d"],
    )

    print(
        "Inliers:",
        ceiling["count"],
    )

    print(
        "P50 residual:",
        f"{ceiling['p50'] * 100:.3f} cm",
    )

    print(
        "P95 residual:",
        f"{ceiling['p95'] * 100:.3f} cm",
    )

    # --------------------------------------------------------
    # Normal consistency
    # --------------------------------------------------------

    normal_angle = plane_angle_degrees(
        floor["normal"],
        ceiling["normal"],
    )

    print()
    print(
        "Floor/ceiling normal angle:",
        f"{normal_angle:.4f}°",
    )

    # --------------------------------------------------------
    # Height
    # --------------------------------------------------------

    # Plane equations:
    #
    # n_f · x + d_f = 0
    # n_c · x + d_c = 0
    #
    # Since normals are approximately parallel,
    # separation is approximately |d_c-d_f|.

    floor_d_vertical = np.dot(
        floor["normal"],
        vertical,
    )

    ceiling_d_vertical = np.dot(
        ceiling["normal"],
        vertical,
    )

    if abs(
        floor_d_vertical
    ) < 0.95:

        raise RuntimeError(
            "Floor normal is not sufficiently vertical."
        )

    if abs(
        ceiling_d_vertical
    ) < 0.95:

        raise RuntimeError(
            "Ceiling normal is not sufficiently vertical."
        )

    # Project both plane offsets onto the reference vertical.
    floor_height = (
        -floor["d"]
        / np.dot(
            floor["normal"],
            vertical,
        )
    )

    ceiling_height = (
        -ceiling["d"]
        / np.dot(
            ceiling["normal"],
            vertical,
        )
    )

    height = (
        ceiling_height
        - floor_height
    )

    # Normal orientation may produce the opposite sign.
    height = abs(height)

    print()
    print("=" * 70)
    print("ROOM HEIGHT")
    print("=" * 70)

    print(
        f"Floor coordinate: "
        f"{floor_height:.5f} m"
    )

    print(
        f"Ceiling coordinate: "
        f"{ceiling_height:.5f} m"
    )

    print(
        f"Estimated height: "
        f"{height:.5f} m"
    )

    print(
        f"Estimated height: "
        f"{height * 100:.2f} cm"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    result = {
        "capture": str(
            CAPTURE_DIR
        ),
        "frames": [
            START_FRAME,
            END_FRAME,
        ],
        "frame_stride": FRAME_STRIDE,
        "pixel_stride": PIXEL_STRIDE,
        "confidence_threshold":
            CONFIDENCE_THRESHOLD,
        "pose_convention":
            "P_world = R.T @ P_camera + t",
        "floor": {
            "normal":
                floor["normal"].tolist(),
            "d": floor["d"],
            "inliers":
                floor["count"],
            "p50_residual_m":
                floor["p50"],
            "p95_residual_m":
                floor["p95"],
        },
        "ceiling": {
            "normal":
                ceiling["normal"].tolist(),
            "d": ceiling["d"],
            "inliers":
                ceiling["count"],
            "p50_residual_m":
                ceiling["p50"],
            "p95_residual_m":
                ceiling["p95"],
        },
        "normal_angle_degrees":
            float(normal_angle),
        "floor_coordinate_m":
            float(floor_height),
        "ceiling_coordinate_m":
            float(ceiling_height),
        "height_m":
            float(height),
    }

    output_path = (
        OUTPUT_DIR
        / "ceiling_height_planes.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            result,
            f,
            indent=2,
        )

    print()
    print(
        "Saved:",
        output_path,
    )


if __name__ == "__main__":
    main()
