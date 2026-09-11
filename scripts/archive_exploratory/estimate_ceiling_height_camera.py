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

# Known world-space vertical from the independently fitted
# floor plane.
WORLD_VERTICAL = np.array(
    [-0.006948, 0.99996554, 0.004543],
    dtype=np.float64,
)

# Expected physical height is used ONLY for candidate
# classification, not as the measured result.
EXPECTED_HEIGHT = 3.078

MIN_HEIGHT = 2.0
MAX_HEIGHT = 4.0

# Plane fitting.
RANSAC_ITERATIONS = 200
RANSAC_DISTANCE = 0.025

MIN_INLIERS = 250

# Plane normals must be close to vertical.
MAX_NORMAL_ANGLE = 5.0

# Spatial sampling.
MAX_DEPTH = 8.0


# ============================================================
# UTILITIES
# ============================================================

def normalize(v):

    n = np.linalg.norm(v)

    if n < 1e-12:
        raise ValueError(
            "Cannot normalize zero vector."
        )

    return v / n


def angle_between(n1, n2):

    n1 = normalize(n1)
    n2 = normalize(n2)

    d = abs(
        np.dot(n1, n2)
    )

    d = np.clip(
        d,
        -1.0,
        1.0,
    )

    return np.degrees(
        np.arccos(d)
    )


def load_camera_matrix():

    K = np.loadtxt(
        CAPTURE_DIR / "camera_matrix.csv",
        delimiter=",",
    )

    sx = 256 / 1920
    sy = 192 / 1440

    return np.array(
        [
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
        ],
        dtype=np.float64,
    )


def load_odometry():

    data = np.genfromtxt(
        CAPTURE_DIR / "odometry.csv",
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )

    return {
        name.strip(): data[name]
        for name in data.dtype.names
    }


def get_png_files(directory):

    return sorted(
        p
        for p in directory.iterdir()
        if p.is_file()
        and p.suffix.lower() == ".png"
    )


# ============================================================
# DEPTH
# ============================================================

def depth_to_camera(
    depth,
    K,
):

    fx = K[0, 0]
    fy = K[1, 1]

    cx = K[0, 2]
    cy = K[1, 2]

    h, w = depth.shape

    v, u = np.indices(
        (h, w)
    )

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
    ].astype(np.float64)

    z = d / DEPTH_SCALE

    valid = (
        np.isfinite(z)
        & (z > 0)
        & (z < MAX_DEPTH)
    )

    u = u[valid].astype(np.float64)
    v = v[valid].astype(np.float64)
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

def get_rotation(
    frame,
    odometry,
):

    idx = np.where(
        odometry["frame"] == frame
    )[0]

    if len(idx) == 0:
        return None

    i = idx[0]

    q = np.array(
        [
            odometry["qx"][i],
            odometry["qy"][i],
            odometry["qz"][i],
            odometry["qw"][i],
        ],
        dtype=np.float64,
    )

    return Rotation.from_quat(
        q
    ).as_matrix()


# ============================================================
# RANSAC PLANE
# ============================================================

def fit_plane(
    points,
    reference_normal,
):

    if len(points) < 3:
        return None

    reference_normal = normalize(
        reference_normal
    )

    rng = np.random.default_rng(
        42
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

        a = p2 - p1
        b = p3 - p1

        n = np.cross(
            a,
            b,
        )

        norm = np.linalg.norm(n)

        if norm < 1e-10:
            continue

        n /= norm

        if np.dot(
            n,
            reference_normal,
        ) < 0:

            n = -n

        angle = angle_between(
            n,
            reference_normal,
        )

        if angle > MAX_NORMAL_ANGLE:
            continue

        d = -np.dot(
            n,
            p1,
        )

        residuals = np.abs(
            points @ n + d
        )

        mask = (
            residuals
            <= RANSAC_DISTANCE
        )

        count = int(
            np.sum(mask)
        )

        if count < MIN_INLIERS:
            continue

        p95 = float(
            np.percentile(
                residuals[mask],
                95,
            )
        )

        if (
            best is None
            or count > best["count"]
        ):

            best = {
                "normal": n,
                "d": float(d),
                "mask": mask,
                "count": count,
                "p95": p95,
            }

    if best is None:
        return None

    # --------------------------------------------------------
    # Least-squares refinement
    # --------------------------------------------------------

    inliers = points[
        best["mask"]
    ]

    centroid = np.mean(
        inliers,
        axis=0,
    )

    centered = (
        inliers
        - centroid
    )

    _, _, vh = np.linalg.svd(
        centered,
        full_matrices=False,
    )

    n = vh[-1]

    n = normalize(n)

    if np.dot(
        n,
        reference_normal,
    ) < 0:

        n = -n

    d = -np.dot(
        n,
        centroid,
    )

    residuals = np.abs(
        points @ n + d
    )

    mask = (
        residuals
        <= RANSAC_DISTANCE
    )

    inlier_residuals = residuals[
        mask
    ]

    return {
        "normal": n,
        "d": float(d),
        "count": int(
            np.sum(mask)
        ),
        "p50": float(
            np.percentile(
                inlier_residuals,
                50,
            )
        ),
        "p95": float(
            np.percentile(
                inlier_residuals,
                95,
            )
        ),
    }


# ============================================================
# FRAME ESTIMATION
# ============================================================

def estimate_frame(
    frame,
    depth_files,
    confidence_files,
    K,
    odometry,
):

    depth = cv2.imread(
        str(depth_files[frame]),
        cv2.IMREAD_UNCHANGED,
    )

    confidence = cv2.imread(
        str(confidence_files[frame]),
        cv2.IMREAD_UNCHANGED,
    )

    if depth is None or confidence is None:
        return None

    mask = (
        confidence
        >= CONFIDENCE_THRESHOLD
    )

    depth = depth.copy()

    depth[~mask] = 0

    points = depth_to_camera(
        depth,
        K,
    )

    if len(points) < 1000:
        return None

    R = get_rotation(
        frame,
        odometry,
    )

    if R is None:
        return None

    # --------------------------------------------------------
    # Convert world vertical to camera coordinates.
    #
    # P_world = R.T P_camera + t
    #
    # Therefore:
    #
    # v_camera = R v_world
    # --------------------------------------------------------

    vertical_camera = normalize(
        R @ WORLD_VERTICAL
    )

    # --------------------------------------------------------
    # First find FLOOR
    # --------------------------------------------------------

    floor_plane = fit_plane(
        points,
        vertical_camera,
    )

    if floor_plane is None:
        return None

    # --------------------------------------------------------
    # Remove floor.
    # --------------------------------------------------------

    floor_distance = np.abs(
        points
        @ floor_plane["normal"]
        + floor_plane["d"]
    )

    remaining = points[
        floor_distance > 0.08
    ]

    if len(remaining) < 500:
        return None

    # --------------------------------------------------------
    # Find horizontal candidate planes.
    #
    # We need the candidate on the opposite side of the
    # camera from the floor, with expected separation.
    # --------------------------------------------------------

    ceiling_plane = fit_plane(
        remaining,
        vertical_camera,
    )

    if ceiling_plane is None:
        return None

    # --------------------------------------------------------
    # Plane separation.
    #
    # Both normals are aligned with the same vertical axis.
    # --------------------------------------------------------

    nf = normalize(
        floor_plane["normal"]
    )

    nc = normalize(
        ceiling_plane["normal"]
    )

    normal_alignment = np.dot(
        nf,
        nc,
    )

    if normal_alignment < 0:
        nc = -nc

    angle = angle_between(
        nf,
        nc,
    )

    if angle > MAX_NORMAL_ANGLE:
        return None

    # --------------------------------------------------------
    # Signed plane coordinates along floor normal.
    # --------------------------------------------------------

    floor_coordinate = (
        -floor_plane["d"]
        / np.dot(
            nf,
            vertical_camera,
        )
    )

    ceiling_coordinate = (
        -ceiling_plane["d"]
        / np.dot(
            nc,
            vertical_camera,
        )
    )

    height = abs(
        ceiling_coordinate
        - floor_coordinate
    )

    if not (
        MIN_HEIGHT
        <= height
        <= MAX_HEIGHT
    ):
        return None

    return {
        "frame": frame,
        "height": float(height),
        "floor": {
            "coordinate":
                float(floor_coordinate),
            "points":
                floor_plane["count"],
            "p50":
                floor_plane["p50"],
            "p95":
                floor_plane["p95"],
        },
        "ceiling": {
            "coordinate":
                float(ceiling_coordinate),
            "points":
                ceiling_plane["count"],
            "p50":
                ceiling_plane["p50"],
            "p95":
                ceiling_plane["p95"],
        },
        "normal_angle":
            float(angle),
    }


# ============================================================
# ROBUST STATISTICS
# ============================================================

def mad(values):

    m = np.median(values)

    return float(
        np.median(
            np.abs(
                values - m
            )
        )
    )


def bootstrap(
    values,
    iterations=10000,
):

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if len(values) < 3:
        return None

    rng = np.random.default_rng(
        42
    )

    medians = []

    for _ in range(
        iterations
    ):

        sample = rng.choice(
            values,
            len(values),
            replace=True,
        )

        medians.append(
            np.median(sample)
        )

    lower = np.percentile(
        medians,
        2.5,
    )

    upper = np.percentile(
        medians,
        97.5,
    )

    return {
        "lower": float(lower),
        "upper": float(upper),
        "half_width":
            float(
                (upper - lower)
                / 2
            ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "CAMERA-SPACE CEILING HEIGHT ESTIMATION"
    )
    print("=" * 70)

    K = load_camera_matrix()

    odometry = load_odometry()

    depth_files = get_png_files(
        CAPTURE_DIR / "depth"
    )

    confidence_files = get_png_files(
        CAPTURE_DIR / "confidence"
    )

    print(
        "Depth frames:",
        len(depth_files),
    )

    print(
        "Odometry frames:",
        len(odometry["frame"]),
    )

    estimates = []

    for frame in range(
        START_FRAME,
        END_FRAME,
        FRAME_STRIDE,
    ):

        result = estimate_frame(
            frame,
            depth_files,
            confidence_files,
            K,
            odometry,
        )

        if result is None:
            continue

        estimates.append(
            result
        )

        print(
            f"frame={frame:5d} "
            f"height={result['height']:.4f} m "
            f"| floor={result['floor']['points']} "
            f"| ceiling={result['ceiling']['points']} "
            f"| angle={result['normal_angle']:.2f}°"
        )

    print()
    print("=" * 70)
    print("RESULT")
    print("=" * 70)

    if len(estimates) < 3:

        raise RuntimeError(
            "Too few valid frame estimates."
        )

    heights = np.array(
        [
            x["height"]
            for x in estimates
        ]
    )

    median = np.median(
        heights
    )

    dispersion = mad(
        heights
    )

    sigma = (
        1.4826
        * dispersion
    )

    print(
        "Valid frames:",
        len(heights),
    )

    print(
        f"Median height: "
        f"{median:.5f} m"
    )

    print(
        f"MAD: "
        f"{dispersion:.5f} m"
    )

    print(
        f"Robust sigma: "
        f"{sigma:.5f} m"
    )

    # Robust temporal rejection.
    if sigma > 0:

        keep = (
            np.abs(
                heights
                - median
            )
            <= 3 * sigma
        )

        filtered = [
            x
            for x, k in zip(
                estimates,
                keep,
            )
            if k
        ]

    else:

        filtered = estimates

    final_heights = np.array(
        [
            x["height"]
            for x in filtered
        ]
    )

    final_height = np.median(
        final_heights
    )

    ci = bootstrap(
        final_heights
    )

    print()
    print(
        "Filtered frames:",
        len(final_heights),
        "/",
        len(estimates),
    )

    print(
        f"Final height: "
        f"{final_height:.5f} m"
    )

    if ci:

        print(
            "95% bootstrap CI:",
            f"[{ci['lower']:.5f}, "
            f"{ci['upper']:.5f}] m"
        )

        print(
            "CI half-width:",
            f"{ci['half_width'] * 100:.2f} cm"
        )

        print()

        if ci["half_width"] <= 0.015:

            print(
                "STATUS: PASS "
                "(<= 1.5 cm)"
            )

        else:

            print(
                "STATUS: FAIL "
                "(> 1.5 cm)"
            )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "pose_convention":
            "P_world = R.T @ P_camera + t",
        "measurement_space":
            "camera",
        "frame_range":
            [START_FRAME, END_FRAME],
        "frame_stride":
            FRAME_STRIDE,
        "pixel_stride":
            PIXEL_STRIDE,
        "confidence_threshold":
            CONFIDENCE_THRESHOLD,
        "estimated_height_m":
            float(final_height),
        "mad_m":
            float(
                mad(final_heights)
            ),
        "bootstrap_ci_95":
            ci,
        "valid_frames":
            len(estimates),
        "filtered_frames":
            len(filtered),
        "frames":
            estimates,
    }

    path = (
        OUTPUT_DIR
        / "ceiling_height_camera.json"
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
        )

    print()
    print(
        "Saved:",
        path,
    )


if __name__ == "__main__":
    main()
