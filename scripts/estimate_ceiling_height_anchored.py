from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


# ============================================================
# CONFIGURATION
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

# Reference normals from independent local plane fits.
FLOOR_NORMAL = np.array(
    [-0.006948, 0.99996554, 0.004543],
    dtype=np.float64,
)

CEILING_NORMAL = np.array(
    [0.001841, 0.99999669, -0.001799],
    dtype=np.float64,
)

# Previously observed approximate surface coordinates.
# These are used only to identify the correct physical surface,
# NOT as the final height measurement.
REFERENCE_FLOOR = -1.497
REFERENCE_CEILING = 1.581

EXPECTED_HEIGHT = (
    REFERENCE_CEILING
    - REFERENCE_FLOOR
)

# Surface matching.
FLOOR_TOLERANCE = 0.08
CEILING_TOLERANCE = 0.12

# Expected room-height range.
MIN_HEIGHT = 2.5
MAX_HEIGHT = 3.5

# Require enough spatial support.
MIN_FLOOR_POINTS = 100
MIN_CEILING_POINTS = 100

# Robust outlier rejection.
MAX_LOCAL_RESIDUAL = 0.04


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

    for name in data.dtype.names:
        columns[name.strip()] = data[name]

    return columns


def get_files(directory):

    return sorted(
        p
        for p in directory.iterdir()
        if p.is_file()
        and p.suffix.lower() == ".png"
    )


# ============================================================
# CAMERA GEOMETRY
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
        & (z < 10)
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

def get_pose(
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

    R = Rotation.from_quat(
        q
    ).as_matrix()

    t = np.array(
        [
            odometry["x"][i],
            odometry["y"][i],
            odometry["z"][i],
        ],
        dtype=np.float64,
    )

    return R, t


def camera_to_world(
    points,
    R,
    t,
):

    # Validated for this capture.
    #
    # P_world = R.T @ P_camera + t
    #
    return points @ R + t


# ============================================================
# FRAME
# ============================================================

def process_frame(
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

    if depth is None:
        return None

    if confidence is None:
        return None

    mask = (
        confidence
        >= CONFIDENCE_THRESHOLD
    )

    depth = depth.copy()

    depth[~mask] = 0

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

    points_world = camera_to_world(
        points_camera,
        R,
        t,
    )

    return points_world


# ============================================================
# LOCAL PLANE ESTIMATION
# ============================================================

def estimate_plane_coordinate(
    points,
    normal,
    reference_coordinate,
    tolerance,
):

    normal = normalize(
        normal
    )

    coordinates = (
        points @ normal
    )

    mask = (
        np.abs(
            coordinates
            - reference_coordinate
        )
        <= tolerance
    )

    selected = coordinates[
        mask
    ]

    if len(selected) < 100:
        return None

    # Robust center.
    center = np.median(
        selected
    )

    residuals = np.abs(
        selected - center
    )

    # Robust rejection.
    robust_mask = (
        residuals
        <= MAX_LOCAL_RESIDUAL
    )

    selected = selected[
        robust_mask
    ]

    if len(selected) < 100:
        return None

    center = np.median(
        selected
    )

    residuals = np.abs(
        selected - center
    )

    return {
        "coordinate": float(
            center
        ),
        "points": int(
            len(selected)
        ),
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
        "mad": float(
            np.median(
                np.abs(
                    selected
                    - center
                )
            )
        ),
    }


# ============================================================
# FRAME HEIGHT
# ============================================================

def estimate_frame_height(
    points,
):

    floor = estimate_plane_coordinate(
        points,
        FLOOR_NORMAL,
        REFERENCE_FLOOR,
        FLOOR_TOLERANCE,
    )

    if floor is None:
        return None

    # Anchor the ceiling search to THIS FRAME'S
    # locally estimated floor coordinate.
    expected_ceiling = (
        floor["coordinate"]
        + EXPECTED_HEIGHT
    )

    ceiling = estimate_plane_coordinate(
        points,
        CEILING_NORMAL,
        expected_ceiling,
        CEILING_TOLERANCE,
    )

    if ceiling is None:
        return None

    height = (
        ceiling["coordinate"]
        - floor["coordinate"]
    )

    # Physical plausibility check.
    if not (
        MIN_HEIGHT
        <= height
        <= MAX_HEIGHT
    ):
        return None

    return {
        "height": float(height),
        "floor": floor,
        "ceiling": ceiling,
    }


# ============================================================
# ROBUST AGGREGATION
# ============================================================

def median_absolute_deviation(
    values,
):

    median = np.median(
        values
    )

    return float(
        np.median(
            np.abs(
                values
                - median
            )
        )
    )


def bootstrap_ci(
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

    medians = np.empty(
        iterations,
        dtype=np.float64,
    )

    for i in range(
        iterations
    ):

        sample = rng.choice(
            values,
            size=len(values),
            replace=True,
        )

        medians[i] = np.median(
            sample
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
        "half_width": float(
            (upper - lower) / 2
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "FLOOR-ANCHORED CEILING HEIGHT ESTIMATION"
    )
    print("=" * 70)

    print()
    print(
        "Expected height:",
        f"{EXPECTED_HEIGHT:.4f} m",
    )

    K_rgb = load_camera_matrix()

    sx = 256 / 1920
    sy = 192 / 1440

    K = np.array(
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

    depth_files = get_files(
        CAPTURE_DIR / "depth"
    )

    confidence_files = get_files(
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

        points = process_frame(
            frame,
            depth_files,
            confidence_files,
            K,
            odometry,
        )

        if points is None:
            continue

        result = estimate_frame_height(
            points
        )

        if result is None:

            print(
                f"frame={frame:5d}: "
                "REJECTED"
            )

            continue

        print(
            f"frame={frame:5d}: "
            f"height={result['height']:.4f} m "
            f"| floor={result['floor']['coordinate']:.4f} "
            f"({result['floor']['points']} pts) "
            f"| ceiling={result['ceiling']['coordinate']:.4f} "
            f"({result['ceiling']['points']} pts)"
        )

        result["frame"] = frame

        estimates.append(
            result
        )

    if len(estimates) < 3:

        raise RuntimeError(
            "Too few valid frame estimates."
        )

    # --------------------------------------------------------
    # Extract heights
    # --------------------------------------------------------

    heights = np.array(
        [
            x["height"]
            for x in estimates
        ],
        dtype=np.float64,
    )

    median = float(
        np.median(heights)
    )

    mad = median_absolute_deviation(
        heights
    )

    print()
    print("=" * 70)
    print("FRAME-LEVEL RESULTS")
    print("=" * 70)

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
        f"{mad:.5f} m"
    )

    print(
        f"Robust sigma: "
        f"{1.4826 * mad:.5f} m"
    )

    # --------------------------------------------------------
    # Robust frame filtering
    # --------------------------------------------------------

    sigma = 1.4826 * mad

    if sigma < 1e-6:

        filtered = estimates

    else:

        filtered = [
            x
            for x in estimates
            if abs(
                x["height"]
                - median
            )
            <= 3 * sigma
        ]

    filtered_heights = np.array(
        [
            x["height"]
            for x in filtered
        ],
        dtype=np.float64,
    )

    final_height = float(
        np.median(
            filtered_heights
        )
    )

    final_mad = median_absolute_deviation(
        filtered_heights
    )

    ci = bootstrap_ci(
        filtered_heights
    )

    print()
    print(
        "Filtered frames:",
        len(filtered),
        "/",
        len(estimates),
    )

    print(
        f"Final height: "
        f"{final_height:.5f} m"
    )

    print(
        f"Final MAD: "
        f"{final_mad:.5f} m"
    )

    if ci:

        print(
            "Bootstrap 95% CI:",
            f"[{ci['lower']:.5f}, "
            f"{ci['upper']:.5f}] m",
        )

        print(
            "CI half-width:",
            f"{ci['half_width'] * 100:.2f} cm",
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

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "pose_convention":
            "P_world = R.T @ P_camera + t",
        "frame_range": [
            START_FRAME,
            END_FRAME,
        ],
        "frame_stride":
            FRAME_STRIDE,
        "pixel_stride":
            PIXEL_STRIDE,
        "confidence_threshold":
            CONFIDENCE_THRESHOLD,
        "expected_height_m":
            float(EXPECTED_HEIGHT),
        "estimated_height_m":
            final_height,
        "mad_m":
            final_mad,
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
        / "ceiling_height_anchored.json"
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
