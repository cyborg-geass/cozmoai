from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


# ============================================================
# CONFIGURATION
# ============================================================

CAPTURE_DIR = Path("../cozmo-dataset/raw_dataset/single_scan_with_ceiling/c7d28f72c6")
OUTPUT_DIR = Path("outputs/single_scan_with_ceiling")

START_FRAME = 5000
END_FRAME = 7200

FRAME_STRIDE = 10
PIXEL_STRIDE = 2

CONFIDENCE_THRESHOLD = 2

# Broad vertical range. These are NOT the final floor/ceiling
# values; they simply remove obviously irrelevant geometry.
MIN_HEIGHT = -2.5
MAX_HEIGHT = 3.0

# Minimum number of points required to consider a surface.
MIN_SURFACE_POINTS = 100

# Surface clustering resolution.
HEIGHT_BIN = 0.01  # 1 cm


# Reference planes obtained independently from the local fits.
FLOOR_NORMAL = np.array(
    [-0.006948, 0.99996554, 0.004543],
    dtype=np.float64,
)

CEILING_NORMAL = np.array(
    [0.001841, 0.99999669, -0.001799],
    dtype=np.float64,
)

FLOOR_D = 1.477840
CEILING_D = -1.584690


# ============================================================
# IO
# ============================================================

def load_camera_matrix(capture_dir: Path) -> np.ndarray:
    path = capture_dir / "camera_matrix.csv"

    K = np.loadtxt(path, delimiter=",")

    if K.shape != (3, 3):
        raise ValueError(f"Expected 3x3 camera matrix, got {K.shape}")

    return K.astype(np.float64)


def load_odometry(capture_dir: Path):
    path = capture_dir / "odometry.csv"

    data = np.genfromtxt(
        path,
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )

    # Strip whitespace from field names.
    names = [name.strip() for name in data.dtype.names]

    # Build a dictionary so whitespace in CSV headers doesn't matter.
    columns = {
        clean: data[original]
        for clean, original in zip(names, data.dtype.names)
    }

    required = [
        "frame",
        "x",
        "y",
        "z",
        "qx",
        "qy",
        "qz",
        "qw",
    ]

    for name in required:
        if name not in columns:
            raise ValueError(
                f"Missing odometry column '{name}'. "
                f"Available columns: {names}"
            )

    return columns


# ============================================================
# GEOMETRY
# ============================================================

def depth_to_camera_points(
    depth: np.ndarray,
    K_depth: np.ndarray,
    pixel_stride: int = 1,
):
    """
    Convert the 256x192 depth image into camera-frame 3D points.

    Depth values are millimeters.
    Output points are meters.
    """

    if depth.ndim != 2:
        raise ValueError("Depth image must be 2D.")

    h, w = depth.shape

    if pixel_stride < 1:
        raise ValueError("pixel_stride must be >= 1")

    fx = K_depth[0, 0]
    fy = K_depth[1, 1]
    cx = K_depth[0, 2]
    cy = K_depth[1, 2]

    v, u = np.indices((h, w))

    u = u[::pixel_stride, ::pixel_stride]
    v = v[::pixel_stride, ::pixel_stride]

    depth_sampled = depth[
        ::pixel_stride,
        ::pixel_stride,
    ].astype(np.float64)

    z = depth_sampled / 1000.0

    valid = (
        np.isfinite(z)
        & (z > 0)
        & (z < 10.0)
    )

    u = u[valid].astype(np.float64)
    v = v[valid].astype(np.float64)
    z = z[valid]

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    return np.column_stack((x, y, z))


def quaternion_to_rotation_matrix(
    qx,
    qy,
    qz,
    qw,
):
    return Rotation.from_quat(
        np.array(
            [qx, qy, qz, qw],
            dtype=np.float64,
        )
    ).as_matrix()


def camera_to_world(
    points_camera: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
):
    """
    Dataset convention validated for this capture:

        P_world = R.T @ P_camera + t

    Since points are stored row-wise:

        P_world = P_camera @ R + t
    """

    return points_camera @ rotation + translation


# ============================================================
# PLANE / HEIGHT FUNCTIONS
# ============================================================

def normalize(v):
    norm = np.linalg.norm(v)

    if norm < 1e-12:
        raise ValueError("Cannot normalize near-zero vector.")

    return v / norm


def signed_plane_distance(
    points: np.ndarray,
    normal: np.ndarray,
    d: float,
):
    normal = normalize(normal)

    return points @ normal + d


def project_height(
    points: np.ndarray,
    vertical: np.ndarray,
):
    """
    Signed coordinate along the vertical axis.

    The absolute origin is arbitrary, but because we use the
    same coordinate system for floor and ceiling, their
    difference gives the height.
    """

    vertical = normalize(vertical)

    return points @ vertical


def robust_median(values):
    if len(values) == 0:
        return np.nan

    return float(np.median(values))


def mad(values):
    if len(values) == 0:
        return np.nan

    med = np.median(values)

    return float(
        np.median(
            np.abs(values - med)
        )
    )


# ============================================================
# HEIGHT HISTOGRAM / SURFACE DETECTION
# ============================================================

def build_height_histogram(
    heights: np.ndarray,
    bin_size: float,
):
    if len(heights) == 0:
        return np.array([]), np.array([])

    minimum = np.floor(
        heights.min() / bin_size
    ) * bin_size

    maximum = np.ceil(
        heights.max() / bin_size
    ) * bin_size

    edges = np.arange(
        minimum,
        maximum + bin_size,
        bin_size,
    )

    counts, edges = np.histogram(
        heights,
        bins=edges,
    )

    centers = (
        edges[:-1] + edges[1:]
    ) / 2.0

    return centers, counts


def find_surface_candidates(
    heights: np.ndarray,
    bin_size: float = 0.01,
):
    """
    Find horizontal-surface candidates from the 1D vertical
    distribution.

    We deliberately do NOT assume that the reference global
    floor/ceiling planes will perfectly align with every frame.
    """

    if len(heights) < MIN_SURFACE_POINTS:
        return []

    centers, counts = build_height_histogram(
        heights,
        bin_size,
    )

    if len(counts) == 0:
        return []

    # Smooth the histogram to make individual depth samples
    # less influential.
    kernel = np.ones(5) / 5.0

    smoothed = np.convolve(
        counts.astype(np.float64),
        kernel,
        mode="same",
    )

    candidates = []

    for i in range(1, len(smoothed) - 1):

        if (
            smoothed[i] >= smoothed[i - 1]
            and smoothed[i] >= smoothed[i + 1]
        ):
            if smoothed[i] < MIN_SURFACE_POINTS:
                continue

            candidates.append(
                {
                    "height": float(centers[i]),
                    "count": float(smoothed[i]),
                }
            )

    candidates.sort(
        key=lambda x: x["count"],
        reverse=True,
    )

    return candidates


# ============================================================
# SURFACE EXTRACTION
# ============================================================

def extract_surface_points(
    heights: np.ndarray,
    target_height: float,
    tolerance: float = 0.03,
):
    """
    Extract points close to a candidate horizontal surface.
    """

    mask = np.abs(
        heights - target_height
    ) <= tolerance

    return heights[mask]


def estimate_surface(
    heights: np.ndarray,
    target_height: float,
    tolerance: float = 0.03,
):
    values = extract_surface_points(
        heights,
        target_height,
        tolerance,
    )

    if len(values) < MIN_SURFACE_POINTS:
        return None

    median = robust_median(values)

    surface_mad = mad(values)

    p05 = float(np.percentile(values, 5))
    p95 = float(np.percentile(values, 95))

    return {
        "height": median,
        "points": int(len(values)),
        "mad": surface_mad,
        "p05": p05,
        "p95": p95,
    }


# ============================================================
# FRAME PROCESSING
# ============================================================

def get_depth_path(capture_dir: Path, frame: int):
    depth_dir = capture_dir / "depth"

    files = sorted(
        p for p in depth_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() == ".png"
    )

    if frame >= len(files):
        raise IndexError(
            f"Frame {frame} unavailable. "
            f"Only {len(files)} depth frames."
        )

    return files[frame]


def get_confidence_path(capture_dir: Path, frame: int):
    confidence_dir = capture_dir / "confidence"

    files = sorted(
        p for p in confidence_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() == ".png"
    )

    if frame >= len(files):
        raise IndexError(
            f"Frame {frame} unavailable. "
            f"Only {len(files)} confidence frames."
        )

    return files[frame]


def process_frame(
    frame: int,
    capture_dir: Path,
    odometry,
    K_depth,
    vertical,
):
    depth_path = get_depth_path(
        capture_dir,
        frame,
    )

    confidence_path = get_confidence_path(
        capture_dir,
        frame,
    )

    depth = cv2.imread(
        str(depth_path),
        cv2.IMREAD_UNCHANGED,
    )

    confidence = cv2.imread(
        str(confidence_path),
        cv2.IMREAD_UNCHANGED,
    )

    if depth is None:
        raise RuntimeError(
            f"Could not read depth: {depth_path}"
        )

    if confidence is None:
        raise RuntimeError(
            f"Could not read confidence: {confidence_path}"
        )

    if depth.shape != confidence.shape:
        raise RuntimeError(
            f"Depth/confidence shape mismatch: "
            f"{depth.shape} vs {confidence.shape}"
        )

    # --------------------------------------------------------
    # Confidence filtering
    # --------------------------------------------------------

    confidence_mask = (
        confidence >= CONFIDENCE_THRESHOLD
    )

    depth = depth.copy()

    depth[~confidence_mask] = 0

    # --------------------------------------------------------
    # Camera → world
    # --------------------------------------------------------

    points_camera = depth_to_camera_points(
        depth,
        K_depth,
        PIXEL_STRIDE,
    )

    if len(points_camera) == 0:
        return None

    # Odometry indexing is frame-based.
    frame_indices = odometry["frame"]

    matches = np.where(
        frame_indices == frame
    )[0]

    if len(matches) == 0:
        return None

    idx = matches[0]

    translation = np.array(
        [
            odometry["x"][idx],
            odometry["y"][idx],
            odometry["z"][idx],
        ],
        dtype=np.float64,
    )

    rotation = quaternion_to_rotation_matrix(
        odometry["qx"][idx],
        odometry["qy"][idx],
        odometry["qz"][idx],
        odometry["qw"][idx],
    )

    points_world = camera_to_world(
        points_camera,
        rotation,
        translation,
    )

    # --------------------------------------------------------
    # Convert to vertical coordinates
    # --------------------------------------------------------

    heights = project_height(
        points_world,
        vertical,
    )

    valid = (
        np.isfinite(heights)
        & (heights >= MIN_HEIGHT)
        & (heights <= MAX_HEIGHT)
    )

    heights = heights[valid]

    if len(heights) < MIN_SURFACE_POINTS:
        return None

    return heights


# ============================================================
# TEMPORAL ESTIMATION
# ============================================================

def estimate_window(
    frames,
    capture_dir,
    odometry,
    K_depth,
    vertical,
):
    all_heights = []

    for frame in frames:

        result = process_frame(
            frame,
            capture_dir,
            odometry,
            K_depth,
            vertical,
        )

        if result is None:
            continue

        all_heights.append(result)

    if not all_heights:
        return None

    heights = np.concatenate(
        all_heights
    )

    if len(heights) < MIN_SURFACE_POINTS:
        return None

    candidates = find_surface_candidates(
        heights,
        HEIGHT_BIN,
    )

    if len(candidates) < 2:
        return None

    # --------------------------------------------------------
    # We expect floor and ceiling to be separated by roughly
    # a few metres. Select the strongest low and high modes.
    # --------------------------------------------------------

    candidates_sorted = sorted(
        candidates,
        key=lambda x: x["height"],
    )

    low_candidates = [
        c for c in candidates_sorted
        if c["height"] < 0
    ]

    high_candidates = [
        c for c in candidates_sorted
        if c["height"] > 0
    ]

    if not low_candidates or not high_candidates:
        return None

    floor_candidate = max(
        low_candidates,
        key=lambda x: x["count"],
    )

    ceiling_candidate = max(
        high_candidates,
        key=lambda x: x["count"],
    )

    floor = estimate_surface(
        heights,
        floor_candidate["height"],
        tolerance=0.03,
    )

    ceiling = estimate_surface(
        heights,
        ceiling_candidate["height"],
        tolerance=0.03,
    )

    if floor is None or ceiling is None:
        return None

    height = (
        ceiling["height"]
        - floor["height"]
    )

    return {
        "floor": floor,
        "ceiling": ceiling,
        "height": float(height),
        "total_points": int(len(heights)),
    }


# ============================================================
# ROBUST AGGREGATION
# ============================================================

def robust_filter(
    estimates,
    threshold=0.10,
):
    """
    Remove grossly inconsistent height estimates.

    threshold is deliberately loose here. We are not claiming
    centimetre-level accuracy through this filter.
    """

    heights = np.array(
        [e["height"] for e in estimates],
        dtype=np.float64,
    )

    median = np.median(heights)
    deviation = np.abs(
        heights - median
    )

    keep = deviation <= threshold

    return [
        e
        for e, valid in zip(
            estimates,
            keep,
        )
        if valid
    ]


def bootstrap_ci(
    estimates,
    iterations=5000,
):
    heights = np.array(
        [e["height"] for e in estimates],
        dtype=np.float64,
    )

    if len(heights) < 2:
        return None

    rng = np.random.default_rng(42)

    bootstrapped = np.empty(
        iterations,
        dtype=np.float64,
    )

    n = len(heights)

    for i in range(iterations):

        sample = rng.choice(
            heights,
            size=n,
            replace=True,
        )

        bootstrapped[i] = np.median(
            sample
        )

    return {
        "lower": float(
            np.percentile(
                bootstrapped,
                2.5,
            )
        ),
        "upper": float(
            np.percentile(
                bootstrapped,
                97.5,
            )
        ),
        "half_width": float(
            (
                np.percentile(
                    bootstrapped,
                    97.5,
                )
                -
                np.percentile(
                    bootstrapped,
                    2.5,
                )
            ) / 2
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("LOCAL CEILING HEIGHT ESTIMATION")
    print("=" * 70)

    print()
    print("Capture:")
    print(CAPTURE_DIR)

    print()
    print("Frame range:")
    print(
        f"{START_FRAME} -> {END_FRAME}"
    )

    print()
    print("Frame stride:", FRAME_STRIDE)
    print("Pixel stride:", PIXEL_STRIDE)
    print(
        "Confidence threshold:",
        CONFIDENCE_THRESHOLD,
    )

    # --------------------------------------------------------
    # Load calibration
    # --------------------------------------------------------

    K_rgb = load_camera_matrix(
        CAPTURE_DIR
    )

    RGB_W = 1920
    RGB_H = 1440

    DEPTH_W = 256
    DEPTH_H = 192

    sx = DEPTH_W / RGB_W
    sy = DEPTH_H / RGB_H

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

    print()
    print("Depth intrinsics:")
    print(K_depth)

    # --------------------------------------------------------
    # Load odometry
    # --------------------------------------------------------

    odometry = load_odometry(
        CAPTURE_DIR
    )

    print()
    print(
        "Odometry frames:",
        len(odometry["frame"]),
    )

    # --------------------------------------------------------
    # Normalize reference vertical
    # --------------------------------------------------------

    vertical = normalize(
        FLOOR_NORMAL
    )

    ceiling_normal = normalize(
        CEILING_NORMAL
    )

    angle = np.degrees(
        np.arccos(
            np.clip(
                np.dot(
                    vertical,
                    ceiling_normal,
                ),
                -1.0,
                1.0,
            )
        )
    )

    print()
    print("Reference vertical:")
    print(vertical)

    print(
        f"Floor/ceiling normal angle: "
        f"{angle:.6f}°"
    )

    # --------------------------------------------------------
    # Estimate overlapping temporal windows
    # --------------------------------------------------------

    WINDOW_SIZE = 150
    STEP = 75

    estimates = []

    window_id = 0

    for start in range(
        START_FRAME,
        END_FRAME,
        STEP,
    ):

        end = min(
            start + WINDOW_SIZE,
            END_FRAME,
        )

        frames = range(
            start,
            end,
            FRAME_STRIDE,
        )

        window_id += 1

        result = estimate_window(
            frames,
            CAPTURE_DIR,
            odometry,
            K_depth,
            vertical,
        )

        if result is None:

            print(
                f"Window {window_id:02d} "
                f"{start}-{end}: "
                f"NO RELIABLE SURFACE PAIR"
            )

            continue

        height = result["height"]

        print(
            f"Window {window_id:02d} "
            f"{start}-{end}: "
            f"height={height:.4f} m "
            f"| floor={result['floor']['height']:.4f} "
            f"({result['floor']['points']} pts) "
            f"| ceiling={result['ceiling']['height']:.4f} "
            f"({result['ceiling']['points']} pts)"
        )

        estimates.append(
            {
                "window": window_id,
                "start_frame": start,
                "end_frame": end,
                **result,
            }
        )

    # --------------------------------------------------------
    # Check result
    # --------------------------------------------------------

    if not estimates:

        raise RuntimeError(
            "No windows produced a reliable "
            "floor/ceiling pair."
        )

    print()
    print("=" * 70)
    print("RAW WINDOW ESTIMATES")
    print("=" * 70)

    heights = np.array(
        [
            e["height"]
            for e in estimates
        ],
        dtype=np.float64,
    )

    print(
        "Number of valid windows:",
        len(heights),
    )

    print(
        "Median:",
        f"{np.median(heights):.4f} m",
    )

    print(
        "MAD:",
        f"{mad(heights):.4f} m",
    )

    # --------------------------------------------------------
    # Robust filtering
    # --------------------------------------------------------

    filtered = robust_filter(
        estimates,
        threshold=0.10,
    )

    print()
    print(
        "After robust filtering:",
        len(filtered),
        "/",
        len(estimates),
    )

    if len(filtered) < 2:

        raise RuntimeError(
            "Too few consistent windows "
            "after robust filtering."
        )

    filtered_heights = np.array(
        [
            e["height"]
            for e in filtered
        ],
        dtype=np.float64,
    )

    final_height = float(
        np.median(filtered_heights)
    )

    final_mad = mad(
        filtered_heights
    )

    robust_sigma = (
        1.4826 * final_mad
    )

    ci = bootstrap_ci(
        filtered
    )

    # --------------------------------------------------------
    # Assignment gate
    # --------------------------------------------------------

    gate = 0.015

    passed = (
        ci is not None
        and ci["half_width"] <= gate
    )

    # --------------------------------------------------------
    # Print final result
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL CEILING HEIGHT")
    print("=" * 70)

    print(
        f"Estimated height: "
        f"{final_height:.4f} m"
    )

    print(
        f"MAD: "
        f"{final_mad:.4f} m"
    )

    print(
        f"Robust sigma: "
        f"{robust_sigma:.4f} m"
    )

    if ci is not None:

        print(
            "Bootstrap 95% CI: "
            f"[{ci['lower']:.4f}, "
            f"{ci['upper']:.4f}] m"
        )

        print(
            "CI half-width: "
            f"{ci['half_width']:.4f} m "
            f"({ci['half_width'] * 100:.2f} cm)"
        )

    print()
    print(
        "Assignment ceiling-height gate:"
    )

    print(
        "Required CI half-width <= 1.5 cm"
    )

    if passed:
        print(
            "STATUS: PASS"
        )
    else:
        print(
            "STATUS: FAIL / INSUFFICIENT "
            "EVIDENCE"
        )

    # --------------------------------------------------------
    # Save output
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "capture": str(
            CAPTURE_DIR
        ),
        "frame_range": [
            START_FRAME,
            END_FRAME,
        ],
        "frame_stride": FRAME_STRIDE,
        "pixel_stride": PIXEL_STRIDE,
        "confidence_threshold":
            CONFIDENCE_THRESHOLD,
        "pose_convention":
            "P_world = R.T @ P_camera + t",
        "vertical": vertical.tolist(),
        "floor_normal":
            FLOOR_NORMAL.tolist(),
        "ceiling_normal":
            CEILING_NORMAL.tolist(),
        "normal_angle_degrees":
            float(angle),
        "raw_window_count":
            len(estimates),
        "filtered_window_count":
            len(filtered),
        "estimated_height_m":
            final_height,
        "mad_m":
            final_mad,
        "robust_sigma_m":
            robust_sigma,
        "bootstrap_ci_95":
            ci,
        "gate_half_width_m":
            gate,
        "gate_pass":
            bool(passed),
        "windows": estimates,
    }

    output_path = (
        OUTPUT_DIR
        / "ceiling_height_local.json"
    )

    with open(
        output_path,
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
        output_path,
    )


if __name__ == "__main__":
    main()
