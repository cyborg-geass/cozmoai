from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial.transform import Rotation

from cozmo_ai.io.calibration import load_camera_calibration
from cozmo_ai.io.odometry import load_odometry
from cozmo_ai.geometry.backprojection import depth_to_camera_points
from cozmo_ai.geometry.pose import camera_to_world


# ============================================================
# CONFIGURATION
# ============================================================

CAPTURE = Path(
    r"..\cozmo-dataset\raw_dataset\single_scan_with_ceiling\c7d28f72c6"
)

DEPTH_DIR = CAPTURE / "depth"
ODOMETRY = CAPTURE / "odometry.csv"

# The ceiling was strongly observed approximately here.
START_FRAME = 5200
END_FRAME = 7100

# Temporal windows.
WINDOW_SIZE = 200
WINDOW_STEP = 100

FRAME_STRIDE = 5
PIXEL_STRIDE = 2

# Candidate vertical bands.
FLOOR_MIN_Y = -1.75
FLOOR_MAX_Y = -1.25

CEILING_MIN_Y = 1.35
CEILING_MAX_Y = 1.80

# RANSAC.
RANSAC_ITERATIONS = 1000
RANSAC_THRESHOLD = 0.02

# Minimum number of points for a reliable local plane.
MIN_CANDIDATE_POINTS = 500
MIN_INLIERS = 500

# Bootstrap.
BOOTSTRAP_ITERATIONS = 10000
RANDOM_SEED = 42


# ============================================================
# FILE HELPERS
# ============================================================

def find_depth_files(depth_dir):
    files = sorted(depth_dir.glob("*.png"))

    if not files:
        files = sorted(
            p
            for p in depth_dir.iterdir()
            if p.is_file() and p.suffix.lower() == ".png"
        )

    if not files:
        raise RuntimeError(
            f"No depth PNG files found in {depth_dir}"
        )

    return files


# ============================================================
# PLANE FITTING
# ============================================================

def fit_plane_ransac(
    points,
    iterations=1000,
    threshold=0.02,
):
    """
    Fit a horizontal plane using RANSAC.

    The normal is constrained to be approximately vertical.
    """

    if len(points) < 3:
        return None, None

    rng = np.random.default_rng(RANDOM_SEED)

    best_count = 0
    best_plane = None
    best_inliers = None

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

        # Horizontal plane => normal approximately along Y.
        if abs(normal[1]) < 0.98:
            continue

        d = -np.dot(normal, p1)

        distances = np.abs(
            points @ normal + d
        )

        inliers = distances < threshold

        count = int(inliers.sum())

        if count > best_count:
            best_count = count
            best_plane = (normal.copy(), float(d))
            best_inliers = inliers.copy()

    return best_plane, best_inliers


def refine_plane(points):
    """
    Refine plane using least squares on the RANSAC inliers.
    """

    centroid = points.mean(axis=0)

    centered = points - centroid

    _, _, vh = np.linalg.svd(
        centered,
        full_matrices=False,
    )

    normal = vh[-1]

    normal /= np.linalg.norm(normal)

    if normal[1] < 0:
        normal = -normal

    d = -np.dot(normal, centroid)

    return normal, float(d)


# ============================================================
# HEIGHT CALCULATION
# ============================================================

def calculate_height_along_floor_normal(
    floor_normal,
    floor_d,
    ceiling_points,
):
    """
    Measure ceiling height along the floor normal.

    The floor normal defines the common vertical coordinate.

    Floor plane:
        n_f . p + d_f = 0

    For each ceiling point p:

        h = n_f . p - s_floor

    where

        s_floor = -d_f

    because n_f is normalized.

    We use the median over ceiling inliers.
    """

    n = floor_normal / np.linalg.norm(
        floor_normal
    )

    floor_coordinate = -floor_d

    ceiling_coordinates = (
        ceiling_points @ n
    )

    heights = (
        ceiling_coordinates
        - floor_coordinate
    )

    return (
        float(np.median(heights)),
        heights,
    )


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("TEMPORAL CEILING HEIGHT ESTIMATION")
print("=" * 70)

calibration = load_camera_calibration(CAPTURE)
odom = load_odometry(ODOMETRY)
depth_files = find_depth_files(DEPTH_DIR)

n_frames = min(
    len(depth_files),
    len(odom),
)

print(f"Depth frames: {len(depth_files)}")
print(f"Odometry rows: {len(odom)}")
print(f"Usable frames: {n_frames}")

print()
print(f"Window size: {WINDOW_SIZE}")
print(f"Window step: {WINDOW_STEP}")
print(f"Frame stride: {FRAME_STRIDE}")
print(f"Pixel stride: {PIXEL_STRIDE}")


# ============================================================
# WINDOW PROCESSING
# ============================================================

window_results = []

window_id = 0

for window_start in range(
    START_FRAME,
    min(END_FRAME, n_frames),
    WINDOW_STEP,
):

    window_end = min(
        window_start + WINDOW_SIZE,
        n_frames,
    )

    if window_end - window_start < WINDOW_SIZE:
        continue

    window_id += 1

    floor_points_all = []
    ceiling_points_all = []

    for frame in range(
        window_start,
        window_end,
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
            ],
            dtype=np.float64,
        )

        points_world = camera_to_world(
            points_camera,
            R,
            t,
        )

        floor_mask = (
            (points_world[:, 1] > FLOOR_MIN_Y)
            & (points_world[:, 1] < FLOOR_MAX_Y)
        )

        ceiling_mask = (
            (points_world[:, 1] > CEILING_MIN_Y)
            & (points_world[:, 1] < CEILING_MAX_Y)
        )

        if floor_mask.any():
            floor_points_all.append(
                points_world[floor_mask]
            )

        if ceiling_mask.any():
            ceiling_points_all.append(
                points_world[ceiling_mask]
            )

    if not floor_points_all or not ceiling_points_all:
        print(
            f"Window {window_id:02d} "
            f"{window_start}-{window_end}: "
            f"INSUFFICIENT DATA"
        )
        continue

    floor_points = np.vstack(
        floor_points_all
    )

    ceiling_points = np.vstack(
        ceiling_points_all
    )

    print(
        f"Window {window_id:02d} "
        f"{window_start}-{window_end}: "
        f"floor={len(floor_points):,}, "
        f"ceiling={len(ceiling_points):,}"
    )

    if (
        len(floor_points) < MIN_CANDIDATE_POINTS
        or len(ceiling_points) < MIN_CANDIDATE_POINTS
    ):
        print("  -> rejected: insufficient candidates")
        continue

    # --------------------------------------------------------
    # FLOOR PLANE
    # --------------------------------------------------------

    floor_plane, floor_inliers = fit_plane_ransac(
        floor_points,
        RANSAC_ITERATIONS,
        RANSAC_THRESHOLD,
    )

    if floor_plane is None:
        print("  -> rejected: floor RANSAC failed")
        continue

    floor_normal, floor_d = floor_plane

    if (
        floor_inliers is None
        or floor_inliers.sum() < MIN_INLIERS
    ):
        print("  -> rejected: insufficient floor inliers")
        continue

    floor_normal, floor_d = refine_plane(
        floor_points[floor_inliers]
    )

    # --------------------------------------------------------
    # CEILING PLANE
    # --------------------------------------------------------

    ceiling_plane, ceiling_inliers = fit_plane_ransac(
        ceiling_points,
        RANSAC_ITERATIONS,
        RANSAC_THRESHOLD,
    )

    if ceiling_plane is None:
        print("  -> rejected: ceiling RANSAC failed")
        continue

    ceiling_normal, ceiling_d = ceiling_plane

    if (
        ceiling_inliers is None
        or ceiling_inliers.sum() < MIN_INLIERS
    ):
        print("  -> rejected: insufficient ceiling inliers")
        continue

    ceiling_normal, ceiling_d = refine_plane(
        ceiling_points[ceiling_inliers]
    )

    # --------------------------------------------------------
    # NORMAL CONSISTENCY
    # --------------------------------------------------------

    normal_angle = np.degrees(
        np.arccos(
            np.clip(
                np.dot(
                    floor_normal,
                    ceiling_normal,
                ),
                -1.0,
                1.0,
            )
        )
    )

    # Since normals may differ slightly, compare the
    # acute angle.
    normal_angle = min(
        normal_angle,
        180.0 - normal_angle,
    )

    if normal_angle > 3.0:
        print(
            f"  -> rejected: "
            f"floor/ceiling angle "
            f"{normal_angle:.3f}°"
        )
        continue

    # --------------------------------------------------------
    # HEIGHT
    # --------------------------------------------------------

    ceiling_inlier_points = (
        ceiling_points[ceiling_inliers]
    )

    height, height_samples = (
        calculate_height_along_floor_normal(
            floor_normal,
            floor_d,
            ceiling_inlier_points,
        )
    )

    floor_residuals = np.abs(
        floor_points[floor_inliers]
        @ floor_normal
        + floor_d
    )

    ceiling_residuals = np.abs(
        ceiling_inlier_points
        @ ceiling_normal
        + ceiling_d
    )

    result = {
        "window_id": window_id,
        "start_frame": window_start,
        "end_frame": window_end,

        "floor_candidates": len(floor_points),
        "floor_inliers": int(
            floor_inliers.sum()
        ),

        "ceiling_candidates": len(ceiling_points),
        "ceiling_inliers": int(
            ceiling_inliers.sum()
        ),

        "floor_inlier_ratio":
            float(
                floor_inliers.mean()
            ),

        "ceiling_inlier_ratio":
            float(
                ceiling_inliers.mean()
            ),

        "floor_residual_median":
            float(
                np.median(floor_residuals)
            ),

        "floor_residual_p95":
            float(
                np.percentile(
                    floor_residuals,
                    95,
                )
            ),

        "ceiling_residual_median":
            float(
                np.median(ceiling_residuals)
            ),

        "ceiling_residual_p95":
            float(
                np.percentile(
                    ceiling_residuals,
                    95,
                )
            ),

        "normal_angle_deg":
            float(normal_angle),

        "height_m":
            height,
    }

    window_results.append(result)

    print(
        f"  floor inliers: "
        f"{floor_inliers.sum():,}"
    )

    print(
        f"  ceiling inliers: "
        f"{ceiling_inliers.sum():,}"
    )

    print(
        f"  normal angle: "
        f"{normal_angle:.4f}°"
    )

    print(
        f"  height: "
        f"{height:.5f} m"
    )


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 70)
print("WINDOW RESULTS")
print("=" * 70)

if not window_results:
    raise RuntimeError(
        "No valid temporal windows were found."
    )

df = pd.DataFrame(window_results)

print(
    df[
        [
            "window_id",
            "start_frame",
            "end_frame",
            "floor_inliers",
            "ceiling_inliers",
            "floor_inlier_ratio",
            "ceiling_inlier_ratio",
            "normal_angle_deg",
            "floor_residual_median",
            "ceiling_residual_median",
            "height_m",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.5f}",
    )
)


# ============================================================
# ROBUST HEIGHT ESTIMATE
# ============================================================

heights = df["height_m"].to_numpy()

median_height = float(
    np.median(heights)
)

mean_height = float(
    np.mean(heights)
)

mad = float(
    np.median(
        np.abs(
            heights - median_height
        )
    )
)

robust_sigma = 1.4826 * mad


print()
print("=" * 70)
print("HEIGHT SUMMARY")
print("=" * 70)

print(
    f"Valid windows: {len(heights)}"
)

print(
    f"Mean height:   {mean_height:.6f} m"
)

print(
    f"Median height: {median_height:.6f} m"
)

print(
    f"MAD:           {mad:.6f} m"
)

print(
    f"Robust sigma:  {robust_sigma:.6f} m"
)

print(
    f"Min height:    {heights.min():.6f} m"
)

print(
    f"Max height:    {heights.max():.6f} m"
)


# ============================================================
# BOOTSTRAP
# ============================================================

print()
print("=" * 70)
print("BOOTSTRAP UNCERTAINTY")
print("=" * 70)

rng = np.random.default_rng(
    RANDOM_SEED
)

bootstrap_medians = np.empty(
    BOOTSTRAP_ITERATIONS,
    dtype=np.float64,
)

for i in range(
    BOOTSTRAP_ITERATIONS
):

    sample = rng.choice(
        heights,
        size=len(heights),
        replace=True,
    )

    bootstrap_medians[i] = np.median(
        sample
    )


ci_low, ci_high = np.percentile(
    bootstrap_medians,
    [2.5, 97.5],
)

ci_half_width = (
    ci_high - ci_low
) / 2.0


print(
    f"Bootstrap iterations: "
    f"{BOOTSTRAP_ITERATIONS:,}"
)

print(
    f"95% CI lower: "
    f"{ci_low:.6f} m"
)

print(
    f"95% CI upper: "
    f"{ci_high:.6f} m"
)

print(
    f"95% CI width: "
    f"{(ci_high - ci_low):.6f} m"
)

print(
    f"95% CI half-width: "
    f"{ci_half_width:.6f} m"
)

print(
    f"95% CI half-width: "
    f"{ci_half_width * 100:.2f} cm"
)


# ============================================================
# ASSIGNMENT GATE
# ============================================================

GATE_CM = 1.5

print()
print("=" * 70)
print("CEILING HEIGHT VALIDATION GATE")
print("=" * 70)

if ci_half_width * 100 <= GATE_CM:

    print(
        f"PASS: 95% CI half-width "
        f"{ci_half_width * 100:.2f} cm "
        f"<= {GATE_CM:.2f} cm"
    )

    gate_pass = True

else:

    print(
        f"FAIL: 95% CI half-width "
        f"{ci_half_width * 100:.2f} cm "
        f"> {GATE_CM:.2f} cm"
    )

    gate_pass = False


# ============================================================
# SAVE MACHINE-READABLE RESULT
# ============================================================

output_dir = Path(
    "outputs/single_scan_with_ceiling"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True,
)

df.to_csv(
    output_dir / "ceiling_height_windows.csv",
    index=False,
)

summary = {
    "height_m": median_height,
    "height_cm": median_height * 100.0,

    "bootstrap_ci_95_lower_m":
        float(ci_low),

    "bootstrap_ci_95_upper_m":
        float(ci_high),

    "bootstrap_ci_95_half_width_m":
        float(ci_half_width),

    "bootstrap_ci_95_half_width_cm":
        float(ci_half_width * 100.0),

    "valid_windows":
        int(len(heights)),

    "window_size":
        WINDOW_SIZE,

    "window_step":
        WINDOW_STEP,

    "floor_ceiling_normal_angle_median_deg":
        float(df["normal_angle_deg"].median()),

    "gate_threshold_cm":
        GATE_CM,

    "gate_pass":
        gate_pass,
}

import json

with open(
    output_dir / "ceiling_height_result.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        summary,
        f,
        indent=2,
    )


print()
print(
    "Saved:"
)

print(
    output_dir
    / "ceiling_height_windows.csv"
)

print(
    output_dir
    / "ceiling_height_result.json"
)

print()
print("=" * 70)
print("DONE")
print("=" * 70)
