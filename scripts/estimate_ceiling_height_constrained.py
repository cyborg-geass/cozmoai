from pathlib import Path
import json

import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial.transform import Rotation

from cozmo_ai.io.calibration import load_camera_calibration
from cozmo_ai.io.odometry import load_odometry
from cozmo_ai.geometry.pose import camera_to_world


# ============================================================
# CONFIGURATION
# ============================================================

CAPTURE = Path(
    r"..\cozmo-dataset\raw_dataset"
    r"\single_scan_with_ceiling\c7d28f72c6"
)

DEPTH_DIR = CAPTURE / "depth"
CONFIDENCE_DIR = CAPTURE / "confidence"
ODOMETRY = CAPTURE / "odometry.csv"

START_FRAME = 5000
END_FRAME = 7200

WINDOW_SIZE = 150
WINDOW_STEP = 75

FRAME_STRIDE = 5
PIXEL_STRIDE = 2

# Confidence values in this dataset are 0, 1, 2.
# Retain only the highest-confidence returns.
MIN_CONFIDENCE = 2

# Distance from the independently fitted reference plane
# used to classify a point as belonging to that surface.
#
# IMPORTANT:
# This is a candidate-selection tolerance.
# It is NOT the final measurement uncertainty.
PLANE_TOLERANCE = 0.03  # metres

# Minimum number of candidate points in a window.
MIN_FLOOR_POINTS = 2000
MIN_CEILING_POINTS = 2000

# Require reasonable residual quality for the points
# assigned to each reference plane.
MAX_P95_RESIDUAL = 0.025  # 2.5 cm

# Require the candidate surface to have spatial support.
# We measure the 2D footprint on the plane, rather than
# the spread along the vertical axis.
MIN_SURFACE_SPREAD = 0.5  # metres

BOOTSTRAP_ITERATIONS = 10000
RANDOM_SEED = 42

# Assignment gate.
GATE_CM = 1.5


# ============================================================
# REFERENCE PLANES
# ============================================================

# Independently fitted local floor plane:
#
# -0.006948 x + 0.999966 y + 0.004543 z
# + 1.477840 = 0
#
FLOOR_NORMAL = np.array(
    [-0.006948, 0.999966, 0.004543],
    dtype=np.float64,
)

FLOOR_D = 1.477840


# Independently fitted local ceiling plane:
#
# 0.001841 x + 0.999997 y - 0.001799 z
# - 1.584690 = 0
#
CEILING_NORMAL = np.array(
    [0.001841, 0.999997, -0.001799],
    dtype=np.float64,
)

CEILING_D = -1.584690


# ============================================================
# HELPERS
# ============================================================

def find_png_files(directory):
    """
    Find PNG files without accidentally duplicating files on
    case-insensitive filesystems such as Windows.
    """
    files = sorted(directory.glob("*.png"))

    if not files:
        files = sorted(
            p
            for p in directory.iterdir()
            if p.is_file() and p.suffix.lower() == ".png"
        )

    if not files:
        raise RuntimeError(
            f"No PNG files found in {directory}"
        )

    return files


def normalize(v):
    """
    Return a unit-length copy of v.
    """
    v = np.asarray(v, dtype=np.float64)
    norm = np.linalg.norm(v)

    if norm == 0:
        raise ValueError("Cannot normalize zero vector.")

    return v / norm


def signed_plane_distance(points, normal, d):
    """
    Signed distance-like plane residual.

    For normalized normal n:
        distance = n dot p + d

    The supplied reference normals are normalized before use.
    """
    normal = normalize(normal)

    return points @ normal + d


def robust_location(values):
    """
    Robust estimate of a surface coordinate.

    Median is used because depth returns may contain outliers.
    """
    values = np.asarray(values, dtype=np.float64)

    if len(values) == 0:
        raise ValueError(
            "Cannot estimate location from zero values."
        )

    return float(np.median(values))


def robust_mad(values):
    """
    Median absolute deviation.
    """
    values = np.asarray(values, dtype=np.float64)

    if len(values) == 0:
        return float("nan")

    med = np.median(values)

    return float(
        np.median(np.abs(values - med))
    )


def make_plane_basis(normal):
    """
    Construct two orthonormal axes lying in a plane whose
    normal is `normal`.

    Returns:
        axis_u, axis_v
    """
    normal = normalize(normal)

    # Choose a reference vector that is not almost parallel
    # to the plane normal.
    if abs(normal[0]) < 0.9:
        reference = np.array(
            [1.0, 0.0, 0.0],
            dtype=np.float64,
        )
    else:
        reference = np.array(
            [0.0, 1.0, 0.0],
            dtype=np.float64,
        )

    axis_u = np.cross(normal, reference)
    axis_u = normalize(axis_u)

    axis_v = np.cross(normal, axis_u)
    axis_v = normalize(axis_v)

    return axis_u, axis_v


def surface_spread(points, normal):
    """
    Measure 2D spatial support of points on a plane.

    Returns:
        spread_u, spread_v
    """
    if len(points) == 0:
        return 0.0, 0.0

    axis_u, axis_v = make_plane_basis(normal)

    u = points @ axis_u
    v = points @ axis_v

    spread_u = float(np.ptp(u))
    spread_v = float(np.ptp(v))

    return spread_u, spread_v


def backproject_with_confidence(
    depth,
    confidence,
    calibration,
    pixel_stride,
    min_confidence,
):
    """
    Backproject a sampled depth image while applying the
    confidence mask on the exact same sampled pixels.

    Depth values are converted from millimetres to metres.

    The depth intrinsics are already represented at the
    native 256x192 depth resolution by CameraCalibration.
    """

    depth_sampled = depth[
        ::pixel_stride,
        ::pixel_stride,
    ]

    confidence_sampled = confidence[
        ::pixel_stride,
        ::pixel_stride,
    ]

    if depth_sampled.shape != confidence_sampled.shape:
        raise ValueError(
            "Depth and confidence shapes do not match: "
            f"{depth_sampled.shape} vs "
            f"{confidence_sampled.shape}"
        )

    z = (
        depth_sampled.astype(np.float64)
        / calibration.depth_scale
    )

    valid = (
        np.isfinite(z)
        & (z > 0)
        & (
            confidence_sampled
            >= min_confidence
        )
    )

    if not valid.any():
        return np.empty(
            (0, 3),
            dtype=np.float64,
        )

    fx = calibration.fx_depth
    fy = calibration.fy_depth
    cx = calibration.cx_depth
    cy = calibration.cy_depth

    v, u = np.indices(
        depth_sampled.shape
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
# REFERENCE GEOMETRY
# ============================================================

floor_normal = normalize(
    FLOOR_NORMAL
)

ceiling_normal = normalize(
    CEILING_NORMAL
)

# IMPORTANT:
# Define architectural vertical using the floor normal.
#
# The floor is the physical reference for "vertical".
# The difference between floor and ceiling normals is
# retained as a diagnostic rather than averaged away.
vertical = floor_normal.copy()

if vertical[1] < 0:
    vertical = -vertical


# Angle between the independently fitted planes.
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

normal_angle = min(
    normal_angle,
    180.0 - normal_angle,
)


# ============================================================
# PRINT CONFIGURATION
# ============================================================

print("=" * 70)
print("CONSTRAINED CEILING HEIGHT ESTIMATION")
print("=" * 70)

print()
print("Reference floor normal:")
print(floor_normal)

print()
print("Reference ceiling normal:")
print(ceiling_normal)

print()
print(
    "Reference floor/ceiling angle: "
    f"{normal_angle:.6f}°"
)

print()
print("Reference vertical:")
print(vertical)

print()
print("Plane tolerance:")
print(
    f"{PLANE_TOLERANCE * 100:.1f} cm"
)

print()
print("Maximum P95 plane residual:")
print(
    f"{MAX_P95_RESIDUAL * 100:.1f} cm"
)


# ============================================================
# LOAD DATA
# ============================================================

calibration = load_camera_calibration(
    CAPTURE
)

odom = load_odometry(
    ODOMETRY
)

depth_files = find_png_files(
    DEPTH_DIR
)

confidence_files = find_png_files(
    CONFIDENCE_DIR
)

n_frames = min(
    len(depth_files),
    len(confidence_files),
    len(odom),
)

print()
print(f"Depth frames:      {len(depth_files)}")
print(
    f"Confidence frames: "
    f"{len(confidence_files)}"
)
print(
    f"Odometry rows:     "
    f"{len(odom)}"
)
print(
    f"Usable frames:     "
    f"{n_frames}"
)

if START_FRAME >= n_frames:
    raise RuntimeError(
        f"START_FRAME={START_FRAME} "
        f"is outside available frame range "
        f"0..{n_frames - 1}"
    )


# ============================================================
# TEMPORAL WINDOWS
# ============================================================

results = []

window_id = 0

for start in range(
    START_FRAME,
    min(END_FRAME, n_frames),
    WINDOW_STEP,
):

    end = min(
        start + WINDOW_SIZE,
        n_frames,
    )

    if end - start < WINDOW_SIZE:
        continue

    window_id += 1

    floor_locations = []
    ceiling_locations = []

    floor_residuals = []
    ceiling_residuals = []

    floor_points_total = 0
    ceiling_points_total = 0

    floor_world_points = []
    ceiling_world_points = []

    # --------------------------------------------------------
    # COLLECT POINTS
    # --------------------------------------------------------

    for frame in range(
        start,
        end,
        FRAME_STRIDE,
    ):

        depth = np.asarray(
            Image.open(
                depth_files[frame]
            ),
            dtype=np.uint16,
        )

        confidence = np.asarray(
            Image.open(
                confidence_files[frame]
            ),
            dtype=np.uint8,
        )

        points_camera = (
            backproject_with_confidence(
                depth,
                confidence,
                calibration,
                PIXEL_STRIDE,
                MIN_CONFIDENCE,
            )
        )

        if len(points_camera) == 0:
            continue

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

        # ----------------------------------------------------
        # CLASSIFY AGAINST REFERENCE PLANES
        # ----------------------------------------------------

        floor_signed_distance = (
            signed_plane_distance(
                points_world,
                floor_normal,
                FLOOR_D,
            )
        )

        ceiling_signed_distance = (
            signed_plane_distance(
                points_world,
                ceiling_normal,
                CEILING_D,
            )
        )

        floor_mask = (
            np.abs(
                floor_signed_distance
            )
            <= PLANE_TOLERANCE
        )

        ceiling_mask = (
            np.abs(
                ceiling_signed_distance
            )
            <= PLANE_TOLERANCE
        )

        floor_points = (
            points_world[floor_mask]
        )

        ceiling_points = (
            points_world[ceiling_mask]
        )

        floor_point_residuals = np.abs(
            floor_signed_distance[
                floor_mask
            ]
        )

        ceiling_point_residuals = np.abs(
            ceiling_signed_distance[
                ceiling_mask
            ]
        )

        floor_points_total += (
            len(floor_points)
        )

        ceiling_points_total += (
            len(ceiling_points)
        )

        if len(floor_points):

            floor_level_samples = (
                floor_points @ vertical
            )

            floor_locations.extend(
                floor_level_samples.tolist()
            )

            floor_residuals.extend(
                floor_point_residuals.tolist()
            )

            floor_world_points.append(
                floor_points
            )

        if len(ceiling_points):

            ceiling_level_samples = (
                ceiling_points @ vertical
            )

            ceiling_locations.extend(
                ceiling_level_samples.tolist()
            )

            ceiling_residuals.extend(
                ceiling_point_residuals.tolist()
            )

            ceiling_world_points.append(
                ceiling_points
            )

    # --------------------------------------------------------
    # CONVERT TO ARRAYS
    # --------------------------------------------------------

    floor_locations = np.asarray(
        floor_locations,
        dtype=np.float64,
    )

    ceiling_locations = np.asarray(
        ceiling_locations,
        dtype=np.float64,
    )

    floor_residuals = np.asarray(
        floor_residuals,
        dtype=np.float64,
    )

    ceiling_residuals = np.asarray(
        ceiling_residuals,
        dtype=np.float64,
    )

    if floor_world_points:
        floor_world_points = np.vstack(
            floor_world_points
        )
    else:
        floor_world_points = np.empty(
            (0, 3),
            dtype=np.float64,
        )

    if ceiling_world_points:
        ceiling_world_points = np.vstack(
            ceiling_world_points
        )
    else:
        ceiling_world_points = np.empty(
            (0, 3),
            dtype=np.float64,
        )

    # --------------------------------------------------------
    # CHECK POINT EVIDENCE
    # --------------------------------------------------------

    if (
        len(floor_locations)
        < MIN_FLOOR_POINTS
        or len(ceiling_locations)
        < MIN_CEILING_POINTS
    ):

        print(
            f"Window {window_id:02d} "
            f"{start}-{end}: REJECTED "
            f"(floor="
            f"{len(floor_locations):,}, "
            f"ceiling="
            f"{len(ceiling_locations):,})"
        )

        continue

    # --------------------------------------------------------
    # RESIDUAL QUALITY
    # --------------------------------------------------------

    floor_p95 = float(
        np.percentile(
            floor_residuals,
            95,
        )
    )

    ceiling_p95 = float(
        np.percentile(
            ceiling_residuals,
            95,
        )
    )

    if (
        floor_p95
        > MAX_P95_RESIDUAL
        or ceiling_p95
        > MAX_P95_RESIDUAL
    ):

        print(
            f"Window {window_id:02d} "
            f"{start}-{end}: REJECTED "
            f"(P95 residual: "
            f"floor={floor_p95 * 100:.2f} cm, "
            f"ceiling={ceiling_p95 * 100:.2f} cm)"
        )

        continue

    # --------------------------------------------------------
    # ROBUST SURFACE LOCATIONS
    # --------------------------------------------------------

    floor_location = robust_location(
        floor_locations
    )

    ceiling_location = robust_location(
        ceiling_locations
    )

    height = (
        ceiling_location
        - floor_location
    )

    floor_mad = robust_mad(
        floor_locations
    )

    ceiling_mad = robust_mad(
        ceiling_locations
    )

    # --------------------------------------------------------
    # SPATIAL SUPPORT
    # --------------------------------------------------------

    (
        floor_spread_u,
        floor_spread_v,
    ) = surface_spread(
        floor_world_points,
        floor_normal,
    )

    (
        ceiling_spread_u,
        ceiling_spread_v,
    ) = surface_spread(
        ceiling_world_points,
        ceiling_normal,
    )

    if (
        floor_spread_u < MIN_SURFACE_SPREAD
        and floor_spread_v < MIN_SURFACE_SPREAD
    ):

        print(
            f"Window {window_id:02d} "
            f"{start}-{end}: REJECTED "
            f"(insufficient floor spatial support)"
        )

        continue

    if (
        ceiling_spread_u < MIN_SURFACE_SPREAD
        and ceiling_spread_v < MIN_SURFACE_SPREAD
    ):

        print(
            f"Window {window_id:02d} "
            f"{start}-{end}: REJECTED "
            f"(insufficient ceiling spatial support)"
        )

        continue

    # --------------------------------------------------------
    # SAVE WINDOW RESULT
    # --------------------------------------------------------

    results.append(
        {
            "window_id": int(window_id),

            "start_frame": int(start),
            "end_frame": int(end),

            "floor_points": int(
                len(floor_locations)
            ),

            "ceiling_points": int(
                len(ceiling_locations)
            ),

            "floor_location_m": float(
                floor_location
            ),

            "ceiling_location_m": float(
                ceiling_location
            ),

            "height_m": float(
                height
            ),

            "floor_mad_m": float(
                floor_mad
            ),

            "ceiling_mad_m": float(
                ceiling_mad
            ),

            "floor_p95_residual_m": float(
                floor_p95
            ),

            "ceiling_p95_residual_m": float(
                ceiling_p95
            ),

            "floor_spread_u_m": float(
                floor_spread_u
            ),

            "floor_spread_v_m": float(
                floor_spread_v
            ),

            "ceiling_spread_u_m": float(
                ceiling_spread_u
            ),

            "ceiling_spread_v_m": float(
                ceiling_spread_v
            ),
        }
    )

    print(
        f"Window {window_id:02d} "
        f"{start}-{end}: "
        f"floor={len(floor_locations):,}, "
        f"ceiling={len(ceiling_locations):,}, "
        f"height={height:.5f} m, "
        f"floor_p95={floor_p95 * 100:.2f} cm, "
        f"ceiling_p95={ceiling_p95 * 100:.2f} cm"
    )


# ============================================================
# WINDOW RESULTS
# ============================================================

if not results:
    raise RuntimeError(
        "No valid temporal windows found."
    )

df = pd.DataFrame(results)

heights = df[
    "height_m"
].to_numpy(
    dtype=np.float64
)


print()
print("=" * 70)
print("VALID WINDOW RESULTS")
print("=" * 70)

print(
    df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)


# ============================================================
# ROBUST SUMMARY
# ============================================================

median_height = float(
    np.median(heights)
)

mean_height = float(
    np.mean(heights)
)

mad_height = float(
    np.median(
        np.abs(
            heights
            - median_height
        )
    )
)

robust_sigma = (
    1.4826 * mad_height
)


print()
print("=" * 70)
print("HEIGHT SUMMARY")
print("=" * 70)

print(
    f"Valid windows: {len(heights)}"
)

print(
    f"Mean height:   "
    f"{mean_height:.6f} m"
)

print(
    f"Median height: "
    f"{median_height:.6f} m"
)

print(
    f"MAD:           "
    f"{mad_height:.6f} m"
)

print(
    f"Robust sigma:  "
    f"{robust_sigma:.6f} m"
)

print(
    f"Min height:    "
    f"{heights.min():.6f} m"
)

print(
    f"Max height:    "
    f"{heights.max():.6f} m"
)


# ============================================================
# ROBUST OUTLIER FILTER
# ============================================================

if robust_sigma > 0:

    robust_mask = (
        np.abs(
            heights
            - median_height
        )
        <= 3.0 * robust_sigma
    )

else:

    robust_mask = np.ones(
        len(heights),
        dtype=bool,
    )

filtered_heights = heights[
    robust_mask
]

if len(filtered_heights) == 0:
    raise RuntimeError(
        "Robust outlier filtering removed "
        "all temporal windows."
    )


print()
print("=" * 70)
print("ROBUST FILTER")
print("=" * 70)

print(
    f"Retained: "
    f"{len(filtered_heights)}"
    f"/{len(heights)}"
)

print(
    f"Rejected: "
    f"{np.sum(~robust_mask)}"
)

print(
    f"Filtered median: "
    f"{np.median(filtered_heights):.6f} m"
)


# ============================================================
# BOOTSTRAP
# ============================================================

rng = np.random.default_rng(
    RANDOM_SEED
)

bootstrap = np.empty(
    BOOTSTRAP_ITERATIONS,
    dtype=np.float64,
)

for i in range(
    BOOTSTRAP_ITERATIONS
):

    sample = rng.choice(
        filtered_heights,
        size=len(filtered_heights),
        replace=True,
    )

    bootstrap[i] = np.median(
        sample
    )


ci_low, ci_high = np.percentile(
    bootstrap,
    [2.5, 97.5],
)

ci_half_width = (
    ci_high - ci_low
) / 2.0


print()
print("=" * 70)
print("BOOTSTRAP UNCERTAINTY")
print("=" * 70)

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
    f"{ci_high - ci_low:.6f} m"
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
# GATE
# ============================================================

print()
print("=" * 70)
print("CEILING HEIGHT VALIDATION GATE")
print("=" * 70)

if (
    ci_half_width * 100
    <= GATE_CM
):

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
# SAVE
# ============================================================

output_dir = Path(
    "outputs/single_scan_with_ceiling"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True,
)

windows_path = (
    output_dir
    / "ceiling_height_constrained_windows.csv"
)

result_path = (
    output_dir
    / "ceiling_height_constrained_result.json"
)

df.to_csv(
    windows_path,
    index=False,
)


summary = {
    "height_m": float(
        np.median(filtered_heights)
    ),

    "height_cm": float(
        np.median(filtered_heights)
        * 100
    ),

    "bootstrap_ci_95_lower_m": float(
        ci_low
    ),

    "bootstrap_ci_95_upper_m": float(
        ci_high
    ),

    "bootstrap_ci_95_half_width_m":
        float(ci_half_width),

    "bootstrap_ci_95_half_width_cm":
        float(
            ci_half_width * 100
        ),

    "valid_windows": int(
        len(filtered_heights)
    ),

    "raw_valid_windows": int(
        len(heights)
    ),

    "rejected_outlier_windows": int(
        np.sum(~robust_mask)
    ),

    "reference_normal_angle_deg":
        float(normal_angle),

    "floor_normal": [
        float(x)
        for x in floor_normal
    ],

    "ceiling_normal": [
        float(x)
        for x in ceiling_normal
    ],

    "vertical_axis": [
        float(x)
        for x in vertical
    ],

    "floor_plane": [
        float(x)
        for x in FLOOR_NORMAL
    ] + [float(FLOOR_D)],

    "ceiling_plane": [
        float(x)
        for x in CEILING_NORMAL
    ] + [float(CEILING_D)],

    "plane_tolerance_m":
        PLANE_TOLERANCE,

    "max_p95_residual_m":
        MAX_P95_RESIDUAL,

    "gate_threshold_cm":
        GATE_CM,

    "gate_pass":
        gate_pass,

    "method":
        "floor-normal vertical axis + "
        "reference-plane distance classification + "
        "confidence-filtered LiDAR + "
        "temporal-window robust median + "
        "bootstrap",
}


with open(
    result_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
    )


print()
print("Saved:")
print(windows_path)
print(result_path)

print()
print("=" * 70)
print("DONE")
print("=" * 70)
