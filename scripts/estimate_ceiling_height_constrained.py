from pathlib import Path
import json

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
CONFIDENCE_DIR = CAPTURE / "confidence"
ODOMETRY = CAPTURE / "odometry.csv"

START_FRAME = 5000
END_FRAME = 7200

WINDOW_SIZE = 150
WINDOW_STEP = 75

FRAME_STRIDE = 5
PIXEL_STRIDE = 2

# Architectural candidate bands.
FLOOR_MIN_Y = -1.75
FLOOR_MAX_Y = -1.25

CEILING_MIN_Y = 1.35
CEILING_MAX_Y = 1.80

# Confidence values in this dataset are 0, 1, 2.
# We retain the highest-confidence returns.
MIN_CONFIDENCE = 2

# Plane residual gate.
RESIDUAL_THRESHOLD = 0.02

# Minimum evidence per temporal window.
MIN_FLOOR_POINTS = 2000
MIN_CEILING_POINTS = 2000

# Minimum spread along the surface.
# Prevents tiny accidental patches from being treated as
# architectural surfaces.
MIN_FLOOR_SPREAD = 0.5
MIN_CEILING_SPREAD = 0.5

BOOTSTRAP_ITERATIONS = 10000
RANDOM_SEED = 42


# ============================================================
# REFERENCE PLANES
# ============================================================

# Previously obtained independent local fits.
#
# Floor:
# n = [-0.006948, 0.999966, 0.004543]
# d =  1.477840
#
# Ceiling:
# n = [ 0.001841, 0.999997,-0.001799]
# d = -1.584690

FLOOR_NORMAL = np.array(
    [-0.006948, 0.999966, 0.004543],
    dtype=np.float64,
)

FLOOR_D = 1.477840

CEILING_NORMAL = np.array(
    [0.001841, 0.999997, -0.001799],
    dtype=np.float64,
)

CEILING_D = -1.584690


# ============================================================
# HELPERS
# ============================================================

def find_png_files(directory):
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
    return v / np.linalg.norm(v)


def plane_distance(points, normal, d):
    normal = normalize(normal)
    return points @ normal + d


def robust_location(values):
    """
    Robust surface location.

    Median is used rather than mean because depth returns can
    contain outliers and mixed surfaces.
    """
    return float(np.median(values))


def robust_mad(values):
    med = np.median(values)
    return float(
        np.median(np.abs(values - med))
    )

def backproject_with_confidence(
    depth,
    confidence,
    calibration,
    pixel_stride,
    min_confidence,
):
    depth_sampled = depth[
        ::pixel_stride,
        ::pixel_stride
    ]

    confidence_sampled = confidence[
        ::pixel_stride,
        ::pixel_stride
    ]

    if depth_sampled.shape != confidence_sampled.shape:
        raise ValueError(
            "Depth and confidence shapes do not match: "
            f"{depth_sampled.shape} vs {confidence_sampled.shape}"
        )

    # Convert depth to metres.
    z = depth_sampled.astype(np.float64) / calibration.depth_scale

    valid = (
        np.isfinite(z)
        & (z > 0)
        & (confidence_sampled >= min_confidence)
    )

    if not valid.any():
        return np.empty((0, 3), dtype=np.float64)

    # Depth intrinsics correspond to the 256x192 depth grid.
    fx = calibration.fx_depth
    fy = calibration.fy_depth
    cx = calibration.cx_depth
    cy = calibration.cy_depth

    v, u = np.indices(depth_sampled.shape)

    u = u[valid].astype(np.float64)
    v = v[valid].astype(np.float64)
    z = z[valid]

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    return np.column_stack((x, y, z))

# ============================================================
# COMMON VERTICAL AXIS
# ============================================================

floor_normal = normalize(FLOOR_NORMAL)
ceiling_normal = normalize(CEILING_NORMAL)

vertical = normalize(
    floor_normal + ceiling_normal
)

if vertical[1] < 0:
    vertical = -vertical


print("=" * 70)
print("CONSTRAINED CEILING HEIGHT ESTIMATION")
print("=" * 70)

print()
print("Reference floor normal:")
print(floor_normal)

print()
print("Reference ceiling normal:")
print(ceiling_normal)

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

print()
print(
    f"Reference floor/ceiling angle: "
    f"{normal_angle:.6f}°"
)

print()
print("Common vertical:")
print(vertical)


# ============================================================
# LOAD DATA
# ============================================================

calibration = load_camera_calibration(CAPTURE)
odom = load_odometry(ODOMETRY)

depth_files = find_png_files(DEPTH_DIR)
confidence_files = find_png_files(CONFIDENCE_DIR)

n_frames = min(
    len(depth_files),
    len(confidence_files),
    len(odom),
)

print()
print(f"Depth frames:      {len(depth_files)}")
print(f"Confidence frames: {len(confidence_files)}")
print(f"Odometry rows:     {len(odom)}")
print(f"Usable frames:     {n_frames}")

print()
print(f"Frame range:       {START_FRAME} -> {END_FRAME}")
print(f"Window size:       {WINDOW_SIZE}")
print(f"Window step:       {WINDOW_STEP}")
print(f"Frame stride:      {FRAME_STRIDE}")
print(f"Pixel stride:      {PIXEL_STRIDE}")
print(f"Min confidence:    {MIN_CONFIDENCE}")


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

    floor_points_total = 0
    ceiling_points_total = 0

    # --------------------------------------------------------
    # COLLECT POINTS
    # --------------------------------------------------------

    for frame in range(
        start,
        end,
        FRAME_STRIDE,
    ):

        depth = np.asarray(
            Image.open(depth_files[frame]),
            dtype=np.uint16,
        )

        confidence = np.asarray(
            Image.open(confidence_files[frame]),
            dtype=np.uint8,
        )
        points_camera = backproject_with_confidence(
            depth,
            confidence,
            calibration,
            PIXEL_STRIDE,
            MIN_CONFIDENCE,
        )

        if len(points_camera) == 0:
            continue
        # points_camera = depth_to_camera_points(
        #     depth,
        #     calibration,
        #     pixel_stride=PIXEL_STRIDE,
        # )

        # # Recreate the confidence sampling used by
        # # depth_to_camera_points.
        # conf = confidence[
        #     ::PIXEL_STRIDE,
        #     ::PIXEL_STRIDE,
        # ].reshape(-1)

        # # Keep only valid depth points.
        # depth_sampled = depth[
        #     ::PIXEL_STRIDE,
        #     ::PIXEL_STRIDE,
        # ]

        # valid = (
        #     np.isfinite(
        #         depth_sampled.astype(np.float64)
        #     )
        #     & (
        #         depth_sampled > 0
        #     )
        #     & (
        #         conf >= MIN_CONFIDENCE
        #     )
        # )

        # valid = valid.reshape(-1)

        # # depth_to_camera_points already removes invalid
        # # points, so we need to reconstruct the corresponding
        # # confidence mask from the same ordering.
        # points_camera = points_camera

        # # The point count after depth filtering must be aligned
        # # with the valid depth pixels.
        # depth_z = (
        #     depth_sampled.astype(np.float64)
        #     / calibration.depth_scale
        # )

        # finite_depth = (
        #     np.isfinite(depth_z)
        #     & (depth_z > 0)
        # )

        # combined_valid = (
        #     finite_depth
        #     & (
        #         confidence[
        #             ::PIXEL_STRIDE,
        #             ::PIXEL_STRIDE
        #         ] >= MIN_CONFIDENCE
        #     )
        # )

        # # Backprojection in this function preserves row-major
        # # ordering for valid points.
        # combined_valid_flat = combined_valid.reshape(-1)

        # points_camera = points_camera[
        #     confidence[
        #         ::PIXEL_STRIDE,
        #         ::PIXEL_STRIDE
        #     ].reshape(-1)[
        #         finite_depth.reshape(-1)
        #     ] >= MIN_CONFIDENCE
        # ]

        # if len(points_camera) == 0:
        #     continue

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
        # CANDIDATE SURFACES
        # ----------------------------------------------------

        floor_mask = (
            (points_world[:, 1] > FLOOR_MIN_Y)
            & (points_world[:, 1] < FLOOR_MAX_Y)
        )

        ceiling_mask = (
            (points_world[:, 1] > CEILING_MIN_Y)
            & (points_world[:, 1] < CEILING_MAX_Y)
        )

        floor_points = points_world[floor_mask]
        ceiling_points = points_world[ceiling_mask]

        floor_points_total += len(floor_points)
        ceiling_points_total += len(ceiling_points)

        if len(floor_points):

            # Project onto the fixed vertical axis.
            s_floor = (
                floor_points @ vertical
            )

            # Remove gross outliers around the expected
            # floor location.
            expected_floor = (
                -FLOOR_D
            )

            good = np.abs(
                s_floor - expected_floor
            ) < 0.15

            s_floor = s_floor[good]

            if len(s_floor):
                floor_locations.extend(
                    s_floor.tolist()
                )

        if len(ceiling_points):

            s_ceiling = (
                ceiling_points @ vertical
            )

            expected_ceiling = (
                -CEILING_D
            )

            good = np.abs(
                s_ceiling - expected_ceiling
            ) < 0.15

            s_ceiling = s_ceiling[good]

            if len(s_ceiling):
                ceiling_locations.extend(
                    s_ceiling.tolist()
                )

    # --------------------------------------------------------
    # CHECK EVIDENCE
    # --------------------------------------------------------

    floor_locations = np.asarray(
        floor_locations,
        dtype=np.float64,
    )

    ceiling_locations = np.asarray(
        ceiling_locations,
        dtype=np.float64,
    )

    if (
        len(floor_locations)
        < MIN_FLOOR_POINTS
        or len(ceiling_locations)
        < MIN_CEILING_POINTS
    ):

        print(
            f"Window {window_id:02d} "
            f"{start}-{end}: REJECTED "
            f"(floor={len(floor_locations):,}, "
            f"ceiling={len(ceiling_locations):,})"
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

    # Spatial coverage.
    floor_spread = np.ptp(
        floor_locations
    )

    ceiling_spread = np.ptp(
        ceiling_locations
    )

    # Robust sanity check.
    if (
        floor_spread < MIN_FLOOR_SPREAD
        or ceiling_spread < MIN_CEILING_SPREAD
    ):

        print(
            f"Window {window_id:02d} "
            f"{start}-{end}: REJECTED "
            f"(insufficient vertical support)"
        )

        continue

    results.append(
        {
            "window_id": window_id,
            "start_frame": start,
            "end_frame": end,

            "floor_points": int(
                len(floor_locations)
            ),

            "ceiling_points": int(
                len(ceiling_locations)
            ),

            "floor_location_m":
                floor_location,

            "ceiling_location_m":
                ceiling_location,

            "height_m":
                height,

            "floor_mad_m":
                floor_mad,

            "ceiling_mad_m":
                ceiling_mad,

            "floor_spread_m":
                floor_spread,

            "ceiling_spread_m":
                ceiling_spread,
        }
    )

    print(
        f"Window {window_id:02d} "
        f"{start}-{end}: "
        f"floor={len(floor_locations):,}, "
        f"ceiling={len(ceiling_locations):,}, "
        f"height={height:.5f} m"
    )


# ============================================================
# WINDOW RESULTS
# ============================================================

if not results:
    raise RuntimeError(
        "No valid temporal windows found."
    )

df = pd.DataFrame(results)

heights = df["height_m"].to_numpy()


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
            heights - median_height
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
    f"Mean height:   {mean_height:.6f} m"
)

print(
    f"Median height: {median_height:.6f} m"
)

print(
    f"MAD:           {mad_height:.6f} m"
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
# ROBUST OUTLIER FILTER
# ============================================================

if robust_sigma > 0:

    robust_mask = (
        np.abs(
            heights - median_height
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

print()
print("=" * 70)
print("ROBUST FILTER")
print("=" * 70)

print(
    f"Retained: "
    f"{len(filtered_heights)}/{len(heights)}"
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

GATE_CM = 1.5

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

df.to_csv(
    output_dir
    / "ceiling_height_constrained_windows.csv",
    index=False,
)

summary = {
    "height_m":
        float(np.median(filtered_heights)),

    "height_cm":
        float(np.median(filtered_heights) * 100),

    "bootstrap_ci_95_lower_m":
        float(ci_low),

    "bootstrap_ci_95_upper_m":
        float(ci_high),

    "bootstrap_ci_95_half_width_m":
        float(ci_half_width),

    "bootstrap_ci_95_half_width_cm":
        float(ci_half_width * 100),

    "valid_windows":
        int(len(filtered_heights)),

    "raw_valid_windows":
        int(len(heights)),

    "rejected_outlier_windows":
        int(np.sum(~robust_mask)),

    "reference_normal_angle_deg":
        float(normal_angle),

    "gate_threshold_cm":
        GATE_CM,

    "gate_pass":
        gate_pass,

    "method":
        "fixed common vertical axis + confidence-filtered "
        "LiDAR + temporal-window bootstrap",
}

with open(
    output_dir
    / "ceiling_height_constrained_result.json",
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
print(
    output_dir
    / "ceiling_height_constrained_windows.csv"
)

print(
    output_dir
    / "ceiling_height_constrained_result.json"
)

print()
print("=" * 70)
print("DONE")
print("=" * 70)
