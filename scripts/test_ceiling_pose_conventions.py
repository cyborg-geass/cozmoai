from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from cozmo_ai.io.calibration import load_camera_calibration
from cozmo_ai.io.odometry import load_odometry


# ============================================================
# CONFIG
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

FRAME_STRIDE = 10
PIXEL_STRIDE = 4

MIN_CONFIDENCE = 2

# Previously fitted floor plane.
FLOOR_NORMAL = np.array(
    [-0.006948, 0.999966, 0.004543],
    dtype=np.float64,
)

FLOOR_D = 1.477840


# ============================================================
# HELPERS
# ============================================================

def normalize(v):
    return v / np.linalg.norm(v)


def find_png_files(directory):
    files = sorted(directory.glob("*.png"))

    if not files:
        files = sorted(
            p
            for p in directory.iterdir()
            if p.is_file()
            and p.suffix.lower() == ".png"
        )

    if not files:
        raise RuntimeError(
            f"No PNG files found in {directory}"
        )

    return files


def backproject(
    depth,
    confidence,
    calibration,
):
    depth_s = depth[
        ::PIXEL_STRIDE,
        ::PIXEL_STRIDE,
    ]

    confidence_s = confidence[
        ::PIXEL_STRIDE,
        ::PIXEL_STRIDE,
    ]

    z = (
        depth_s.astype(np.float64)
        / calibration.depth_scale
    )

    valid = (
        np.isfinite(z)
        & (z > 0)
        & (
            confidence_s
            >= MIN_CONFIDENCE
        )
    )

    if not np.any(valid):
        return np.empty(
            (0, 3),
            dtype=np.float64,
        )

    v, u = np.indices(
        depth_s.shape
    )

    fx = calibration.fx_depth
    fy = calibration.fy_depth
    cx = calibration.cx_depth
    cy = calibration.cy_depth

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


def evaluate(
    points_camera,
    R,
    t,
):
    """
    Evaluate four possible pose conventions.

    A = R @ P + t
    B = R.T @ (P - t)
    C = R.T @ P + t
    D = R @ (P - t)
    """

    A = (
        points_camera @ R.T
        + t
    )

    B = (
        (points_camera - t)
        @ R
    )

    C = (
        points_camera @ R
        + t
    )

    D = (
        points_camera - t
    ) @ R.T

    return {
        "A_RP_plus_t": A,
        "B_RT_P_minus_t": B,
        "C_RT_P_plus_t": C,
        "D_R_P_minus_t": D,
    }


# ============================================================
# LOAD
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

n = min(
    len(depth_files),
    len(confidence_files),
    len(odom),
)

floor_normal = normalize(
    FLOOR_NORMAL
)


# ============================================================
# RUN
# ============================================================

print("=" * 70)
print("CEILING CAPTURE POSE CONVENTION TEST")
print("=" * 70)

print()
print(
    f"Frames: {START_FRAME} -> {END_FRAME}"
)

print(
    f"Frame stride: {FRAME_STRIDE}"
)

print(
    f"Pixel stride: {PIXEL_STRIDE}"
)

print()
print("Reference floor:")
print(
    f"normal = {floor_normal}"
)
print(
    f"d      = {FLOOR_D}"
)


all_distances = {
    "A_RP_plus_t": [],
    "B_RT_P_minus_t": [],
    "C_RT_P_plus_t": [],
    "D_R_P_minus_t": [],
}


for frame in range(
    START_FRAME,
    min(END_FRAME, n),
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

    points_camera = backproject(
        depth,
        confidence,
        calibration,
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

    transformed = evaluate(
        points_camera,
        R,
        t,
    )

    for name, points in transformed.items():

        distances = np.abs(
            points @ floor_normal
            + FLOOR_D
        )

        all_distances[name].extend(
            distances.tolist()
        )


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 70)
print("RESULTS")
print("=" * 70)

summary = []

for name, values in all_distances.items():

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if len(values) == 0:
        continue

    minimum = np.min(values)
    p01 = np.percentile(values, 1)
    p05 = np.percentile(values, 5)
    median = np.median(values)

    within_1cm = np.sum(
        values <= 0.01
    )

    within_3cm = np.sum(
        values <= 0.03
    )

    within_5cm = np.sum(
        values <= 0.05
    )

    print()
    print(name)
    print("-" * 50)

    print(
        f"Minimum distance: "
        f"{minimum * 100:.3f} cm"
    )

    print(
        f"P01 distance:     "
        f"{p01 * 100:.3f} cm"
    )

    print(
        f"P05 distance:     "
        f"{p05 * 100:.3f} cm"
    )

    print(
        f"Median distance:   "
        f"{median * 100:.3f} cm"
    )

    print(
        f"<= 1 cm: "
        f"{within_1cm:,}"
    )

    print(
        f"<= 3 cm: "
        f"{within_3cm:,}"
    )

    print(
        f"<= 5 cm: "
        f"{within_5cm:,}"
    )

    summary.append(
        (
            name,
            minimum,
            p01,
            p05,
            median,
        )
    )


# ============================================================
# RANKING
# ============================================================

print()
print("=" * 70)
print("RANKING")
print("=" * 70)

summary.sort(
    key=lambda x: x[2]
)

for rank, item in enumerate(
    summary,
    start=1,
):

    name, minimum, p01, p05, median = item

    print(
        f"{rank}. {name:22s} "
        f"min={minimum * 100:7.3f}cm "
        f"p01={p01 * 100:7.3f}cm "
        f"p05={p05 * 100:7.3f}cm "
        f"median={median * 100:7.3f}cm"
    )


print()
print("=" * 70)
print("DONE")
print("=" * 70)
