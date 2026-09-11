from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from cozmo_ai.io.calibration import load_camera_calibration
from cozmo_ai.io.odometry import load_odometry
from cozmo_ai.geometry.pose import camera_to_world


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

FRAME_STRIDE = 5
PIXEL_STRIDE = 2

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
            p for p in directory.iterdir()
            if p.is_file()
            and p.suffix.lower() == ".png"
        )

    return files


def backproject(depth, confidence, calibration, confidence_threshold):
    depth_s = depth[
        ::PIXEL_STRIDE,
        ::PIXEL_STRIDE
    ]

    conf_s = confidence[
        ::PIXEL_STRIDE,
        ::PIXEL_STRIDE
    ]

    z = (
        depth_s.astype(np.float64)
        / calibration.depth_scale
    )

    valid = (
        np.isfinite(z)
        & (z > 0)
        & (conf_s >= confidence_threshold)
    )

    if not np.any(valid):
        return np.empty((0, 3)), np.empty((0,))


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

    confidence_values = conf_s[valid]

    return (
        np.column_stack((x, y, z)),
        confidence_values,
    )


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


print("=" * 70)
print("FLOOR PLANE MATCH DIAGNOSTIC")
print("=" * 70)

print()
print("Floor plane:")
print(
    f"{floor_normal[0]:.6f}x + "
    f"{floor_normal[1]:.6f}y + "
    f"{floor_normal[2]:.6f}z + "
    f"{FLOOR_D:.6f} = 0"
)

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


# ============================================================
# TEST BOTH CONFIDENCE POLICIES
# ============================================================

for confidence_threshold in [0, 1, 2]:

    print()
    print("=" * 70)
    print(
        f"CONFIDENCE THRESHOLD = "
        f"{confidence_threshold}"
    )
    print("=" * 70)

    all_distances = []
    near_1cm = 0
    near_2cm = 0
    near_3cm = 0
    near_5cm = 0

    total_points = 0

    frame_reports = []

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

        points_camera, _ = backproject(
            depth,
            confidence,
            calibration,
            confidence_threshold,
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

        distances = np.abs(
            points_world @ floor_normal
            + FLOOR_D
        )

        total_points += len(distances)

        all_distances.extend(
            distances.tolist()
        )

        near_1cm += np.sum(
            distances <= 0.01
        )

        near_2cm += np.sum(
            distances <= 0.02
        )

        near_3cm += np.sum(
            distances <= 0.03
        )

        near_5cm += np.sum(
            distances <= 0.05
        )

        frame_reports.append(
            (
                frame,
                len(points_world),
                float(np.min(distances)),
                float(np.percentile(
                    distances,
                    1
                )),
                float(np.median(distances)),
            )
        )

    distances = np.asarray(
        all_distances,
        dtype=np.float64,
    )

    print()
    print(
        f"Total points: "
        f"{total_points:,}"
    )

    if len(distances) == 0:
        print("NO POINTS")
        continue

    print(
        f"Minimum plane distance: "
        f"{np.min(distances) * 100:.3f} cm"
    )

    print(
        f"P01 distance: "
        f"{np.percentile(distances, 1) * 100:.3f} cm"
    )

    print(
        f"P05 distance: "
        f"{np.percentile(distances, 5) * 100:.3f} cm"
    )

    print(
        f"Median distance: "
        f"{np.median(distances) * 100:.3f} cm"
    )

    print()
    print(
        f"<= 1 cm: "
        f"{near_1cm:,}"
    )

    print(
        f"<= 2 cm: "
        f"{near_2cm:,}"
    )

    print(
        f"<= 3 cm: "
        f"{near_3cm:,}"
    )

    print(
        f"<= 5 cm: "
        f"{near_5cm:,}"
    )

    print()
    print("Closest frames:")

    frame_reports.sort(
        key=lambda x: x[2]
    )

    for report in frame_reports[:10]:

        frame, count, minimum, p01, median = report

        print(
            f"frame={frame:5d} "
            f"points={count:6d} "
            f"min={minimum * 100:7.3f}cm "
            f"p01={p01 * 100:7.3f}cm "
            f"median={median * 100:7.3f}cm"
        )


print()
print("=" * 70)
print("DONE")
print("=" * 70)
