import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree


# ============================================================
# Configuration
# ============================================================

RGB_WIDTH = 1920
RGB_HEIGHT = 1440

DEPTH_WIDTH = 256
DEPTH_HEIGHT = 192

FRAME_STRIDE = 1
PIXEL_STRIDE = 8

# Candidate depth scales.
DEPTH_SCALES = [500.0, 1000.0, 2000.0]


# ============================================================
# Load odometry
# ============================================================

def load_odometry(path):

    df = pd.read_csv(path)

    df.columns = df.columns.str.strip()

    required = [
        "frame",
        "x",
        "y",
        "z",
        "qx",
        "qy",
        "qz",
        "qw",
        "fx",
        "fy",
        "cx",
        "cy",
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    return df


# ============================================================
# Intrinsics
# ============================================================

def get_depth_intrinsics(row):

    sx = DEPTH_WIDTH / RGB_WIDTH
    sy = DEPTH_HEIGHT / RGB_HEIGHT

    fx = row["fx"] * sx
    fy = row["fy"] * sy
    cx = row["cx"] * sx
    cy = row["cy"] * sy

    return fx, fy, cx, cy


# ============================================================
# Depth → camera coordinates
# ============================================================

def depth_to_camera(depth, row, depth_scale):

    fx, fy, cx, cy = get_depth_intrinsics(row)

    h, w = depth.shape

    v, u = np.indices((h, w))

    u = u[::PIXEL_STRIDE, ::PIXEL_STRIDE]
    v = v[::PIXEL_STRIDE, ::PIXEL_STRIDE]
    d = depth[::PIXEL_STRIDE, ::PIXEL_STRIDE]

    z = d.astype(np.float64) / depth_scale

    valid = np.isfinite(z) & (z > 0)

    u = u[valid]
    v = v[valid]
    z = z[valid]

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    return np.column_stack((x, y, z))


# ============================================================
# Pose
# ============================================================

def get_pose(row):

    q = np.array([
        row["qx"],
        row["qy"],
        row["qz"],
        row["qw"],
    ])

    R = Rotation.from_quat(q).as_matrix()

    t = np.array([
        row["x"],
        row["y"],
        row["z"],
    ])

    return R, t


# ============================================================
# Transform conventions
# ============================================================

def transform_A(P, R, t):
    return (R @ P.T).T + t


def transform_B(P, R, t):
    return (R.T @ (P - t).T).T


def transform_C(P, R, t):
    return (R.T @ P.T).T + t


def transform_D(P, R, t):
    return (R @ (P - t).T).T


TRANSFORMS = {
    "A": transform_A,
    "B": transform_B,
    "C": transform_C,
    "D": transform_D,
}


# ============================================================
# Main evaluation
# ============================================================

def main():

    if len(sys.argv) != 2:

        print(
            "Usage:\n"
            'uv run python scripts\\evaluate_geometry.py '
            '"..\\cozmo-dataset\\raw_dataset\\single_room\\c00a170fe1"'
        )

        sys.exit(1)

    capture_dir = Path(sys.argv[1])

    depth_dir = capture_dir / "depth"
    odometry_path = capture_dir / "odometry.csv"

    depth_files = sorted(
        depth_dir.glob("*.png")
    )

    odometry = load_odometry(
        odometry_path
    )

    n = min(
        len(depth_files),
        len(odometry)
    )

    print("=" * 70)
    print("GEOMETRIC CONSISTENCY EVALUATION")
    print("=" * 70)

    print(f"\nDepth frames:    {len(depth_files)}")
    print(f"Odometry rows:   {len(odometry)}")
    print(f"Frames evaluated: {n}")

    print(
        "\nComparing consecutive frames using "
        "nearest-neighbor distances."
    )

    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    results = []

    # --------------------------------------------------------
    # Evaluate each scale and pose convention
    # --------------------------------------------------------

    for scale in DEPTH_SCALES:

        print(
            f"\nDepth scale = {scale}"
        )

        for convention, transform in TRANSFORMS.items():

            distances = []

            previous_cloud = None

            for i in range(
                0,
                n,
                FRAME_STRIDE
            ):

                depth = cv2.imread(
                    str(depth_files[i]),
                    cv2.IMREAD_UNCHANGED,
                )

                if depth is None:
                    continue

                row = odometry.iloc[i]

                P_camera = depth_to_camera(
                    depth,
                    row,
                    scale,
                )

                R, t = get_pose(row)

                P_world = transform(
                    P_camera,
                    R,
                    t,
                )

                # Compare against previous frame.
                if previous_cloud is not None:

                    tree = cKDTree(
                        previous_cloud
                    )

                    nn_distances, _ = tree.query(
                        P_world,
                        k=1,
                        workers=-1,
                    )

                    # Ignore extreme outliers.
                    nn_distances = nn_distances[
                        nn_distances < 0.5
                    ]

                    if len(nn_distances) > 0:

                        distances.append(
                            np.median(nn_distances)
                        )

                previous_cloud = P_world

            if not distances:
                continue

            distances = np.asarray(
                distances
            )

            median = np.median(distances)

            mean = np.mean(distances)

            p90 = np.percentile(
                distances,
                90
            )

            results.append({
                "convention": convention,
                "scale": scale,
                "median": median,
                "mean": mean,
                "p90": p90,
                "comparisons": len(distances),
            })

            print(
                f"  Pose {convention}: "
                f"median={median:.4f} m, "
                f"mean={mean:.4f} m, "
                f"P90={p90:.4f} m"
            )

    # --------------------------------------------------------
    # Final ranking
    # --------------------------------------------------------

    results.sort(
        key=lambda x: x["median"]
    )

    print("\n")
    print("=" * 70)
    print("RANKING")
    print("=" * 70)

    print(
        "\nLower median nearest-neighbor distance "
        "means better frame-to-frame consistency.\n"
    )

    for rank, result in enumerate(
        results,
        start=1
    ):

        print(
            f"{rank:2d}. "
            f"Pose {result['convention']} | "
            f"scale={result['scale']:4.0f} | "
            f"median={result['median']:.4f} m | "
            f"mean={result['mean']:.4f} m | "
            f"P90={result['p90']:.4f} m"
        )


if __name__ == "__main__":
    main()
