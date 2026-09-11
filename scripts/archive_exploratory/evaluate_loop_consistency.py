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

PIXEL_STRIDE = 8

DEPTH_SCALES = [500.0, 1000.0, 2000.0]

POSE_CONVENTIONS = ["A", "C"]

# Minimum temporal separation between two frames.
MIN_FRAME_SEPARATION = 300

# Maximum distance between camera positions for a candidate
# revisit pair.
MAX_CAMERA_DISTANCE = 0.75

# Number of revisit pairs to evaluate.
MAX_PAIRS = 20


# ============================================================
# Odometry
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

def depth_to_camera(depth, row, scale):

    fx, fy, cx, cy = get_depth_intrinsics(row)

    h, w = depth.shape

    v, u = np.indices((h, w))

    u = u[::PIXEL_STRIDE, ::PIXEL_STRIDE]
    v = v[::PIXEL_STRIDE, ::PIXEL_STRIDE]
    d = depth[::PIXEL_STRIDE, ::PIXEL_STRIDE]

    z = d.astype(np.float64) / scale

    valid = (
        np.isfinite(z)
        & (z > 0)
    )

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
# Transformations
# ============================================================

def transform_A(P, R, t):

    return (R @ P.T).T + t


def transform_C(P, R, t):

    return (R.T @ P.T).T + t


TRANSFORMS = {
    "A": transform_A,
    "C": transform_C,
}


# ============================================================
# Find revisit pairs
# ============================================================

def find_revisit_pairs(odometry):

    positions = odometry[
        ["x", "y", "z"]
    ].to_numpy()

    pairs = []

    n = len(positions)

    for i in range(n):

        # Only compare with frames substantially later.
        j_start = i + MIN_FRAME_SEPARATION

        if j_start >= n:
            break

        for j in range(
            j_start,
            n,
            10,
        ):

            distance = np.linalg.norm(
                positions[i] - positions[j]
            )

            if distance <= MAX_CAMERA_DISTANCE:

                pairs.append(
                    (
                        distance,
                        i,
                        j,
                    )
                )

    pairs.sort()

    # Remove pairs that are too redundant.
    selected = []

    used = set()

    for distance, i, j in pairs:

        # Avoid having many pairs involving
        # exactly the same frame.
        if i in used or j in used:
            continue

        selected.append(
            (
                distance,
                i,
                j,
            )
        )

        used.add(i)
        used.add(j)

        if len(selected) >= MAX_PAIRS:
            break

    return selected


# ============================================================
# Load a transformed frame
# ============================================================

def load_world_cloud(
    index,
    depth_files,
    odometry,
    scale,
    convention,
):

    depth = cv2.imread(
        str(depth_files[index]),
        cv2.IMREAD_UNCHANGED,
    )

    if depth is None:
        return None

    row = odometry.iloc[index]

    P_camera = depth_to_camera(
        depth,
        row,
        scale,
    )

    R, t = get_pose(row)

    P_world = TRANSFORMS[
        convention
    ](
        P_camera,
        R,
        t,
    )

    valid = np.all(
        np.isfinite(P_world),
        axis=1,
    )

    return P_world[valid]


# ============================================================
# Main
# ============================================================

def main():

    if len(sys.argv) != 2:

        print(
            "Usage:\n"
            'uv run python scripts\\evaluate_loop_consistency.py '
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
    print("LOOP-CLOSURE GEOMETRIC CONSISTENCY")
    print("=" * 70)

    print(f"\nFrames: {n}")

    print(
        "\nFinding spatially close camera positions "
        "that are far apart in time..."
    )

    pairs = find_revisit_pairs(
        odometry
    )

    print(
        f"Candidate revisit pairs: {len(pairs)}"
    )

    if not pairs:

        print(
            "\nNo suitable revisit pairs found."
        )

        sys.exit(0)

    print("\nSelected pairs:")

    for distance, i, j in pairs:

        print(
            f"  frame {i:4d} ↔ frame {j:4d} "
            f"| camera distance = {distance:.3f} m"
        )

    print("\n" + "=" * 70)
    print("EVALUATION")
    print("=" * 70)

    results = []

    for convention in POSE_CONVENTIONS:

        for scale in DEPTH_SCALES:

            pair_scores = []

            for distance, i, j in pairs:

                cloud_i = load_world_cloud(
                    i,
                    depth_files,
                    odometry,
                    scale,
                    convention,
                )

                cloud_j = load_world_cloud(
                    j,
                    depth_files,
                    odometry,
                    scale,
                    convention,
                )

                if (
                    cloud_i is None
                    or cloud_j is None
                ):
                    continue

                tree = cKDTree(
                    cloud_i
                )

                distances, _ = tree.query(
                    cloud_j,
                    k=1,
                    workers=-1,
                )

                # Robust score.
                median = np.median(
                    distances
                )

                p90 = np.percentile(
                    distances,
                    90,
                )

                pair_scores.append(
                    (
                        median,
                        p90,
                    )
                )

            if not pair_scores:
                continue

            pair_scores = np.asarray(
                pair_scores
            )

            median_score = np.median(
                pair_scores[:, 0]
            )

            mean_score = np.mean(
                pair_scores[:, 0]
            )

            p90_score = np.median(
                pair_scores[:, 1]
            )

            results.append(
                {
                    "convention": convention,
                    "scale": scale,
                    "median": median_score,
                    "mean": mean_score,
                    "p90": p90_score,
                }
            )

            print(
                f"\nPose {convention}, "
                f"scale={scale:.0f}"
            )

            print(
                f"  Median NN: "
                f"{median_score:.4f} m"
            )

            print(
                f"  Mean NN:   "
                f"{mean_score:.4f} m"
            )

            print(
                f"  Median P90:"
                f" {p90_score:.4f} m"
            )

    # --------------------------------------------------------
    # Ranking
    # --------------------------------------------------------

    results.sort(
        key=lambda x: x["median"]
    )

    print("\n")
    print("=" * 70)
    print("FINAL RANKING")
    print("=" * 70)

    for rank, result in enumerate(
        results,
        start=1,
    ):

        print(
            f"{rank:2d}. "
            f"Pose {result['convention']} | "
            f"scale={result['scale']:.0f} | "
            f"median={result['median']:.4f} m | "
            f"mean={result['mean']:.4f} m | "
            f"P90={result['p90']:.4f} m"
        )


if __name__ == "__main__":
    main()
