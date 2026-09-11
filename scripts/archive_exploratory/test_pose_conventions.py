import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import open3d as o3d
from scipy.spatial.transform import Rotation


# ============================================================
# Configuration
# ============================================================

RGB_WIDTH = 1920
RGB_HEIGHT = 1440

DEPTH_WIDTH = 256
DEPTH_HEIGHT = 192

DEPTH_SCALE = 1000.0

# Use sparse sampling for this diagnostic.
FRAME_STRIDE = 30
PIXEL_STRIDE = 8

OUTPUT_DIR = Path("outputs") / "single_room"


# ============================================================
# Helpers
# ============================================================

def load_odometry(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # The dataset has leading spaces in some column names.
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
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing odometry columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    return df


def get_depth_intrinsics(odometry_row):
    """
    Odometry intrinsics appear to be specified for the RGB
    resolution (1920 x 1440).

    Depth is 256 x 192, so scale the RGB intrinsics down.
    """

    scale_x = DEPTH_WIDTH / RGB_WIDTH
    scale_y = DEPTH_HEIGHT / RGB_HEIGHT

    fx = odometry_row["fx"] * scale_x
    fy = odometry_row["fy"] * scale_y
    cx = odometry_row["cx"] * scale_x
    cy = odometry_row["cy"] * scale_y

    return fx, fy, cx, cy


def depth_to_camera(depth, fx, fy, cx, cy):
    """
    Convert depth image pixels into 3D camera coordinates.

    Returns:
        Nx3 array in camera coordinates.
    """

    h, w = depth.shape

    v, u = np.indices((h, w))

    # Sparse sampling.
    u = u[::PIXEL_STRIDE, ::PIXEL_STRIDE]
    v = v[::PIXEL_STRIDE, ::PIXEL_STRIDE]
    depth = depth[::PIXEL_STRIDE, ::PIXEL_STRIDE]

    z = depth.astype(np.float64) / DEPTH_SCALE

    valid = np.isfinite(z) & (z > 0)

    u = u[valid]
    v = v[valid]
    z = z[valid]

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    return np.column_stack((x, y, z))


def get_rotation(row):
    """
    scipy expects quaternion ordering:
        [qx, qy, qz, qw]
    """

    q = np.array([
        row["qx"],
        row["qy"],
        row["qz"],
        row["qw"],
    ], dtype=np.float64)

    return Rotation.from_quat(q).as_matrix()


# ============================================================
# Pose conventions
# ============================================================

def transform_A(P, R, t):
    """
    A:
        P_world = R @ P_camera + t
    """
    return (R @ P.T).T + t


def transform_B(P, R, t):
    """
    B:
        P_world = R.T @ (P_camera - t)
    """
    return (R.T @ (P - t).T).T


def transform_C(P, R, t):
    """
    C:
        P_world = R.T @ P_camera + t
    """
    return (R.T @ P.T).T + t


def transform_D(P, R, t):
    """
    D:
        P_world = R @ (P_camera - t)
    """
    return (R @ (P - t).T).T


TRANSFORMS = {
    "A": transform_A,
    "B": transform_B,
    "C": transform_C,
    "D": transform_D,
}


# ============================================================
# Main
# ============================================================

def main():

    if len(sys.argv) != 2:
        print(
            "Usage:\n"
            '  uv run python scripts\\test_pose_conventions.py '
            '"..\\cozmo-dataset\\raw_dataset\\single_room\\c00a170fe1"'
        )
        sys.exit(1)

    capture_dir = Path(sys.argv[1])

    rgb_path = capture_dir / "rgb.mp4"
    depth_dir = capture_dir / "depth"
    odometry_path = capture_dir / "odometry.csv"

    if not rgb_path.exists():
        raise FileNotFoundError(f"Missing: {rgb_path}")

    if not depth_dir.exists():
        raise FileNotFoundError(f"Missing: {depth_dir}")

    if not odometry_path.exists():
        raise FileNotFoundError(f"Missing: {odometry_path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Load odometry
    # --------------------------------------------------------

    odometry = load_odometry(odometry_path)

    print(f"Odometry rows: {len(odometry)}")

    # --------------------------------------------------------
    # Find depth frames
    # --------------------------------------------------------

    depth_files = sorted(depth_dir.glob("*.png"))

    print(f"Depth frames:  {len(depth_files)}")

    if len(depth_files) != len(odometry):
        print(
            "\nWARNING:"
            "\nDepth frame count and odometry row count differ."
        )

    # --------------------------------------------------------
    # Open RGB video just to verify frame count
    # --------------------------------------------------------

    cap = cv2.VideoCapture(str(rgb_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {rgb_path}")

    video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.release()

    print(f"RGB frames:    {video_frames}")

    # --------------------------------------------------------
    # Containers for each convention
    # --------------------------------------------------------

    clouds = {
        "A": [],
        "B": [],
        "C": [],
        "D": [],
    }

    # --------------------------------------------------------
    # Process frames
    # --------------------------------------------------------

    frame_indices = range(
        0,
        min(len(depth_files), len(odometry)),
        FRAME_STRIDE,
    )

    processed = 0

    for frame_idx in frame_indices:

        depth_path = depth_files[frame_idx]

        depth = cv2.imread(
            str(depth_path),
            cv2.IMREAD_UNCHANGED,
        )

        if depth is None:
            print(f"Could not read {depth_path}")
            continue

        row = odometry.iloc[frame_idx]

        fx, fy, cx, cy = get_depth_intrinsics(row)

        P_camera = depth_to_camera(
            depth,
            fx,
            fy,
            cx,
            cy,
        )

        if len(P_camera) == 0:
            continue

        R = get_rotation(row)

        t = np.array([
            row["x"],
            row["y"],
            row["z"],
        ], dtype=np.float64)

        for name, transform in TRANSFORMS.items():

            P_world = transform(
                P_camera,
                R,
                t,
            )

            clouds[name].append(P_world)

        processed += 1

        if processed % 5 == 0:
            print(
                f"Processed {processed} diagnostic frames..."
            )

    # --------------------------------------------------------
    # Save clouds
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    for name in ["A", "B", "C", "D"]:

        if not clouds[name]:
            print(f"\n{name}: NO POINTS")
            continue

        points = np.vstack(clouds[name])

        # Remove obviously invalid values.
        valid = np.all(np.isfinite(points), axis=1)
        points = points[valid]

        print(f"\nPose {name}")
        print(f"Points: {len(points):,}")

        print(
            "Bounds:"
            f"\n  X: {points[:, 0].min():.3f}"
            f" → {points[:, 0].max():.3f}"
            f"\n  Y: {points[:, 1].min():.3f}"
            f" → {points[:, 1].max():.3f}"
            f"\n  Z: {points[:, 2].min():.3f}"
            f" → {points[:, 2].max():.3f}"
        )

        pcd = o3d.geometry.PointCloud()

        pcd.points = o3d.utility.Vector3dVector(points)

        output_path = OUTPUT_DIR / f"pose_{name}.ply"

        o3d.io.write_point_cloud(
            str(output_path),
            pcd,
        )

        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
