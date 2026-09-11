from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

from cozmo_ai.geometry.backprojection import depth_to_camera_points
from cozmo_ai.io.calibration import load_camera_calibration
from cozmo_ai.io.odometry import load_odometry


DATASET = Path("../cozmo-dataset/raw_dataset")
CAPTURE = DATASET / "single_scan_with_ceiling" / "c7d28f72c6"

FRAME_ID = 5610


def load_rgb_frame(video_path, frame_id):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ok, frame = cap.read()
    cap.release()

    if not ok:
        raise RuntimeError(f"Could not read RGB frame {frame_id}")

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def load_depth_frame(depth_path):
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)

    if depth is None:
        raise RuntimeError(f"Could not read {depth_path}")

    return depth


def ransac_plane(points, iterations=300, threshold=0.025, seed=42):
    """
    Generic RANSAC plane fitting.
    Returns:
        normal, d, inlier_mask
    where plane equation is:

        n . p + d = 0
    """

    rng = np.random.default_rng(seed)

    if len(points) < 3:
        return None, None, np.zeros(len(points), dtype=bool)

    best_mask = None
    best_count = 0
    best_plane = None

    for _ in range(iterations):

        ids = rng.choice(len(points), 3, replace=False)

        p1, p2, p3 = points[ids]

        n = np.cross(p2 - p1, p3 - p1)

        norm = np.linalg.norm(n)

        if norm < 1e-8:
            continue

        n /= norm

        d = -np.dot(n, p1)

        distances = np.abs(points @ n + d)

        mask = distances < threshold

        count = int(mask.sum())

        if count > best_count:
            best_count = count
            best_mask = mask
            best_plane = (n, d)

    if best_plane is None:
        return None, None, np.zeros(len(points), dtype=bool)

    return best_plane[0], best_plane[1], best_mask


def main():

    calibration = load_camera_calibration(CAPTURE)
    odometry = load_odometry(CAPTURE / "odometry.csv")

    depth_files = sorted((CAPTURE / "depth").glob("*.png"))

    depth = load_depth_frame(depth_files[FRAME_ID])

    # ------------------------------------------------------------
    # RGB
    # ------------------------------------------------------------

    rgb = load_rgb_frame(
        CAPTURE / "rgb.mp4",
        FRAME_ID
    )

    # ------------------------------------------------------------
    # Backproject depth
    # ------------------------------------------------------------

    points_camera = depth_to_camera_points(
        depth,
        calibration,
        pixel_stride=1
    )

    print("Camera points:", len(points_camera))

    # ------------------------------------------------------------
    # Camera intrinsics
    # ------------------------------------------------------------

    fx = calibration.fx_rgb
    fy = calibration.fy_rgb
    cx = calibration.cx_rgb
    cy = calibration.cy_rgb

    depth_h, depth_w = depth.shape

    scale_x = calibration.rgb_width / depth_w
    scale_y = calibration.rgb_height / depth_h

    fx_d = calibration.fx_depth
    fy_d = calibration.fy_depth
    cx_d = calibration.cx_depth
    cy_d = calibration.cy_depth

    # ------------------------------------------------------------
    # Recover depth pixel coordinates for every point
    # ------------------------------------------------------------

    v, u = np.indices(depth.shape)

    z = depth.astype(np.float64) / calibration.depth_scale

    valid = (
        np.isfinite(z)
        & (z > 0)
    )

    u = u[valid]
    v = v[valid]

    points = points_camera

    # ------------------------------------------------------------
    # Fit arbitrary planes just for visualization
    # ------------------------------------------------------------

    n1, d1, mask1 = ransac_plane(
        points,
        iterations=500,
        threshold=0.025
    )

    if n1 is None:
        raise RuntimeError("Could not fit first plane")

    print()
    print("PLANE 1")
    print("normal:", n1)
    print("d:", d1)
    print("inliers:", mask1.sum())

    remaining = points[~mask1]

    n2, d2, mask2_remaining = ransac_plane(
        remaining,
        iterations=500,
        threshold=0.025,
        seed=123
    )

    print()
    print("PLANE 2")
    print("normal:", n2)
    print("d:", d2)
    print("inliers:", mask2_remaining.sum())

    # ------------------------------------------------------------
    # Reconstruct full masks in depth-image coordinates
    # ------------------------------------------------------------

    point_indices = np.flatnonzero(valid)

    plane1_indices = point_indices[mask1]

    remaining_indices = point_indices[~mask1]

    plane2_indices = remaining_indices[mask2_remaining]

    # ------------------------------------------------------------
    # Project depth pixels to RGB
    # ------------------------------------------------------------

    rgb_u = np.round(u * scale_x).astype(int)
    rgb_v = np.round(v * scale_y).astype(int)

    rgb_u = np.clip(rgb_u, 0, calibration.rgb_width - 1)
    rgb_v = np.clip(rgb_v, 0, calibration.rgb_height - 1)

    # ------------------------------------------------------------
    # Create overlays
    # ------------------------------------------------------------

    overlay1 = rgb.copy()
    overlay2 = rgb.copy()

    # Plane 1
    p1_u = rgb_u[mask1]
    p1_v = rgb_v[mask1]

    overlay1[p1_v, p1_u] = [255, 0, 0]

    # Plane 2
    p2_u = rgb_u[~mask1][mask2_remaining]
    p2_v = rgb_v[~mask1][mask2_remaining]

    overlay2[p2_v, p2_u] = [0, 255, 0]

    # ------------------------------------------------------------
    # Blend
    # ------------------------------------------------------------

    result = rgb.copy()

    alpha = 0.65

    result[p1_v, p1_u] = (
        alpha * np.array([255, 0, 0])
        + (1 - alpha) * result[p1_v, p1_u]
    ).astype(np.uint8)

    result[p2_v, p2_u] = (
        alpha * np.array([0, 255, 0])
        + (1 - alpha) * result[p2_v, p2_u]
    ).astype(np.uint8)

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------

    output_dir = Path("outputs/single_scan_with_ceiling")
    output_dir.mkdir(parents=True, exist_ok=True)

    output = output_dir / f"plane_overlay_{FRAME_ID}.png"

    cv2.imwrite(
        str(output),
        cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    )

    print()
    print("Saved:", output)

    # ------------------------------------------------------------
    # Also save individual overlays
    # ------------------------------------------------------------

    cv2.imwrite(
        str(output_dir / f"plane1_{FRAME_ID}.png"),
        cv2.cvtColor(overlay1, cv2.COLOR_RGB2BGR)
    )

    cv2.imwrite(
        str(output_dir / f"plane2_{FRAME_ID}.png"),
        cv2.cvtColor(overlay2, cv2.COLOR_RGB2BGR)
    )

    # ------------------------------------------------------------
    # Display
    # ------------------------------------------------------------

    plt.figure(figsize=(14, 10))

    plt.imshow(result)

    plt.title(
        f"Frame {FRAME_ID}\n"
        "Plane 1 = RED | Plane 2 = GREEN"
    )

    plt.axis("off")

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
