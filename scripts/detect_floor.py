import sys
from pathlib import Path

import numpy as np
import open3d as o3d


# ============================================================
# Configuration
# ============================================================

NORMAL_Y_THRESHOLD = 0.95

MIN_FLOOR_POINTS = 20_000
MIN_FOOTPRINT_AREA_M2 = 1.0
MIN_RELATIVE_FOOTPRINT_AREA = 0.35
MIN_RELATIVE_SUPPORT = 0.10

RANSAC_DISTANCE_THRESHOLD = 0.03
RANSAC_ITERATIONS = 3000
MAX_PLANES = 20
RANSAC_RANDOM_SEED = 7


# ============================================================
# Utilities
# ============================================================

def normalize_plane(plane_model):
    plane = np.asarray(
        plane_model,
        dtype=np.float64,
    )

    normal_norm = np.linalg.norm(
        plane[:3]
    )

    if normal_norm < 1e-12:
        raise ValueError(
            "Invalid plane with zero normal."
        )

    plane = plane / normal_norm

    if plane[1] < 0:
        plane = -plane

    return plane


def plane_angle_to_y(plane):
    return float(
        np.degrees(
            np.arccos(
                np.clip(
                    abs(plane[1]),
                    0.0,
                    1.0,
                )
            )
        )
    )


def candidate_metrics(
    plane,
    points,
    original_inliers,
    iteration,
):
    x_low, y_low, z_low = np.percentile(
        points,
        1,
        axis=0,
    )

    x_high, y_high, z_high = np.percentile(
        points,
        99,
        axis=0,
    )

    footprint_x = float(
        max(0.0, x_high - x_low)
    )

    footprint_z = float(
        max(0.0, z_high - z_low)
    )

    footprint_area = (
        footprint_x
        * footprint_z
    )

    residuals = np.abs(
        points @ plane[:3]
        + plane[3]
    )

    return {
        "iteration": int(iteration),
        "plane": plane,
        "normal": plane[:3].copy(),
        "points": int(len(points)),
        "centroid": points.mean(axis=0),
        "median_y": float(np.median(points[:, 1])),
        "y_range_p01_p99": [
            float(y_low),
            float(y_high),
        ],
        "footprint_x_m": footprint_x,
        "footprint_z_m": footprint_z,
        "footprint_area_m2": float(footprint_area),
        "angle_to_y": plane_angle_to_y(plane),
        "residual_p95_m": float(
            np.percentile(residuals, 95)
        ),
        "inliers": original_inliers.copy(),
    }


def select_floor_candidate(candidates):
    horizontal = [
        candidate
        for candidate in candidates
        if abs(candidate["normal"][1]) >= NORMAL_Y_THRESHOLD
    ]

    if not horizontal:
        return None, []

    max_points = max(
        candidate["points"]
        for candidate in horizontal
    )

    max_area = max(
        candidate["footprint_area_m2"]
        for candidate in horizontal
    )

    min_points = max(
        MIN_FLOOR_POINTS,
        int(max_points * MIN_RELATIVE_SUPPORT),
    )

    min_area = max(
        MIN_FOOTPRINT_AREA_M2,
        max_area * MIN_RELATIVE_FOOTPRINT_AREA,
    )

    viable = [
        candidate
        for candidate in horizontal
        if (
            candidate["points"] >= min_points
            and candidate["footprint_area_m2"] >= min_area
        )
    ]

    if not viable:
        return None, horizontal

    viable.sort(
        key=lambda candidate: (
            candidate["median_y"],
            -candidate["footprint_area_m2"],
            -candidate["points"],
        )
    )

    return viable[0], horizontal


def print_candidate(
    label,
    candidate,
):
    plane = candidate["plane"]

    print(f"\n{label}")

    print(
        f"  Iteration: {candidate['iteration']}"
    )

    print(
        f"  Points: {candidate['points']:,}"
    )

    print(
        f"  Median Y: {candidate['median_y']:.4f} m"
    )

    print(
        "  Y p01/p99: "
        f"{candidate['y_range_p01_p99'][0]:.4f} -> "
        f"{candidate['y_range_p01_p99'][1]:.4f} m"
    )

    print(
        "  Footprint: "
        f"{candidate['footprint_x_m']:.3f} x "
        f"{candidate['footprint_z_m']:.3f} m "
        f"({candidate['footprint_area_m2']:.3f} m^2)"
    )

    print(
        f"  Angle to Y: {candidate['angle_to_y']:.3f}°"
    )

    print(
        f"  Residual P95: "
        f"{candidate['residual_p95_m']:.4f} m"
    )

    print(
        f"  Normal: {candidate['normal']}"
    )

    print(
        f"  Centroid: {candidate['centroid']}"
    )

    print(
        "  Plane: "
        f"{plane[0]:.6f}x + "
        f"{plane[1]:.6f}y + "
        f"{plane[2]:.6f}z + "
        f"{plane[3]:.6f} = 0"
    )


# ============================================================
# Main
# ============================================================

def main():

    if len(sys.argv) != 2:

        print(
            "Usage:\n"
            "uv run python scripts\\detect_floor.py "
            "outputs\\single_room\\pointcloud_production.ply"
        )

        sys.exit(1)

    input_path = Path(
        sys.argv[1]
    )

    # --------------------------------------------------------
    # Load point cloud
    # --------------------------------------------------------

    cloud = o3d.io.read_point_cloud(
        str(input_path)
    )

    if cloud.is_empty():

        raise RuntimeError(
            "Point cloud is empty."
        )

    points = np.asarray(
        cloud.points
    )

    print("=" * 70)
    print("FLOOR PLANE DETECTION")
    print("=" * 70)

    print(
        f"\nInput points: {len(points):,}"
    )

    o3d.utility.random.seed(
        RANSAC_RANDOM_SEED
    )

    print(
        f"RANSAC random seed: {RANSAC_RANDOM_SEED}"
    )

    # --------------------------------------------------------
    # RANSAC planes
    # --------------------------------------------------------

    remaining = cloud
    remaining_indices = np.arange(
        len(points)
    )

    candidates = []

    print(
        "\n" + "=" * 70
    )

    print(
        "RANSAC PLANE SEARCH"
    )

    print(
        "=" * 70
    )

    for iteration in range(MAX_PLANES):

        if len(remaining.points) < MIN_FLOOR_POINTS:
            break

        # ----------------------------------------------------
        # Detect dominant plane
        # ----------------------------------------------------

        plane_model, inliers = (
            remaining.segment_plane(
                distance_threshold=RANSAC_DISTANCE_THRESHOLD,
                ransac_n=3,
                num_iterations=RANSAC_ITERATIONS,
            )
        )

        if len(inliers) < MIN_FLOOR_POINTS:
            break

        # ----------------------------------------------------
        # Get plane points
        # ----------------------------------------------------

        candidate_points = np.asarray(
            remaining.points
        )[inliers]

        original_inliers = remaining_indices[
            np.asarray(
                inliers,
                dtype=np.int64,
            )
        ]

        plane = normalize_plane(
            plane_model
        )

        candidate = candidate_metrics(
            plane,
            candidate_points,
            original_inliers,
            iteration + 1,
        )

        candidates.append(
            candidate
        )

        print_candidate(
            f"Plane {iteration + 1}",
            candidate,
        )

        # ----------------------------------------------------
        # Remove detected plane and continue searching
        # ----------------------------------------------------

        remaining = remaining.select_by_index(
            inliers,
            invert=True,
        )

        remaining_indices = np.delete(
            remaining_indices,
            np.asarray(
                inliers,
                dtype=np.int64,
            )
        )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print("FLOOR CANDIDATES")

    print(
        "=" * 70
    )

    floor, horizontal_candidates = select_floor_candidate(
        candidates
    )

    if not horizontal_candidates:

        print(
            "\nNo large near-horizontal planes found."
        )

        raise RuntimeError(
            "Floor detection failed: no near-horizontal plane candidates."
        )

    for i, candidate in enumerate(
        sorted(
            horizontal_candidates,
            key=lambda c: (
                c["median_y"],
                -c["footprint_area_m2"],
                -c["points"],
            ),
        ),
        start=1,
    ):
        print_candidate(
            f"Horizontal candidate {i}",
            candidate,
        )

    if floor is None:
        print(
            "\nNear-horizontal planes were found, but none had enough "
            "support and footprint to be accepted as the room floor."
        )

        raise RuntimeError(
            "Floor detection failed: no viable floor candidate."
        )

    print(
        "\n" + "=" * 70
    )

    print(
        "SELECTED FLOOR"
    )

    print(
        "=" * 70
    )

    print(
        f"\nPoints: "
        f"{floor['points']:,}"
    )

    print(
        f"Median Y: "
        f"{floor['median_y']:.4f} m"
    )

    print(
        "Footprint: "
        f"{floor['footprint_x_m']:.3f} x "
        f"{floor['footprint_z_m']:.3f} m "
        f"({floor['footprint_area_m2']:.3f} m^2)"
    )

    print(
        f"Normal: "
        f"{floor['normal']}"
    )

    print(
        f"Angle to Y: "
        f"{floor['angle_to_y']:.3f}°"
    )

    print(
        f"Centroid: "
        f"{floor['centroid']}"
    )

    print(
        f"Residual P95: "
        f"{floor['residual_p95_m']:.4f} m"
    )

    # --------------------------------------------------------
    # Save floor points
    # --------------------------------------------------------

    floor_cloud = cloud.select_by_index(
        floor["inliers"]
    )

    output_path = (
        input_path.parent
        / "floor_plane.ply"
    )

    success = o3d.io.write_point_cloud(
        str(output_path),
        floor_cloud,
    )

    if not success:

        raise RuntimeError(
            f"Failed to save floor plane to "
            f"{output_path}"
        )

    print(
        f"\nSaved floor plane:"
        f"\n{output_path}"
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()
