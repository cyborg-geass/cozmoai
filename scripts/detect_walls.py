import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d


# ============================================================
# Configuration
# ============================================================

NORMAL_Y_THRESHOLD = 0.15

MIN_WALL_POINTS = 8_000

RANSAC_DISTANCE_THRESHOLD = 0.03
RANSAC_ITERATIONS = 3000

MAX_PLANES = 20

MIN_HEIGHT_ABOVE_FLOOR = 0.10
MAX_HEIGHT_ABOVE_FLOOR = 3.5

PARALLEL_ANGLE_DEGREES = 10.0

OUTPUT_PREFIX = "wall_plane"


# ============================================================
# Utility functions
# ============================================================

def angle_between_normals(n1, n2):

    dot = np.clip(
        abs(np.dot(n1, n2)),
        0.0,
        1.0,
    )

    return np.degrees(
        np.arccos(dot)
    )


def point_plane_distance(points, plane):

    normal = plane[:3]
    d = plane[3]

    return np.abs(
        points @ normal + d
    )


def normalize_plane(plane):

    plane = np.asarray(
        plane,
        dtype=np.float64,
    )

    normal = plane[:3]

    norm = np.linalg.norm(
        normal
    )

    if norm < 1e-12:
        raise ValueError(
            "Invalid plane with zero normal."
        )

    return plane / norm


def load_floor_plane(floor_path):

    floor_cloud = o3d.io.read_point_cloud(
        str(floor_path)
    )

    if floor_cloud.is_empty():

        raise RuntimeError(
            f"Floor point cloud is empty: {floor_path}"
        )

    plane_model, inliers = (
        floor_cloud.segment_plane(
            distance_threshold=0.03,
            ransac_n=3,
            num_iterations=3000,
        )
    )

    if len(inliers) < 1000:

        raise RuntimeError(
            "Could not reliably refit the floor plane."
        )

    plane = normalize_plane(
        plane_model
    )

    if plane[1] < 0:
        plane = -plane

    return plane


def height_above_floor(
    points,
    floor_plane,
):

    normal = floor_plane[:3]
    d = floor_plane[3]

    return (
        points @ normal + d
    )


# ============================================================
# Main
# ============================================================

def main():

    if len(sys.argv) != 2:

        print(
            "Usage:\n"
            "uv run python scripts\\detect_walls.py "
            "outputs\\single_room\\pointcloud_production.ply"
        )

        sys.exit(1)

    input_path = Path(
        sys.argv[1]
    )

    if not input_path.exists():

        raise FileNotFoundError(
            f"Point cloud not found: {input_path}"
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

    print("=" * 70)
    print("WALL PLANE DETECTION")
    print("=" * 70)

    print(
        f"\nInput points: "
        f"{len(cloud.points):,}"
    )

    # --------------------------------------------------------
    # Load floor
    # --------------------------------------------------------

    floor_path = (
        input_path.parent
        / "floor_plane.ply"
    )

    if not floor_path.exists():

        raise FileNotFoundError(
            "\nFloor plane not found:\n"
            f"{floor_path}\n\n"
            "Run detect_floor.py first."
        )

    floor_plane = load_floor_plane(
        floor_path
    )

    print(
        "\nFloor plane:"
    )

    print(
        f"  Normal: "
        f"{floor_plane[:3]}"
    )

    print(
        f"  Equation: "
        f"{floor_plane[0]:.6f}x + "
        f"{floor_plane[1]:.6f}y + "
        f"{floor_plane[2]:.6f}z + "
        f"{floor_plane[3]:.6f} = 0"
    )

    # --------------------------------------------------------
    # Keep points above floor
    # --------------------------------------------------------

    all_points = np.asarray(
        cloud.points
    )

    signed_height = height_above_floor(
        all_points,
        floor_plane,
    )

    height_mask = (
        (signed_height >= MIN_HEIGHT_ABOVE_FLOOR)
        &
        (signed_height <= MAX_HEIGHT_ABOVE_FLOOR)
    )

    candidate_indices = np.where(
        height_mask
    )[0]

    print(
        f"\nPoints above floor: "
        f"{len(candidate_indices):,}"
    )

    wall_cloud = cloud.select_by_index(
        candidate_indices
    )

    # --------------------------------------------------------
    # RANSAC
    # --------------------------------------------------------

    remaining = wall_cloud

    candidates = []

    print(
        "\n" + "=" * 70
    )

    print(
        "RANSAC WALL SEARCH"
    )

    print(
        "=" * 70
    )

    for iteration in range(
        MAX_PLANES
    ):

        if (
            len(remaining.points)
            < MIN_WALL_POINTS
        ):
            break

        plane_model, inliers = (
            remaining.segment_plane(
                distance_threshold=RANSAC_DISTANCE_THRESHOLD,
                ransac_n=3,
                num_iterations=RANSAC_ITERATIONS,
            )
        )

        if len(inliers) < MIN_WALL_POINTS:

            break

        plane = normalize_plane(
            plane_model
        )

        normal = plane[:3]

        if normal[1] < 0:

            normal = -normal
            plane = -plane

        is_vertical = (
            abs(normal[1])
            <= NORMAL_Y_THRESHOLD
        )

        if is_vertical:

            plane_points = np.asarray(
                remaining.points
            )[inliers]

            centroid = (
                plane_points.mean(
                    axis=0
                )
            )

            candidates.append(
                {
                    "plane": plane.copy(),

                    "normal": normal.copy(),

                    "points": int(
                        len(inliers)
                    ),

                    "centroid": centroid.copy(),

                    "inliers": inliers.copy(),
                }
            )

        remaining = (
            remaining.select_by_index(
                inliers,
                invert=True,
            )
        )

    # --------------------------------------------------------
    # Check candidates
    # --------------------------------------------------------

    if not candidates:

        print(
            "\nNo wall candidates found."
        )

        return

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    candidates.sort(
        key=lambda x: -x["points"]
    )

    # --------------------------------------------------------
    # Print candidates
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "WALL CANDIDATES"
    )

    print(
        "=" * 70
    )

    for i, candidate in enumerate(
        candidates,
        start=1,
    ):

        normal = candidate[
            "normal"
        ]

        plane = candidate[
            "plane"
        ]

        print(
            f"\nWall Candidate {i}"
        )

        print(
            f"  Points: "
            f"{candidate['points']:,}"
        )

        print(
            f"  Normal: "
            f"{normal}"
        )

        print(
            f"  Normal Y: "
            f"{normal[1]:.6f}"
        )

        print(
            f"  Centroid: "
            f"{candidate['centroid']}"
        )

        print(
            f"  Plane: "
            f"{plane[0]:.6f}x + "
            f"{plane[1]:.6f}y + "
            f"{plane[2]:.6f}z + "
            f"{plane[3]:.6f} = 0"
        )

    # --------------------------------------------------------
    # Orientation groups
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "WALL ORIENTATION GROUPS"
    )

    print(
        "=" * 70
    )

    groups = []

    for candidate in candidates:

        assigned = False

        for group in groups:

            representative = group[0]

            angle = angle_between_normals(
                candidate["normal"],
                representative["normal"],
            )

            if (
                angle
                <= PARALLEL_ANGLE_DEGREES
            ):

                group.append(
                    candidate
                )

                assigned = True

                break

        if not assigned:

            groups.append(
                [candidate]
            )

    for i, group in enumerate(
        groups,
        start=1,
    ):

        representative = group[0][
            "normal"
        ]

        total_points = sum(
            candidate["points"]
            for candidate in group
        )

        print(
            f"\nOrientation Group {i}"
        )

        print(
            f"  Members: "
            f"{len(group)}"
        )

        print(
            f"  Total points: "
            f"{total_points:,}"
        )

        print(
            f"  Representative normal: "
            f"{representative}"
        )

        for j, candidate in enumerate(
            group,
            start=1,
        ):

            print(
                f"    Plane {j}: "
                f"{candidate['points']:,} points, "
                f"centroid="
                f"{candidate['centroid']}"
            )

    # --------------------------------------------------------
    # Save individual wall point clouds
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "SAVING WALL PLANES"
    )

    print(
        "=" * 70
    )

    wall_points = (
        np.asarray(cloud.points)
        [candidate_indices]
    )

    json_walls = []

    for i, candidate in enumerate(
        candidates,
        start=1,
    ):

        distances = point_plane_distance(
            wall_points,
            candidate["plane"],
        )

        inlier_mask = (
            distances
            <= RANSAC_DISTANCE_THRESHOLD
        )

        selected_points = (
            wall_points[inlier_mask]
        )

        output_cloud = (
            o3d.geometry.PointCloud()
        )

        output_cloud.points = (
            o3d.utility.Vector3dVector(
                selected_points
            )
        )

        output_path = (
            input_path.parent
            / f"{OUTPUT_PREFIX}_{i}.ply"
        )

        o3d.io.write_point_cloud(
            str(output_path),
            output_cloud,
        )

        print(
            f"\nWall {i}: "
            f"{len(selected_points):,} points"
        )

        print(
            f"  Saved: "
            f"{output_path}"
        )

        # ----------------------------------------------------
        # JSON-safe metadata
        # ----------------------------------------------------

        json_walls.append(
            {
                "wall_id": i,

                "plane": [
                    float(x)
                    for x in candidate["plane"]
                ],

                "normal": [
                    float(x)
                    for x in candidate["normal"]
                ],

                "points": int(
                    candidate["points"]
                ),

                "saved_points": int(
                    len(selected_points)
                ),

                "centroid": [
                    float(x)
                    for x in candidate["centroid"]
                ],
            }
        )

    # --------------------------------------------------------
    # Save wall metadata
    # --------------------------------------------------------

    walls_json_path = (
        input_path.parent
        / "walls.json"
    )

    walls_data = {
        "source_pointcloud": str(
            input_path
        ),

        "floor_plane": [
            float(x)
            for x in floor_plane
        ],

        "configuration": {
            "normal_y_threshold": NORMAL_Y_THRESHOLD,
            "min_wall_points": MIN_WALL_POINTS,
            "ransac_distance_threshold": RANSAC_DISTANCE_THRESHOLD,
            "ransac_iterations": RANSAC_ITERATIONS,
            "min_height_above_floor": MIN_HEIGHT_ABOVE_FLOOR,
            "max_height_above_floor": MAX_HEIGHT_ABOVE_FLOOR,
            "parallel_angle_degrees": PARALLEL_ANGLE_DEGREES,
        },

        "walls": json_walls,
    }

    with open(
        walls_json_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            walls_data,
            f,
            indent=2,
        )

    print(
        f"\nSaved wall metadata:"
        f"\n{walls_json_path}"
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "WALL DETECTION COMPLETE"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()
