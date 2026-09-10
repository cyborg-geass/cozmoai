import sys
from pathlib import Path

import numpy as np
import open3d as o3d


# ============================================================
# Configuration
# ============================================================

# A wall normal should be approximately horizontal.
# Since Y is the vertical axis, wall normals should have
# very little Y component.
MAX_NORMAL_Y = 0.15

# Minimum number of points required for a wall candidate.
MIN_WALL_POINTS = 8_000

# RANSAC parameters.
RANSAC_DISTANCE_THRESHOLD = 0.03
RANSAC_ITERATIONS = 3000

# Maximum number of planes to extract.
MAX_PLANES = 20

# Ignore points too close to the floor.
MIN_HEIGHT_ABOVE_FLOOR = 0.10

# Ignore points extremely high above the floor.
# This is deliberately generous for now.
MAX_HEIGHT_ABOVE_FLOOR = 3.5

# Two wall planes whose normals differ by less than this
# angle are considered parallel.
PARALLEL_ANGLE_DEGREES = 10.0

# Output directory.
OUTPUT_PREFIX = "wall_plane"


# ============================================================
# Utility functions
# ============================================================

def angle_between_normals(n1, n2):
    """
    Return the acute angle between two plane normals.

    Since n and -n represent the same plane orientation,
    we use the absolute value of their dot product.
    """

    dot = np.clip(
        abs(np.dot(n1, n2)),
        0.0,
        1.0,
    )

    return np.degrees(
        np.arccos(dot)
    )


def point_plane_distance(points, plane):
    """
    Calculate absolute point-to-plane distance.

    Plane:
        ax + by + cz + d = 0

    Assumes the normal is normalized.
    """

    a, b, c, d = plane

    return np.abs(
        points @ np.array([a, b, c]) + d
    )


def normalize_plane(plane):
    """
    Normalize a plane equation:

        ax + by + cz + d = 0

    so that [a,b,c] has unit length.
    """

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
    """
    Estimate the floor plane from floor_plane.ply.

    We don't rely on the point cloud file containing
    the original RANSAC plane equation. Instead, we
    fit a plane to the saved floor points.
    """

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

    # Make floor normal point approximately +Y.
    if plane[1] < 0:
        plane = -plane

    return plane, floor_cloud


def height_above_floor(points, floor_plane):
    """
    Signed distance above the floor.

    Since the floor normal points approximately +Y,
    positive signed distance corresponds to points
    above the floor.
    """

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
    # Load global point cloud
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
    # Load detected floor
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

    floor_plane, floor_cloud = (
        load_floor_plane(
            floor_path
        )
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
    # Remove floor
    # --------------------------------------------------------

    all_points = np.asarray(
        cloud.points
    )

    signed_height = (
        height_above_floor(
            all_points,
            floor_plane,
        )
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
    # Iterative RANSAC
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

        # ----------------------------------------------------
        # Ensure normal has consistent vertical orientation.
        #
        # We don't actually care whether a wall normal points
        # inward or outward at this stage.
        # ----------------------------------------------------

        if normal[1] < 0:
            normal = -normal
            plane = -plane

        # ----------------------------------------------------
        # Wall test
        #
        # A wall is vertical, so its normal should be
        # approximately perpendicular to the floor normal.
        #
        # Since floor normal ~= +Y:
        #
        #       wall normal ~= X/Z plane
        #
        # therefore |normal_y| should be small.
        # ----------------------------------------------------

        is_vertical = (
            abs(normal[1])
            <= MAX_NORMAL_Y
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
                    "points": len(inliers),
                    "centroid": centroid.copy(),
                    "inliers": inliers.copy(),
                }
            )

        # ----------------------------------------------------
        # Remove detected plane regardless of orientation.
        #
        # This allows us to discover additional surfaces.
        # ----------------------------------------------------

        remaining = (
            remaining.select_by_index(
                inliers,
                invert=True,
            )
        )

    # --------------------------------------------------------
    # No walls
    # --------------------------------------------------------

    if not candidates:

        print(
            "\nNo wall candidates found."
        )

        return

    # --------------------------------------------------------
    # Sort by number of points
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
            f"{candidate['plane'][0]:.6f}x + "
            f"{candidate['plane'][1]:.6f}y + "
            f"{candidate['plane'][2]:.6f}z + "
            f"{candidate['plane'][3]:.6f} = 0"
        )

    # --------------------------------------------------------
    # Group candidates by orientation
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
    # Save wall planes
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

    # --------------------------------------------------------
    # Re-extract each wall directly from the original
    # height-filtered cloud using its plane equation.
    # --------------------------------------------------------

    wall_points = np.asarray(
        cloud.points
    )[candidate_indices]

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
            f"\n  Saved: {output_path}"
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


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()
