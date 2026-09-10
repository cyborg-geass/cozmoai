import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

MIN_WALL_POINTS = 8_000

# Walls with approximately the same normal orientation
# belong to the same direction family.
PARALLEL_ANGLE_DEGREES = 10.0

# Two parallel wall planes closer than this are treated
# as potentially representing the same physical boundary.
MIN_WALL_SEPARATION = 0.20

# Maximum reasonable wall separation for a room boundary.
MAX_WALL_SEPARATION = 15.0


# ============================================================
# Geometry helpers
# ============================================================

def normalize(v):

    v = np.asarray(
        v,
        dtype=np.float64,
    )

    norm = np.linalg.norm(v)

    if norm < 1e-12:

        raise ValueError(
            "Cannot normalize zero vector."
        )

    return v / norm


def angle_between_normals(n1, n2):

    dot = np.clip(
        abs(np.dot(n1, n2)),
        0.0,
        1.0,
    )

    return np.degrees(
        np.arccos(dot)
    )


def choose_floor_axes(
    floor_normal,
    wall_normals,
):
    """
    Construct a 2D coordinate system on the floor.

    u and v lie in the floor plane.

    We use one dominant wall normal to establish
    the first horizontal direction.

    If n_wall is a wall normal, the wall itself lies
    perpendicular to n_wall. We therefore choose:

        u = wall direction

    and:

        v = floor_normal x u

    """

    floor_normal = normalize(
        floor_normal
    )

    # Pick the first wall normal.
    reference_normal = normalize(
        wall_normals[0]
    )

    # Remove any component along floor normal.
    reference_normal = (
        reference_normal
        - np.dot(
            reference_normal,
            floor_normal,
        )
        * floor_normal
    )

    reference_normal = normalize(
        reference_normal
    )

    # Wall direction.
    u = np.cross(
        floor_normal,
        reference_normal,
    )

    u = normalize(u)

    # Second floor direction.
    v = np.cross(
        floor_normal,
        u,
    )

    v = normalize(v)

    return u, v


def project_point_to_floor(
    point,
    origin,
    u,
    v,
):
    """
    Project a 3D point onto the 2D floor frame.
    """

    relative = (
        np.asarray(point)
        - origin
    )

    return np.array(
        [
            np.dot(relative, u),
            np.dot(relative, v),
        ],
        dtype=np.float64,
    )


def plane_to_2d_line(
    plane,
    origin,
    u,
    v,
):
    """
    Convert a vertical 3D plane into a 2D line
    on the floor coordinate system.

    3D plane:

        n.x + d = 0

    For a point on the floor:

        p = origin + u*x + v*y

    Therefore:

        n.(origin + u*x + v*y) + d = 0

    giving:

        A*x + B*y + C = 0

    where:

        A = n.u
        B = n.v
        C = n.origin + d
    """

    n = normalize(
        plane[:3]
    )

    d = plane[3]

    A = np.dot(
        n,
        u,
    )

    B = np.dot(
        n,
        v,
    )

    C = (
        np.dot(
            n,
            origin,
        )
        + d
    )

    norm = np.sqrt(
        A * A
        + B * B
    )

    if norm < 1e-10:

        raise ValueError(
            "Plane does not intersect floor coordinate frame."
        )

    A /= norm
    B /= norm
    C /= norm

    return np.array(
        [A, B, C],
        dtype=np.float64,
    )


def line_intersection(
    line1,
    line2,
):
    """
    Solve:

        A1*x + B1*y + C1 = 0
        A2*x + B2*y + C2 = 0

    Returns None for parallel lines.
    """

    A1, B1, C1 = line1
    A2, B2, C2 = line2

    determinant = (
        A1 * B2
        - A2 * B1
    )

    if abs(determinant) < 1e-10:

        return None

    x = (
        B1 * C2
        - B2 * C1
    ) / determinant

    y = (
        C1 * A2
        - C2 * A1
    ) / determinant

    return np.array(
        [x, y],
        dtype=np.float64,
    )


def line_distance(
    line1,
    line2,
):
    """
    Distance between two parallel normalized lines.
    """

    A1, B1, C1 = line1
    A2, B2, C2 = line2

    # Ensure same normal direction.
    if (
        A1 * A2
        + B1 * B2
        < 0
    ):

        C2 = -C2

    return abs(
        C1 - C2
    )


def group_parallel_walls(
    walls,
):
    """
    Group walls according to their 2D orientation.
    """

    groups = []

    for wall in walls:

        assigned = False

        for group in groups:

            representative = group[0]

            angle = angle_between_normals(
                representative["normal"],
                wall["normal"],
            )

            if (
                angle
                <= PARALLEL_ANGLE_DEGREES
            ):

                group.append(
                    wall
                )

                assigned = True

                break

        if not assigned:

            groups.append(
                [wall]
            )

    return groups


# ============================================================
# Main
# ============================================================

def main():

    if len(sys.argv) != 2:

        print(
            "Usage:\n"
            "uv run python scripts\\build_room_geometry.py "
            "outputs\\single_room\\walls.json"
        )

        sys.exit(1)

    walls_path = Path(
        sys.argv[1]
    )

    if not walls_path.exists():

        raise FileNotFoundError(
            f"walls.json not found: "
            f"{walls_path}"
        )

    # --------------------------------------------------------
    # Load metadata
    # --------------------------------------------------------

    with open(
        walls_path,
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(f)

    floor_plane = np.asarray(
        data["floor_plane"],
        dtype=np.float64,
    )

    walls = data["walls"]

    if len(walls) < 2:

        raise RuntimeError(
            "Need at least two wall planes."
        )

    # --------------------------------------------------------
    # Filter
    # --------------------------------------------------------

    walls = [
        wall
        for wall in walls
        if wall["points"]
        >= MIN_WALL_POINTS
    ]

    print("=" * 70)
    print("ROOM GEOMETRY")
    print("=" * 70)

    print(
        f"\nWall planes: "
        f"{len(walls)}"
    )

    # --------------------------------------------------------
    # Floor coordinate system
    # --------------------------------------------------------

    floor_normal = normalize(
        floor_plane[:3]
    )

    wall_normals = [
        np.asarray(
            wall["normal"],
            dtype=np.float64,
        )
        for wall in walls
    ]

    u, v = choose_floor_axes(
        floor_normal,
        wall_normals,
    )

    # Use floor centroid as origin.
    #
    # Since walls.json doesn't currently contain the
    # floor centroid, use the plane closest to the
    # global origin as the coordinate origin.
    #
    # A point on the plane is:
    #
    #       p = -d*n
    #
    origin = (
        -floor_plane[3]
        * floor_normal
    )

    print(
        "\nFloor coordinate system:"
    )

    print(
        f"  Origin: "
        f"{origin}"
    )

    print(
        f"  U axis: "
        f"{u}"
    )

    print(
        f"  V axis: "
        f"{v}"
    )

    print(
        f"  Floor normal: "
        f"{floor_normal}"
    )

    # --------------------------------------------------------
    # Convert wall planes to 2D lines
    # --------------------------------------------------------

    for wall in walls:

        plane = np.asarray(
            wall["plane"],
            dtype=np.float64,
        )

        line = plane_to_2d_line(
            plane,
            origin,
            u,
            v,
        )

        wall["line_2d"] = [
            float(x)
            for x in line
        ]

    # --------------------------------------------------------
    # Group parallel walls
    # --------------------------------------------------------

    groups = group_parallel_walls(
        walls
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "WALL FAMILIES"
    )

    print(
        "=" * 70
    )

    for i, group in enumerate(
        groups,
        start=1,
    ):

        print(
            f"\nFamily {i}"
        )

        print(
            f"  Members: "
            f"{len(group)}"
        )

        for wall in group:

            print(
                f"    Wall {wall['wall_id']}: "
                f"{wall['points']:,} points"
            )

    # --------------------------------------------------------
    # Find opposite walls within each family
    # --------------------------------------------------------

    boundary_pairs = []

    for group_index, group in enumerate(
        groups
    ):

        if len(group) < 2:
            continue

        for i in range(
            len(group)
        ):

            for j in range(
                i + 1,
                len(group),
            ):

                line1 = np.asarray(
                    group[i]["line_2d"]
                )

                line2 = np.asarray(
                    group[j]["line_2d"]
                )

                distance = line_distance(
                    line1,
                    line2,
                )

                if (
                    MIN_WALL_SEPARATION
                    <= distance
                    <= MAX_WALL_SEPARATION
                ):

                    boundary_pairs.append(
                        {
                            "family": group_index,
                            "wall_a": group[i]["wall_id"],
                            "wall_b": group[j]["wall_id"],
                            "distance": float(distance),
                        }
                    )

    print(
        "\n" + "=" * 70
    )

    print(
        "PARALLEL WALL PAIRS"
    )

    print(
        "=" * 70
    )

    for pair in boundary_pairs:

        print(
            f"\nFamily {pair['family'] + 1}"
        )

        print(
            f"  Walls: "
            f"{pair['wall_a']} "
            f"<-> "
            f"{pair['wall_b']}"
        )

        print(
            f"  Separation: "
            f"{pair['distance']:.3f} m"
        )

    # --------------------------------------------------------
    # Build intersections between different wall families
    # --------------------------------------------------------

    intersections = []

    for i in range(
        len(groups)
    ):

        for j in range(
            i + 1,
            len(groups),
        ):

            group_a = groups[i]
            group_b = groups[j]

            for wall_a in group_a:

                for wall_b in group_b:

                    line_a = np.asarray(
                        wall_a["line_2d"]
                    )

                    line_b = np.asarray(
                        wall_b["line_2d"]
                    )

                    point = line_intersection(
                        line_a,
                        line_b,
                    )

                    if point is None:
                        continue

                    intersections.append(
                        {
                            "wall_a": wall_a["wall_id"],
                            "wall_b": wall_b["wall_id"],
                            "point": [
                                float(point[0]),
                                float(point[1]),
                            ],
                        }
                    )

    print(
        "\n" + "=" * 70
    )

    print(
        "WALL INTERSECTIONS"
    )

    print(
        "=" * 70
    )

    for item in intersections:

        print(
            f"\nWalls "
            f"{item['wall_a']} "
            f"x "
            f"{item['wall_b']}"
        )

        print(
            f"  Point: "
            f"{item['point']}"
        )

    # --------------------------------------------------------
    # Save room geometry
    # --------------------------------------------------------

    output_json = (
        walls_path.parent
        / "room_geometry.json"
    )

    output_data = {

        "floor_plane": [
            float(x)
            for x in floor_plane
        ],

        "floor_coordinate_system": {

            "origin": [
                float(x)
                for x in origin
            ],

            "u_axis": [
                float(x)
                for x in u
            ],

            "v_axis": [
                float(x)
                for x in v
            ],

            "floor_normal": [
                float(x)
                for x in floor_normal
            ],
        },

        "walls": walls,

        "wall_families": [
            [
                wall["wall_id"]
                for wall in group
            ]
            for group in groups
        ],

        "parallel_wall_pairs":
            boundary_pairs,

        "intersections":
            intersections,
    }

    with open(
        output_json,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output_data,
            f,
            indent=2,
        )

    print(
        "\nSaved:"
        f"\n{output_json}"
    )

    # --------------------------------------------------------
    # Plot 2D wall geometry
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 8)
    )

    # Draw wall lines.
    line_extent = 15.0

    for wall in walls:

        A, B, C = wall[
            "line_2d"
        ]

        wall_id = wall[
            "wall_id"
        ]

        if abs(B) > abs(A):

            x = np.linspace(
                -line_extent,
                line_extent,
                500,
            )

            y = (
                -A * x - C
            ) / B

        else:

            y = np.linspace(
                -line_extent,
                line_extent,
                500,
            )

            x = (
                -B * y - C
            ) / A

        plt.plot(
            x,
            y,
            linewidth=2,
            label=f"Wall {wall_id}",
        )

    # Draw intersections.
    for item in intersections:

        x, y = item[
            "point"
        ]

        plt.scatter(
            x,
            y,
            s=40,
        )

    plt.xlabel(
        "Floor U (m)"
    )

    plt.ylabel(
        "Floor V (m)"
    )

    plt.title(
        "Detected Wall Geometry"
    )

    plt.axis(
        "equal"
    )

    plt.grid(
        True
    )

    plt.legend()

    plot_path = (
        walls_path.parent
        / "room_geometry.png"
    )

    plt.savefig(
        plot_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"\nSaved plot:"
        f"\n{plot_path}"
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "ROOM GEOMETRY COMPLETE"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()
