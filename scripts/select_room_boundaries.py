import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

# Walls whose normals differ by <= this angle are considered
# to have the same orientation.
ORIENTATION_ANGLE_DEG = 10.0

# Near-coplanar planes are merged.
MERGE_ANGLE_DEG = 5.0
MERGE_DISTANCE_M = 0.15

# Only consider reasonably vertical portions of the scan.
MIN_HEIGHT_ABOVE_FLOOR = 0.10
MAX_HEIGHT_ABOVE_FLOOR = 3.5

# Distance from wall plane used to recover wall points.
WALL_DISTANCE_THRESHOLD = 0.05

# Minimum wall support after reconstruction.
MIN_WALL_POINTS = 5_000

# Minimum overlap of two opposite walls along their common
# wall direction.
MIN_EXTENT_OVERLAP = 0.20

# Reject extremely small/large room dimensions.
MIN_ROOM_DIMENSION = 1.0
MAX_ROOM_DIMENSION = 15.0

# Plot extent padding.
PLOT_PADDING = 0.5


# ============================================================
# Basic geometry
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

def normalize_plane(plane):
    """
    Normalize a plane equation:

        ax + by + cz + d = 0

    so that [a, b, c] has unit length.
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


def angle_between_normals(n1, n2):

    dot = np.clip(
        abs(
            np.dot(
                normalize(n1),
                normalize(n2),
            )
        ),
        0.0,
        1.0,
    )

    return np.degrees(
        np.arccos(dot)
    )


def plane_distance(
    plane_a,
    plane_b,
):
    """
    Approximate distance between two parallel planes.

    Assumes plane normals have already been normalized
    and aligned to the same direction.
    """

    n1 = normalize(
        plane_a[:3]
    )

    n2 = normalize(
        plane_b[:3]
    )

    d1 = plane_a[3]
    d2 = plane_b[3]

    if np.dot(n1, n2) < 0:

        d2 = -d2

    return abs(
        d1 - d2
    )


def signed_height(
    points,
    floor_plane,
):

    n = normalize(
        floor_plane[:3]
    )

    d = floor_plane[3]

    return (
        points @ n + d
    )


# ============================================================
# Floor coordinate system
# ============================================================

def choose_floor_axes(
    floor_normal,
    walls,
):

    floor_normal = normalize(
        floor_normal
    )

    # Find the first valid wall normal.
    reference_normal = None

    for wall in walls:

        n = normalize(
            np.asarray(
                wall["normal"],
                dtype=np.float64,
            )
        )

        # Project wall normal into floor plane.
        projected = (
            n
            - np.dot(
                n,
                floor_normal,
            )
            * floor_normal
        )

        if np.linalg.norm(
            projected
        ) > 1e-8:

            reference_normal = normalize(
                projected
            )

            break

    if reference_normal is None:

        raise RuntimeError(
            "Could not construct floor coordinate system."
        )

    # Wall direction.
    u = normalize(
        np.cross(
            floor_normal,
            reference_normal,
        )
    )

    # Second floor direction.
    v = normalize(
        np.cross(
            floor_normal,
            u,
        )
    )

    return u, v


def plane_to_2d_line(
    plane,
    origin,
    u,
    v,
):

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

    norm = np.hypot(
        A,
        B,
    )

    if norm < 1e-10:

        raise RuntimeError(
            "Wall plane cannot be projected to floor."
        )

    return np.array(
        [
            A / norm,
            B / norm,
            C / norm,
        ],
        dtype=np.float64,
    )


def project_points(
    points,
    origin,
    u,
    v,
):

    relative = (
        points - origin
    )

    return np.column_stack(
        (
            relative @ u,
            relative @ v,
        )
    )


# ============================================================
# Wall merging
# ============================================================

def merge_wall_planes(
    walls,
):

    merged = []

    used = set()

    for i, wall in enumerate(
        walls
    ):

        if i in used:
            continue

        current = [wall]

        used.add(i)

        changed = True

        while changed:

            changed = False

            representative = current[0]

            for j, other in enumerate(
                walls
            ):

                if j in used:
                    continue

                angle = angle_between_normals(
                    representative["normal"],
                    other["normal"],
                )

                if angle > MERGE_ANGLE_DEG:
                    continue

                distance = plane_distance(
                    representative["plane"],
                    other["plane"],
                )

                if distance <= MERGE_DISTANCE_M:

                    current.append(
                        other
                    )

                    used.add(j)

                    changed = True

        # ----------------------------------------------------
        # Weighted representative plane
        # ----------------------------------------------------

        total_points = sum(
            wall["points"]
            for wall in current
        )

        normal = np.zeros(
            3,
            dtype=np.float64,
        )

        d = 0.0

        for wall in current:

            weight = (
                wall["points"]
                / total_points
            )

            n = normalize(
                np.asarray(
                    wall["normal"]
                )
            )

            p = np.asarray(
                wall["plane"]
            )

            if np.dot(
                n,
                normal,
            ) < 0:

                n = -n
                p = -p

            normal += (
                weight * n
            )

            d += (
                weight * p[3]
            )

        normal = normalize(
            normal
        )

        plane = np.array(
            [
                normal[0],
                normal[1],
                normal[2],
                d,
            ],
            dtype=np.float64,
        )

        merged.append(
            {
                "source_wall_ids": [
                    wall["wall_id"]
                    for wall in current
                ],

                "plane": plane,

                "normal": normal,

                "points": int(
                    total_points
                ),

                "members": len(
                    current
                ),
            }
        )

    return merged


# ============================================================
# Wall extent calculation
# ============================================================

def calculate_wall_extent(
    wall,
    global_points,
    floor_plane,
    origin,
    u,
    v,
):

    plane = wall["plane"]

    distances = np.abs(
        global_points @ plane[:3]
        + plane[3]
    )

    heights = signed_height(
        global_points,
        floor_plane,
    )

    mask = (
        (distances <= WALL_DISTANCE_THRESHOLD)
        &
        (heights >= MIN_HEIGHT_ABOVE_FLOOR)
        &
        (heights <= MAX_HEIGHT_ABOVE_FLOOR)
    )

    points = global_points[
        mask
    ]

    if len(points) == 0:

        return None

    projected = project_points(
        points,
        origin,
        u,
        v,
    )

    u_values = projected[:, 0]
    v_values = projected[:, 1]

    # --------------------------------------------------------
    # Determine wall direction.
    #
    # A wall normal projected into the floor plane is
    # perpendicular to the wall direction.
    # --------------------------------------------------------

    wall_normal_floor = (
        wall["normal"]
        - np.dot(
            wall["normal"],
            normalize(floor_plane[:3]),
        )
        * normalize(floor_plane[:3])
    )

    wall_normal_floor = normalize(
        wall_normal_floor
    )

    wall_direction = normalize(
        np.cross(
            normalize(floor_plane[:3]),
            wall_normal_floor,
        )
    )

    # Coordinates along wall direction.
    wall_coordinates = (
        points - origin
    ) @ wall_direction

    wall_min = float(
        np.min(
            wall_coordinates
        )
    )

    wall_max = float(
        np.max(
            wall_coordinates
        )
    )

    extent = wall_max - wall_min

    return {
        "points": int(len(points)),

        "u_min": float(
            np.min(u_values)
        ),

        "u_max": float(
            np.max(u_values)
        ),

        "v_min": float(
            np.min(v_values)
        ),

        "v_max": float(
            np.max(v_values)
        ),

        "wall_min": wall_min,

        "wall_max": wall_max,

        "extent": float(
            extent
        ),

        "projected_points": projected,
    }


# ============================================================
# Wall orientation families
# ============================================================

def group_wall_families(
    walls,
):

    families = []

    for wall in walls:

        assigned = False

        for family in families:

            representative = family[0]

            angle = angle_between_normals(
                representative["normal"],
                wall["normal"],
            )

            if (
                angle
                <= ORIENTATION_ANGLE_DEG
            ):

                family.append(
                    wall
                )

                assigned = True

                break

        if not assigned:

            families.append(
                [wall]
            )

    return families


# ============================================================
# Opposite-wall candidate
# ============================================================

def calculate_overlap(
    wall_a,
    wall_b,
):
    """
    Calculate the overlap between two walls along
    their wall-direction coordinate.
    """

    extent_a = wall_a["extent_data"]
    extent_b = wall_b["extent_data"]

    start = max(
        extent_a["wall_min"],
        extent_b["wall_min"],
    )

    end = min(
        extent_a["wall_max"],
        extent_b["wall_max"],
    )

    overlap = max(
        0.0,
        end - start,
    )

    smaller_extent = min(
        extent_a["extent"],
        extent_b["extent"],
    )

    if smaller_extent <= 1e-8:
        return 0.0

    return overlap / smaller_extent

def line_separation(
    wall_a,
    wall_b,
):

    plane_a = wall_a["plane"]
    plane_b = wall_b["plane"]

    n_a = normalize(
        plane_a[:3]
    )

    n_b = normalize(
        plane_b[:3]
    )

    d_a = plane_a[3]
    d_b = plane_b[3]

    if np.dot(
        n_a,
        n_b,
    ) < 0:

        d_b = -d_b

    return abs(
        d_a - d_b
    )


def evaluate_wall_pair(
    wall_a,
    wall_b,
):
    """
    Evaluate whether two parallel wall surfaces can form
    opposite boundaries of a room.

    Wall extent information is stored inside extent_data.
    """

    separation = line_separation(
        wall_a,
        wall_b,
    )

    overlap = calculate_overlap(
        wall_a,
        wall_b,
    )

    extent_a = wall_a["extent_data"]
    extent_b = wall_b["extent_data"]

    min_extent = min(
        extent_a["extent"],
        extent_b["extent"],
    )

    max_extent = max(
        extent_a["extent"],
        extent_b["extent"],
    )

    # --------------------------------------------------------
    # Reject unreasonable room dimensions
    # --------------------------------------------------------

    if (
        separation
        < MIN_ROOM_DIMENSION
    ):
        return None

    if (
        separation
        > MAX_ROOM_DIMENSION
    ):
        return None

    # --------------------------------------------------------
    # The two walls should overlap along their length.
    # --------------------------------------------------------

    if (
        overlap
        < MIN_EXTENT_OVERLAP
    ):
        return None

    # --------------------------------------------------------
    # Score candidate pair.
    #
    # Strong candidates have:
    #   - lots of supporting points
    #   - long walls
    #   - high overlap
    #   - reasonable separation
    # --------------------------------------------------------

    support_score = np.log1p(
        wall_a["points"]
        + wall_b["points"]
    )

    extent_score = min(
        min_extent,
        10.0,
    )

    overlap_score = (
        2.0 * overlap
    )

    separation_score = min(
        separation,
        10.0,
    )

    score = (
        support_score
        + extent_score
        + overlap_score
        + 0.5 * separation_score
    )

    return {
        "wall_a": wall_a,
        "wall_b": wall_b,

        "separation": float(
            separation
        ),

        "overlap": float(
            overlap
        ),

        "min_extent": float(
            min_extent
        ),

        "max_extent": float(
            max_extent
        ),

        "score": float(
            score
        ),
    }


# ============================================================
# Main
# ============================================================

def main():

    if len(sys.argv) != 2:

        print(
            "Usage:\n"
            "uv run python scripts\\select_room_boundaries.py "
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

    source_pointcloud = Path(
        data["source_pointcloud"]
    )

    if not source_pointcloud.exists():

        # Handle paths relative to repo if needed.
        source_pointcloud = (
            Path.cwd()
            / data["source_pointcloud"]
        )

    if not source_pointcloud.exists():

        raise FileNotFoundError(
            "Source point cloud not found:\n"
            f"{source_pointcloud}"
        )

    floor_plane = normalize_plane(
        np.asarray(
            data["floor_plane"],
            dtype=np.float64,
        )
    )

    walls = data["walls"]

    # --------------------------------------------------------
    # Load point cloud
    # --------------------------------------------------------

    cloud = o3d.io.read_point_cloud(
        str(source_pointcloud)
    )

    if cloud.is_empty():

        raise RuntimeError(
            "Source point cloud is empty."
        )

    global_points = np.asarray(
        cloud.points
    )

    print("=" * 70)
    print("ROOM BOUNDARY SELECTION")
    print("=" * 70)

    print(
        f"\nGlobal points: "
        f"{len(global_points):,}"
    )

    print(
        f"Original wall planes: "
        f"{len(walls)}"
    )

    # --------------------------------------------------------
    # Floor coordinate system
    # --------------------------------------------------------

    floor_normal = normalize(
        floor_plane[:3]
    )

    u, v = choose_floor_axes(
        floor_normal,
        walls,
    )

    # Point on floor plane closest to origin.
    origin = (
        -floor_plane[3]
        * floor_normal
    )

    print(
        "\nFloor frame:"
    )

    print(
        f"  Origin: {origin}"
    )

    print(
        f"  U: {u}"
    )

    print(
        f"  V: {v}"
    )

    print(
        f"  N: {floor_normal}"
    )

    # --------------------------------------------------------
    # Merge coplanar fragments
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "MERGING WALL FRAGMENTS"
    )

    print(
        "=" * 70
    )

    merged_walls = merge_wall_planes(
        walls
    )

    print(
        f"\nMerged wall surfaces: "
        f"{len(merged_walls)}"
    )

    for i, wall in enumerate(
        merged_walls,
        start=1,
    ):

        print(
            f"\nSurface {i}"
        )

        print(
            f"  Source walls: "
            f"{wall['source_wall_ids']}"
        )

        print(
            f"  Points: "
            f"{wall['points']:,}"
        )

        print(
            f"  Normal: "
            f"{wall['normal']}"
        )

    # --------------------------------------------------------
    # Calculate wall extents
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "WALL EXTENTS"
    )

    print(
        "=" * 70
    )

    valid_walls = []

    for i, wall in enumerate(
        merged_walls,
        start=1,
    ):

        extent = calculate_wall_extent(
            wall,
            global_points,
            floor_plane,
            origin,
            u,
            v,
        )

        if extent is None:
            continue

        wall.update(
            {
                "wall_index": i,
                "extent_data": extent,
            }
        )

        if (
            extent["points"]
            >= MIN_WALL_POINTS
        ):

            valid_walls.append(
                wall
            )

        print(
            f"\nSurface {i}"
        )

        print(
            f"  Source walls: "
            f"{wall['source_wall_ids']}"
        )

        print(
            f"  Recovered points: "
            f"{extent['points']:,}"
        )

        print(
            f"  Wall extent: "
            f"{extent['extent']:.3f} m"
        )

        print(
            f"  U range: "
            f"{extent['u_min']:.3f} -> "
            f"{extent['u_max']:.3f}"
        )

        print(
            f"  V range: "
            f"{extent['v_min']:.3f} -> "
            f"{extent['v_max']:.3f}"
        )

    if len(valid_walls) < 4:

        raise RuntimeError(
            "\nNot enough valid wall surfaces "
            "to construct a room."
        )

    # --------------------------------------------------------
    # Wall families
    # --------------------------------------------------------

    families = group_wall_families(
        valid_walls
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

    for i, family in enumerate(
        families,
        start=1,
    ):

        print(
            f"\nFamily {i}"
        )

        for wall in family:

            print(
                f"  Surface {wall['wall_index']} "
                f"from walls "
                f"{wall['source_wall_ids']} "
                f"extent="
                f"{wall['extent_data']['extent']:.3f} m"
            )

    if len(families) < 2:

        raise RuntimeError(
            "Could not find two perpendicular wall families."
        )

    # --------------------------------------------------------
    # Candidate pairs
    # --------------------------------------------------------

    pair_candidates = []

    for family_index, family in enumerate(
        families
    ):

        if len(family) < 2:
            continue

        for i in range(
            len(family)
        ):

            for j in range(
                i + 1,
                len(family),
            ):

                result = evaluate_wall_pair(
                    family[i],
                    family[j],
                )

                if result is None:
                    continue

                result[
                    "family_index"
                ] = family_index

                pair_candidates.append(
                    result
                )

    pair_candidates.sort(
        key=lambda x: -x["score"]
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "OPPOSITE WALL CANDIDATES"
    )

    print(
        "=" * 70
    )

    if not pair_candidates:

        raise RuntimeError(
            "No valid opposite-wall pairs found."
        )

    for i, pair in enumerate(
        pair_candidates,
        start=1,
    ):

        a = pair["wall_a"]
        b = pair["wall_b"]

        print(
            f"\nCandidate Pair {i}"
        )

        print(
            f"  Family: "
            f"{pair['family_index'] + 1}"
        )

        print(
            f"  Surfaces: "
            f"{a['wall_index']} "
            f"<-> "
            f"{b['wall_index']}"
        )

        print(
            f"  Source walls: "
            f"{a['source_wall_ids']} "
            f"<-> "
            f"{b['source_wall_ids']}"
        )

        print(
            f"  Separation: "
            f"{pair['separation']:.3f} m"
        )

        print(
            f"  Extent overlap: "
            f"{pair['overlap']:.3f}"
        )

        print(
            f"  Minimum wall extent: "
            f"{pair['min_extent']:.3f} m"
        )

        print(
            f"  Score: "
            f"{pair['score']:.3f}"
        )

    # --------------------------------------------------------
    # Select one pair from each of the two strongest
    # families.
    #
    # First select the highest scoring pair in each family.
    # --------------------------------------------------------

    selected_pairs = []

    for family_index in range(
        len(families)
    ):

        family_candidates = [
            pair
            for pair in pair_candidates
            if pair["family_index"]
            == family_index
        ]

        if not family_candidates:
            continue

        selected_pairs.append(
            family_candidates[0]
        )

    if len(selected_pairs) < 2:

        raise RuntimeError(
            "Could not construct two room dimensions."
        )

    # --------------------------------------------------------
    # Choose the two most orthogonal families.
    # --------------------------------------------------------

    best_family_pair = None
    best_angle_error = float(
        "inf"
    )

    for i in range(
        len(selected_pairs)
    ):

        for j in range(
            i + 1,
            len(selected_pairs),
        ):

            pair_a = selected_pairs[i]
            pair_b = selected_pairs[j]

            normal_a = pair_a[
                "wall_a"
            ]["normal"]

            normal_b = pair_b[
                "wall_a"
            ]["normal"]

            angle = angle_between_normals(
                normal_a,
                normal_b,
            )

            error = abs(
                angle - 90.0
            )

            if error < best_angle_error:

                best_angle_error = error

                best_family_pair = (
                    pair_a,
                    pair_b,
                )

    if best_family_pair is None:

        raise RuntimeError(
            "Could not determine perpendicular wall families."
        )

    pair_a, pair_b = (
        best_family_pair
    )

    # --------------------------------------------------------
    # Room dimensions
    # --------------------------------------------------------

    dimension_a = (
        pair_a["separation"]
    )

    dimension_b = (
        pair_b["separation"]
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "SELECTED ROOM BOUNDARIES"
    )

    print(
        "=" * 70
    )

    print(
        f"\nDimension A:"
        f"\n  Family: "
        f"{pair_a['family_index'] + 1}"
        f"\n  Surfaces: "
        f"{pair_a['wall_a']['wall_index']} "
        f"<-> "
        f"{pair_a['wall_b']['wall_index']}"
        f"\n  Separation: "
        f"{dimension_a:.3f} m"
    )

    print(
        f"\nDimension B:"
        f"\n  Family: "
        f"{pair_b['family_index'] + 1}"
        f"\n  Surfaces: "
        f"{pair_b['wall_a']['wall_index']} "
        f"<-> "
        f"{pair_b['wall_b']['wall_index']}"
        f"\n  Separation: "
        f"{dimension_b:.3f} m"
    )

    print(
        f"\nApproximate room area: "
        f"{dimension_a * dimension_b:.3f} m²"
    )

    # --------------------------------------------------------
    # Determine four boundary lines
    # --------------------------------------------------------

    selected_walls = [
        pair_a["wall_a"],
        pair_a["wall_b"],
        pair_b["wall_a"],
        pair_b["wall_b"],
    ]

    for wall in selected_walls:

        wall["line_2d"] = (
            plane_to_2d_line(
                wall["plane"],
                origin,
                u,
                v,
            )
        )

    # --------------------------------------------------------
    # Intersections
    # --------------------------------------------------------

    def intersect_lines(
        line1,
        line2,
    ):

        A1, B1, C1 = line1
        A2, B2, C2 = line2

        determinant = (
            A1 * B2
            - A2 * B1
        )

        if abs(
            determinant
        ) < 1e-10:

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

    corners = []

    for wall_a in (
        pair_a["wall_a"],
        pair_a["wall_b"],
    ):

        for wall_b in (
            pair_b["wall_a"],
            pair_b["wall_b"],
        ):

            point = intersect_lines(
                wall_a["line_2d"],
                wall_b["line_2d"],
            )

            if point is None:
                continue

            corners.append(
                {
                    "wall_a": wall_a[
                        "wall_index"
                    ],
                    "wall_b": wall_b[
                        "wall_index"
                    ],
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
        "ROOM CORNERS"
    )

    print(
        "=" * 70
    )

    for corner in corners:

        print(
            f"\nWalls "
            f"{corner['wall_a']} x "
            f"{corner['wall_b']}"
        )

        print(
            f"  "
            f"({corner['point'][0]:.3f}, "
            f"{corner['point'][1]:.3f})"
        )

    # --------------------------------------------------------
    # Sort corners around centroid
    # --------------------------------------------------------

    corner_points = np.asarray(
        [
            corner["point"]
            for corner in corners
        ]
    )

    if len(
        corner_points
    ) != 4:

        raise RuntimeError(
            "Expected four room corners."
        )

    center = (
        corner_points.mean(
            axis=0
        )
    )

    angles = np.arctan2(
        corner_points[:, 1]
        - center[1],
        corner_points[:, 0]
        - center[0],
    )

    order = np.argsort(
        angles
    )

    polygon = (
        corner_points[order]
    )

    # --------------------------------------------------------
    # Polygon area
    # --------------------------------------------------------

    x = polygon[:, 0]
    y = polygon[:, 1]

    polygon_area = 0.5 * abs(
        np.sum(
            x * np.roll(y, -1)
            - y * np.roll(x, -1)
        )
    )

    # --------------------------------------------------------
    # Save room geometry
    # --------------------------------------------------------

    output_json = (
        walls_path.parent
        / "room_boundary.json"
    )

    serializable_walls = []

    for wall in selected_walls:

        serializable_walls.append(
            {
                "surface_id": wall[
                    "wall_index"
                ],

                "source_wall_ids":
                    wall[
                        "source_wall_ids"
                    ],

                "plane": [
                    float(x)
                    for x in wall[
                        "plane"
                    ]
                ],

                "normal": [
                    float(x)
                    for x in wall[
                        "normal"
                    ]
                ],

                "points": int(
                    wall[
                        "points"
                    ]
                ),

                "extent_m": float(
                    wall[
                        "extent_data"
                    ]["extent"]
                ),

                "u_min": float(
                    wall[
                        "extent_data"
                    ]["u_min"]
                ),

                "u_max": float(
                    wall[
                        "extent_data"
                    ]["u_max"]
                ),

                "v_min": float(
                    wall[
                        "extent_data"
                    ]["v_min"]
                ),

                "v_max": float(
                    wall[
                        "extent_data"
                    ]["v_max"]
                ),
            }
        )

    room_data = {

        "room_type": "single_room",

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

        "dimensions_m": {

            "dimension_a":
                float(dimension_a),

            "dimension_b":
                float(dimension_b),
        },

        "area_m2": float(
            polygon_area
        ),

        "corners": [
            [
                float(point[0]),
                float(point[1]),
            ]
            for point in polygon
        ],

        "selected_wall_pairs": [

            {
                "family":
                    int(
                        pair_a[
                            "family_index"
                        ]
                    ),

                "walls": [
                    int(
                        pair_a[
                            "wall_a"
                        ]["wall_index"]
                    ),
                    int(
                        pair_a[
                            "wall_b"
                        ]["wall_index"]
                    ),
                ],

                "separation_m":
                    float(
                        pair_a[
                            "separation"
                        ]
                    ),

                "overlap":
                    float(
                        pair_a[
                            "overlap"
                        ]
                    ),

                "score":
                    float(
                        pair_a[
                            "score"
                        ]
                    ),
            },

            {
                "family":
                    int(
                        pair_b[
                            "family_index"
                        ]
                    ),

                "walls": [
                    int(
                        pair_b[
                            "wall_a"
                        ]["wall_index"]
                    ),
                    int(
                        pair_b[
                            "wall_b"
                        ]["wall_index"]
                    ),
                ],

                "separation_m":
                    float(
                        pair_b[
                            "separation"
                        ]
                    ),

                "overlap":
                    float(
                        pair_b[
                            "overlap"
                        ]
                    ),

                "score":
                    float(
                        pair_b[
                            "score"
                        ]
                    ),
            },
        ],

        "walls": serializable_walls,

        "corners": [
            {
                "u": float(
                    point[0]
                ),
                "v": float(
                    point[1]
                ),
            }
            for point in polygon
        ],
    }

    with open(
        output_json,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            room_data,
            f,
            indent=2,
        )

    print(
        "\nSaved:"
        f"\n{output_json}"
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 8)
    )

    # Draw polygon.
    closed_polygon = np.vstack(
        [
            polygon,
            polygon[0],
        ]
    )

    plt.plot(
        closed_polygon[:, 0],
        closed_polygon[:, 1],
        linewidth=3,
        marker="o",
    )

    # Label corners.
    for i, point in enumerate(
        polygon,
        start=1,
    ):

        plt.annotate(
            f"C{i}",
            (
                point[0],
                point[1],
            ),
            xytext=(5, 5),
            textcoords="offset points",
        )

    # Draw dimensions.
    for i in range(4):

        p1 = polygon[i]
        p2 = polygon[
            (i + 1) % 4
        ]

        midpoint = (
            p1 + p2
        ) / 2.0

        length = np.linalg.norm(
            p2 - p1
        )

        plt.annotate(
            f"{length:.2f} m",
            midpoint,
            ha="center",
            va="center",
        )

    plt.xlabel(
        "Floor U (m)"
    )

    plt.ylabel(
        "Floor V (m)"
    )

    plt.title(
        "Selected Room Boundary"
    )

    plt.axis(
        "equal"
    )

    plt.grid(
        True
    )

    plot_path = (
        walls_path.parent
        / "room_boundary.png"
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
        "ROOM BOUNDARY SELECTION COMPLETE"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()
