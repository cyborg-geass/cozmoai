import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import open3d as o3d
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

ORIENTATION_ANGLE_DEG = 10.0

MIN_ROOM_DIMENSION = 1.5
MAX_ROOM_DIMENSION = 12.0

MIN_TRAJECTORY_INSIDE_RATIO = 0.70
TRAJECTORY_MARGIN = 0.15

MIN_WALL_POINTS = 5_000

MIN_WALL_EXTENT = 1.5

WALL_DISTANCE_THRESHOLD = 0.05

MIN_HEIGHT_ABOVE_FLOOR = 0.10
MAX_HEIGHT_ABOVE_FLOOR = 3.5


# ============================================================
# Geometry utilities
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

    plane = np.asarray(
        plane,
        dtype=np.float64,
    )

    norm = np.linalg.norm(
        plane[:3]
    )

    if norm < 1e-12:
        raise ValueError(
            "Invalid plane."
        )

    return plane / norm


def angle_between_normals(
    n1,
    n2,
):

    n1 = normalize(n1)
    n2 = normalize(n2)

    dot = np.clip(
        abs(
            np.dot(
                n1,
                n2,
            )
        ),
        0.0,
        1.0,
    )

    return np.degrees(
        np.arccos(dot)
    )


def point_plane_signed_distance(
    points,
    plane,
):

    n = plane[:3]
    d = plane[3]

    return (
        points @ n + d
    )


def point_plane_distance(
    points,
    plane,
):

    return np.abs(
        point_plane_signed_distance(
            points,
            plane,
        )
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

    reference = None

    for wall in walls:

        n = normalize(
            np.asarray(
                wall["normal"],
                dtype=np.float64,
            )
        )

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

            reference = normalize(
                projected
            )

            break

    if reference is None:

        raise RuntimeError(
            "Could not construct floor axes."
        )

    u = normalize(
        np.cross(
            floor_normal,
            reference,
        )
    )

    v = normalize(
        np.cross(
            floor_normal,
            u,
        )
    )

    return u, v


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
        [
            relative @ u,
            relative @ v,
        ]
    )


# ============================================================
# Wall → floor line
# ============================================================

def plane_to_floor_line(
    plane,
    origin,
    u,
    v,
):

    plane = normalize_plane(
        plane
    )

    n = plane[:3]
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


def line_value(
    line,
    points,
):

    A, B, C = line

    return (
        A * points[:, 0]
        + B * points[:, 1]
        + C
    )


def intersect_lines(
    line_a,
    line_b,
):

    A1, B1, C1 = line_a
    A2, B2, C2 = line_b

    det = (
        A1 * B2
        - A2 * B1
    )

    if abs(det) < 1e-10:

        return None

    x = (
        B1 * C2
        - B2 * C1
    ) / det

    y = (
        C1 * A2
        - C2 * A1
    ) / det

    return np.array(
        [
            x,
            y,
        ],
        dtype=np.float64,
    )


# ============================================================
# Load trajectory
# ============================================================

def load_trajectory(
    capture_dir,
):

    odometry_path = (
        capture_dir
        / "odometry.csv"
    )

    if not odometry_path.exists():

        raise FileNotFoundError(
            f"odometry.csv not found:\n"
            f"{odometry_path}"
        )

    odom = pd.read_csv(
        odometry_path
    )

    odom.columns = [
        column.strip()
        for column in odom.columns
    ]

    required = [
        "x",
        "y",
        "z",
    ]

    missing = [
        column
        for column in required
        if column not in odom.columns
    ]

    if missing:

        raise RuntimeError(
            "Missing odometry columns: "
            f"{missing}"
        )

    return odom[
        [
            "x",
            "y",
            "z",
        ]
    ].to_numpy(
        dtype=np.float64
    )


# ============================================================
# Reconstruct wall extent
# ============================================================

def reconstruct_wall_extent(
    wall,
    global_points,
    floor_plane,
    origin,
    u,
    v,
):

    plane = normalize_plane(
        np.asarray(
            wall["plane"],
            dtype=np.float64,
        )
    )

    distances = point_plane_distance(
        global_points,
        plane,
    )

    heights = point_plane_signed_distance(
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

    if len(points) < MIN_WALL_POINTS:

        return None

    # --------------------------------------------------------
    # Wall direction.
    # --------------------------------------------------------

    floor_normal = normalize(
        floor_plane[:3]
    )

    wall_normal = plane[:3]

    projected_normal = (
        wall_normal
        - np.dot(
            wall_normal,
            floor_normal,
        )
        * floor_normal
    )

    if np.linalg.norm(
        projected_normal
    ) < 1e-8:

        return None

    projected_normal = normalize(
        projected_normal
    )

    wall_direction = normalize(
        np.cross(
            floor_normal,
            projected_normal,
        )
    )

    # --------------------------------------------------------
    # Coordinates along wall.
    # --------------------------------------------------------

    relative = (
        points - origin
    )

    coordinate = (
        relative
        @ wall_direction
    )

    wall_min = float(
        coordinate.min()
    )

    wall_max = float(
        coordinate.max()
    )

    extent = (
        wall_max
        - wall_min
    )

    projected = np.column_stack(
        [
            relative @ u,
            relative @ v,
        ]
    )

    return {
        "points": int(
            len(points)
        ),

        "wall_min": wall_min,

        "wall_max": wall_max,

        "extent": float(
            extent
        ),

        "u_min": float(
            projected[:, 0].min()
        ),

        "u_max": float(
            projected[:, 0].max()
        ),

        "v_min": float(
            projected[:, 1].min()
        ),

        "v_max": float(
            projected[:, 1].max()
        ),
    }


# ============================================================
# Wall families
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
# Pair geometry
# ============================================================

def pair_separation(
    wall_a,
    wall_b,
):

    plane_a = normalize_plane(
        wall_a["plane"]
    )

    plane_b = normalize_plane(
        wall_b["plane"]
    )

    n_a = plane_a[:3]
    d_a = plane_a[3]

    n_b = plane_b[:3]
    d_b = plane_b[3]

    if np.dot(
        n_a,
        n_b,
    ) < 0:

        d_b = -d_b

    return abs(
        d_a - d_b
    )


def pair_overlap(
    wall_a,
    wall_b,
):

    start = max(
        wall_a["extent_data"]["wall_min"],
        wall_b["extent_data"]["wall_min"],
    )

    end = min(
        wall_a["extent_data"]["wall_max"],
        wall_b["extent_data"]["wall_max"],
    )

    overlap = max(
        0.0,
        end - start,
    )

    smaller = min(
        wall_a["extent_data"]["extent"],
        wall_b["extent_data"]["extent"],
    )

    if smaller <= 1e-8:

        return 0.0

    return (
        overlap / smaller
    )


# ============================================================
# Candidate room
# ============================================================

def build_candidate(
    pair_a,
    pair_b,
    trajectory_2d,
):

    wall_a1, wall_a2 = pair_a
    wall_b1, wall_b2 = pair_b

    dimension_a = pair_separation(
        wall_a1,
        wall_a2,
    )

    dimension_b = pair_separation(
        wall_b1,
        wall_b2,
    )

    # --------------------------------------------------------
    # Basic dimension sanity checks
    # --------------------------------------------------------

    if (
        dimension_a < MIN_ROOM_DIMENSION
        or dimension_b < MIN_ROOM_DIMENSION
    ):
        return None

    if (
        dimension_a > MAX_ROOM_DIMENSION
        or dimension_b > MAX_ROOM_DIMENSION
    ):
        return None

    # --------------------------------------------------------
    # Wall extent overlap
    # --------------------------------------------------------

    overlap_a = pair_overlap(
        wall_a1,
        wall_a2,
    )

    overlap_b = pair_overlap(
        wall_b1,
        wall_b2,
    )

    if (
        overlap_a < 0.30
        or overlap_b < 0.30
    ):
        return None

    # --------------------------------------------------------
    # Four wall boundary lines
    # --------------------------------------------------------

    lines = [
        wall_a1["line_2d"],
        wall_a2["line_2d"],
        wall_b1["line_2d"],
        wall_b2["line_2d"],
    ]

    # --------------------------------------------------------
    # Compute the four corners
    # --------------------------------------------------------

    corners = []

    for line_a in (
        wall_a1["line_2d"],
        wall_a2["line_2d"],
    ):

        for line_b in (
            wall_b1["line_2d"],
            wall_b2["line_2d"],
        ):

            point = intersect_lines(
                line_a,
                line_b,
            )

            if point is None:
                return None

            corners.append(
                point
            )

    corners = np.asarray(
        corners,
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # Order corners around their centroid
    # --------------------------------------------------------

    center = corners.mean(
        axis=0
    )

    angles = np.arctan2(
        corners[:, 1] - center[1],
        corners[:, 0] - center[0],
    )

    order = np.argsort(
        angles
    )

    polygon = corners[
        order
    ]

    # --------------------------------------------------------
    # Determine which side of each wall is interior
    # --------------------------------------------------------

    polygon_center = polygon.mean(
        axis=0
    )

    trajectory_inside = np.ones(
        len(trajectory_2d),
        dtype=bool,
    )

    for line in lines:

        center_value = line_value(
            line,
            polygon_center.reshape(
                1,
                2,
            ),
        )[0]

        trajectory_values = line_value(
            line,
            trajectory_2d,
        )

        if center_value >= 0:

            trajectory_inside &= (
                trajectory_values
                >= -TRAJECTORY_MARGIN
            )

        else:

            trajectory_inside &= (
                trajectory_values
                <= TRAJECTORY_MARGIN
            )

    # --------------------------------------------------------
    # Trajectory containment ratio
    # --------------------------------------------------------

    if len(trajectory_2d) == 0:
        return None

    trajectory_ratio = float(
        trajectory_inside.mean()
    )

    if (
        trajectory_ratio
        < MIN_TRAJECTORY_INSIDE_RATIO
    ):
        return None

    # --------------------------------------------------------
    # Polygon area
    # --------------------------------------------------------

    x = polygon[:, 0]
    y = polygon[:, 1]

    area = 0.5 * abs(
        np.sum(
            x * np.roll(
                y,
                -1,
            )
            -
            y * np.roll(
                x,
                -1,
            )
        )
    )

    # --------------------------------------------------------
    # Edge lengths
    # --------------------------------------------------------

    edge_lengths = []

    for i in range(4):

        p1 = polygon[i]

        p2 = polygon[
            (i + 1) % 4
        ]

        edge_lengths.append(
            float(
                np.linalg.norm(
                    p2 - p1
                )
            )
        )

    # --------------------------------------------------------
    # Wall support
    #
    # IMPORTANT:
    # Reconstructed wall support is stored in:
    #
    #     wall["extent_data"]["points"]
    #
    # not wall["points"].
    # --------------------------------------------------------

    selected_walls = [
        wall_a1,
        wall_a2,
        wall_b1,
        wall_b2,
    ]

    support = sum(
        wall[
            "extent_data"
        ][
            "points"
        ]
        for wall in selected_walls
    )

    # --------------------------------------------------------
    # Score
    #
    # Trajectory containment is deliberately dominant.
    # --------------------------------------------------------

    support_score = np.log1p(
        support
    )

    overlap_score = (
        overlap_a
        + overlap_b
    )

    score = (
        100.0
        * trajectory_ratio
        + 5.0
        * support_score
        + 10.0
        * overlap_score
    )

    return {
        "pair_a": pair_a,
        "pair_b": pair_b,

        "dimension_a":
            float(
                dimension_a
            ),

        "dimension_b":
            float(
                dimension_b
            ),

        "area":
            float(
                area
            ),

        "corners":
            polygon,

        "edge_lengths":
            edge_lengths,

        "trajectory_inside_ratio":
            trajectory_ratio,

        "support_points":
            int(
                support
            ),

        "score":
            float(
                score
            ),
    }

# ============================================================
# Main
# ============================================================

def main():

    if len(sys.argv) != 3:

        print(
            "Usage:\n"
            "uv run python scripts\\select_room_envelope.py "
            "outputs\\single_room\\walls.json "
            "..\\cozmo-dataset\\raw_dataset\\single_room\\c00a170fe1"
        )

        sys.exit(1)

    walls_path = Path(
        sys.argv[1]
    )

    capture_dir = Path(
        sys.argv[2]
    )

    # --------------------------------------------------------
    # Validate paths.
    # --------------------------------------------------------

    if not walls_path.exists():

        raise FileNotFoundError(
            f"walls.json not found:\n"
            f"{walls_path}"
        )

    if not capture_dir.exists():

        raise FileNotFoundError(
            f"Capture directory not found:\n"
            f"{capture_dir}"
        )

    # --------------------------------------------------------
    # Load walls.
    # --------------------------------------------------------

    with open(
        walls_path,
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(f)

    floor_plane = normalize_plane(
        data["floor_plane"]
    )

    raw_walls = data[
        "walls"
    ]

    print("=" * 70)
    print("ROOM ENVELOPE SELECTION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load point cloud.
    # --------------------------------------------------------

    pointcloud_path = Path(
        data.get(
            "source_pointcloud",
            "outputs/single_room/"
            "pointcloud_production.ply",
        )
    )

    if not pointcloud_path.exists():

        pointcloud_path = (
            Path.cwd()
            / pointcloud_path
        )

    if not pointcloud_path.exists():

        raise FileNotFoundError(
            "Could not locate source point cloud:\n"
            f"{pointcloud_path}"
        )

    cloud = o3d.io.read_point_cloud(
        str(pointcloud_path)
    )

    if cloud.is_empty():

        raise RuntimeError(
            "Point cloud is empty."
        )

    points = np.asarray(
        cloud.points
    )

    print(
        f"\nGlobal points: "
        f"{len(points):,}"
    )

    print(
        f"Original wall planes: "
        f"{len(raw_walls)}"
    )

    # --------------------------------------------------------
    # Reconstruct wall objects.
    # --------------------------------------------------------

    walls = []

    for index, raw_wall in enumerate(
        raw_walls,
        start=1,
    ):

        wall = {
            "wall_id": index,

            "source_wall_ids":
                raw_wall.get(
                    "source_wall_ids",
                    [index],
                ),

            "plane":
                np.asarray(
                    raw_wall["plane"],
                    dtype=np.float64,
                ),

            "normal":
                normalize(
                    raw_wall["normal"]
                ),
        }

        walls.append(
            wall
        )

    # --------------------------------------------------------
    # Floor frame.
    # --------------------------------------------------------

    floor_normal = floor_plane[
        :3
    ]

    u, v = choose_floor_axes(
        floor_normal,
        walls,
    )

    origin = (
        -floor_plane[3]
        * floor_normal
    )

    print(
        "\nFloor frame:"
    )

    print(
        f"  Origin: "
        f"{origin}"
    )

    print(
        f"  U: "
        f"{u}"
    )

    print(
        f"  V: "
        f"{v}"
    )

    print(
        f"  N: "
        f"{floor_normal}"
    )

    # --------------------------------------------------------
    # Reconstruct extents from point cloud.
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "RECONSTRUCTING WALL EXTENTS"
    )

    print(
        "=" * 70
    )

    valid_walls = []

    for wall in walls:

        extent_data = reconstruct_wall_extent(
            wall,
            points,
            floor_plane,
            origin,
            u,
            v,
        )

        if extent_data is None:

            print(
                f"\nWall {wall['wall_id']}: "
                f"rejected — insufficient support."
            )

            continue

        wall[
            "extent_data"
        ] = extent_data

        wall[
            "line_2d"
        ] = plane_to_floor_line(
            wall["plane"],
            origin,
            u,
            v,
        )

        if (
            extent_data["extent"]
            < MIN_WALL_EXTENT
        ):

            print(
                f"\nWall {wall['wall_id']}: "
                f"rejected — extent too small."
            )

            continue

        valid_walls.append(
            wall
        )

        print(
            f"\nWall {wall['wall_id']}"
        )

        print(
            f"  Source: "
            f"{wall['source_wall_ids']}"
        )

        print(
            f"  Points: "
            f"{extent_data['points']:,}"
        )

        print(
            f"  Extent: "
            f"{extent_data['extent']:.3f} m"
        )

        print(
            f"  Wall coordinate: "
            f"{extent_data['wall_min']:.3f}"
            f" -> "
            f"{extent_data['wall_max']:.3f}"
        )

        print(
            f"  U range: "
            f"{extent_data['u_min']:.3f}"
            f" -> "
            f"{extent_data['u_max']:.3f}"
        )

        print(
            f"  V range: "
            f"{extent_data['v_min']:.3f}"
            f" -> "
            f"{extent_data['v_max']:.3f}"
        )

    if len(valid_walls) < 4:

        raise RuntimeError(
            "Fewer than four valid wall candidates."
        )

    # --------------------------------------------------------
    # Trajectory.
    # --------------------------------------------------------

    trajectory = load_trajectory(
        capture_dir
    )

    trajectory_2d = project_points(
        trajectory,
        origin,
        u,
        v,
    )

    print(
        "\nTrajectory:"
    )

    print(
        f"  Frames: "
        f"{len(trajectory_2d):,}"
    )

    print(
        f"  U: "
        f"{trajectory_2d[:, 0].min():.3f}"
        f" -> "
        f"{trajectory_2d[:, 0].max():.3f}"
    )

    print(
        f"  V: "
        f"{trajectory_2d[:, 1].min():.3f}"
        f" -> "
        f"{trajectory_2d[:, 1].max():.3f}"
    )

    # --------------------------------------------------------
    # Wall families.
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

    for family_index, family in enumerate(
        families,
        start=1,
    ):

        print(
            f"\nFamily {family_index}"
        )

        for wall in family:

            print(
                f"  Wall {wall['wall_id']}"
                f"  extent="
                f"{wall['extent_data']['extent']:.3f} m"
            )

    if len(families) < 2:

        raise RuntimeError(
            "Need at least two wall families."
        )

    # --------------------------------------------------------
    # Pair candidates.
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

                a = family[i]
                b = family[j]

                separation = pair_separation(
                    a,
                    b,
                )

                overlap = pair_overlap(
                    a,
                    b,
                )

                if (
                    separation
                    < MIN_ROOM_DIMENSION
                    or separation
                    > MAX_ROOM_DIMENSION
                ):

                    continue

                if overlap < 0.30:

                    continue

                pair_candidates.append(
                    {
                        "family":
                            family_index,

                        "walls":
                            (
                                a,
                                b,
                            ),

                        "separation":
                            separation,

                        "overlap":
                            overlap,
                    }
                )

    print(
        "\n" + "=" * 70
    )

    print(
        "PAIR CANDIDATES"
    )

    print(
        "=" * 70
    )

    for candidate in pair_candidates:

        a, b = candidate[
            "walls"
        ]

        print(
            f"\nFamily "
            f"{candidate['family'] + 1}"
        )

        print(
            f"  Walls: "
            f"{a['wall_id']} "
            f"<-> "
            f"{b['wall_id']}"
        )

        print(
            f"  Separation: "
            f"{candidate['separation']:.3f} m"
        )

        print(
            f"  Overlap: "
            f"{candidate['overlap']:.3f}"
        )

    # --------------------------------------------------------
    # Combine perpendicular families.
    # --------------------------------------------------------

    candidates = []

    for i in range(
        len(pair_candidates)
    ):

        for j in range(
            i + 1,
            len(pair_candidates),
        ):

            pair_a = pair_candidates[i]
            pair_b = pair_candidates[j]

            if (
                pair_a["family"]
                == pair_b["family"]
            ):

                continue

            wall_a = pair_a[
                "walls"
            ][0]

            wall_b = pair_b[
                "walls"
            ][0]

            angle = angle_between_normals(
                wall_a["normal"],
                wall_b["normal"],
            )

            if abs(
                angle - 90.0
            ) > 20.0:

                continue

            result = build_candidate(
                pair_a["walls"],
                pair_b["walls"],
                trajectory_2d,
            )

            if result is None:
                continue

            result[
                "family_a"
            ] = pair_a["family"]

            result[
                "family_b"
            ] = pair_b["family"]

            candidates.append(
                result
            )

    if not candidates:

        raise RuntimeError(
            "\nNo room envelope candidates "
            "passed trajectory containment."
        )

    candidates.sort(
        key=lambda candidate:
        -candidate["score"]
    )

    # --------------------------------------------------------
    # Candidate results.
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "ROOM ENVELOPE CANDIDATES"
    )

    print(
        "=" * 70
    )

    for index, candidate in enumerate(
        candidates,
        start=1,
    ):

        a = candidate[
            "pair_a"
        ]

        b = candidate[
            "pair_b"
        ]

        print(
            f"\nCandidate {index}"
        )

        print(
            f"  Family A: "
            f"{candidate['family_a'] + 1}"
        )

        print(
            f"  Walls: "
            f"{a[0]['wall_id']} "
            f"<-> "
            f"{a[1]['wall_id']}"
        )

        print(
            f"  Dimension A: "
            f"{candidate['dimension_a']:.3f} m"
        )

        print(
            f"  Family B: "
            f"{candidate['family_b'] + 1}"
        )

        print(
            f"  Walls: "
            f"{b[0]['wall_id']} "
            f"<-> "
            f"{b[1]['wall_id']}"
        )

        print(
            f"  Dimension B: "
            f"{candidate['dimension_b']:.3f} m"
        )

        print(
            f"  Area: "
            f"{candidate['area']:.3f} m²"
        )

        print(
            f"  Trajectory inside: "
            f"{candidate['trajectory_inside_ratio']:.3f}"
        )

        print(
            f"  Support: "
            f"{candidate['support_points']:,}"
        )

        print(
            f"  Score: "
            f"{candidate['score']:.3f}"
        )

    # --------------------------------------------------------
    # Best candidate.
    # --------------------------------------------------------

    best = candidates[0]

    pair_a = best[
        "pair_a"
    ]

    pair_b = best[
        "pair_b"
    ]

    polygon = best[
        "corners"
    ]

    print(
        "\n" + "=" * 70
    )

    print(
        "SELECTED ROOM ENVELOPE"
    )

    print(
        "=" * 70
    )

    print(
        f"\nWall pair A: "
        f"{pair_a[0]['wall_id']} "
        f"<-> "
        f"{pair_a[1]['wall_id']}"
    )

    print(
        f"Dimension A: "
        f"{best['dimension_a']:.3f} m"
    )

    print(
        f"\nWall pair B: "
        f"{pair_b[0]['wall_id']} "
        f"<-> "
        f"{pair_b[1]['wall_id']}"
    )

    print(
        f"Dimension B: "
        f"{best['dimension_b']:.3f} m"
    )

    print(
        f"\nArea: "
        f"{best['area']:.3f} m²"
    )

    print(
        f"Trajectory coverage: "
        f"{best['trajectory_inside_ratio'] * 100:.1f}%"
    )

    print(
        "\nCorners:"
    )

    for index, corner in enumerate(
        polygon,
        start=1,
    ):

        print(
            f"  C{index}: "
            f"("
            f"{corner[0]:.3f}, "
            f"{corner[1]:.3f}"
            f")"
        )

    # --------------------------------------------------------
    # Save JSON.
    # --------------------------------------------------------

    output_json = (
        walls_path.parent
        / "room_envelope.json"
    )

    output_data = {

        "method":
            "trajectory_constrained_room_envelope",

        "dimensions_m": {

            "dimension_a":
                best["dimension_a"],

            "dimension_b":
                best["dimension_b"],
        },

        "area_m2":
            best["area"],

        "trajectory_inside_ratio":
            best[
                "trajectory_inside_ratio"
            ],

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

        "selected_wall_pairs": [

            {
                "walls": [
                    int(
                        pair_a[0]["wall_id"]
                    ),
                    int(
                        pair_a[1]["wall_id"]
                    ),
                ],

                "dimension_m":
                    best[
                        "dimension_a"
                    ],
            },

            {
                "walls": [
                    int(
                        pair_b[0]["wall_id"]
                    ),
                    int(
                        pair_b[1]["wall_id"]
                    ),
                ],

                "dimension_m":
                    best[
                        "dimension_b"
                    ],
            },
        ],

        "edge_lengths_m": [
            float(length)
            for length in best[
                "edge_lengths"
            ]
        ],
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
        f"\nSaved:"
        f"\n{output_json}"
    )

    # --------------------------------------------------------
    # Plot.
    # --------------------------------------------------------

    closed_polygon = np.vstack(
        [
            polygon,
            polygon[0],
        ]
    )

    plt.figure(
        figsize=(10, 8)
    )

    plt.plot(
        closed_polygon[:, 0],
        closed_polygon[:, 1],
        linewidth=3,
        marker="o",
        label="Selected room",
    )

    plt.plot(
        trajectory_2d[:, 0],
        trajectory_2d[:, 1],
        linewidth=1,
        label="Camera trajectory",
    )

    for index, point in enumerate(
        polygon,
        start=1,
    ):

        plt.annotate(
            f"C{index}",
            (
                point[0],
                point[1],
            ),
            xytext=(5, 5),
            textcoords="offset points",
        )

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
        "Selected Room Envelope + Camera Trajectory"
    )

    plt.axis(
        "equal"
    )

    plt.grid(
        True
    )

    plt.legend()

    output_plot = (
        walls_path.parent
        / "room_envelope.png"
    )

    plt.savefig(
        output_plot,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"\nSaved:"
        f"\n{output_plot}"
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "ROOM ENVELOPE SELECTION COMPLETE"
    )

    print(
        "=" * 70
    )

if __name__ == "__main__":
    main()
