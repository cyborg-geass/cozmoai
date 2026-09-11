from pathlib import Path
import json
import sys

import numpy as np
import open3d as o3d


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

WALL_ID = 5

# Distance from the fitted wall plane used to collect points.
# Keep this fairly tight so nearby objects don't contaminate
# the wall occupancy map.
PLANE_DISTANCE = 0.05

# Resolution of the wall occupancy grid.
HORIZONTAL_BIN = 0.05   # metres along wall
HEIGHT_BIN = 0.05       # metres vertically

# Minimum fraction of occupied cells in a horizontal bin
# for considering that bin to contain wall material.
WALL_OCCUPANCY_THRESHOLD = 0.20

# A doorway should be a sufficiently tall continuous gap.
MIN_OPENING_HEIGHT = 1.8
MAX_OPENING_HEIGHT = 2.5
MIN_OPENING_WIDTH = 0.5
MAX_OPENING_WIDTH = 2.5

# The top of an opening is only considered observed when there
# is sustained wall material across the opening span above the
# minimum doorway height. Sparse/noisy points are reported as
# unknown instead of being clamped into a made-up height.
TOP_SUPPORT_FRACTION = 0.50
MIN_TOP_SUPPORT_BINS = 2


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def normalize_plane(plane):
    plane = np.asarray(plane, dtype=np.float64)
    norm = np.linalg.norm(plane[:3])

    if norm == 0:
        raise ValueError("Invalid plane: zero normal.")

    return plane / norm


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_wall(walls_data, wall_id):
    """
    Find a wall using the actual walls.json schema.
    """
    walls = walls_data.get("walls", walls_data)

    for wall in walls:
        if int(wall.get("wall_id", -1)) == wall_id:
            return wall

    available = [
        wall.get("wall_id")
        for wall in walls
        if "wall_id" in wall
    ]

    raise ValueError(
        f"Wall {wall_id} not found. "
        f"Available wall IDs: {available}"
    )


def extract_plane(wall):
    """
    Accept the plane under the schemas used by our earlier
    geometry scripts.
    """
    if "plane" in wall:
        return np.asarray(wall["plane"], dtype=np.float64)

    if "equation" in wall:
        return np.asarray(wall["equation"], dtype=np.float64)

    raise ValueError("Could not find plane equation in wall record.")


def project_to_floor(points, origin, U, V, N):
    d = points - origin

    u = d @ U
    v = d @ V
    h = d @ N

    return u, v, h


def estimate_opening_height(
    occupied,
    start,
    end,
    height_bin,
    min_opening_height,
    max_opening_height,
    support_fraction=TOP_SUPPORT_FRACTION,
    min_support_bins=MIN_TOP_SUPPORT_BINS,
):
    opening_columns = occupied[start:end + 1]

    if opening_columns.size == 0:
        return None, "not_observed", {
            "reason": "opening span has no occupancy columns",
            "top_support_fraction": support_fraction,
            "min_top_support_bins": min_support_bins,
        }

    occupancy_by_height = opening_columns.mean(axis=0)

    min_bin = int(
        np.ceil(
            min_opening_height
            / height_bin
        )
    )

    max_bin = min(
        len(occupancy_by_height),
        int(
            np.floor(
                max_opening_height
                / height_bin
            )
        ) + 1,
    )

    supported = (
        occupancy_by_height[min_bin:max_bin]
        >= support_fraction
    )

    run_start = None

    for index, value in enumerate(supported):
        if value and run_start is None:
            run_start = index
        elif not value and run_start is not None:
            run_length = index - run_start

            if run_length >= min_support_bins:
                height = (
                    min_bin
                    + run_start
                ) * height_bin

                return float(height), "observed", {
                    "top_support_fraction": support_fraction,
                    "top_support_bins": int(run_length),
                    "top_support_height_bin": int(
                        min_bin + run_start
                    ),
                }

            run_start = None

    if run_start is not None:
        run_length = len(supported) - run_start

        if run_length >= min_support_bins:
            height = (
                min_bin
                + run_start
            ) * height_bin

            return float(height), "observed", {
                "top_support_fraction": support_fraction,
                "top_support_bins": int(run_length),
                "top_support_height_bin": int(
                    min_bin + run_start
                ),
            }

    return None, "not_observed", {
        "reason": (
            "no sustained wall material observed across the opening "
            "span above the minimum opening height"
        ),
        "top_support_fraction": support_fraction,
        "min_top_support_bins": min_support_bins,
    }


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    if len(sys.argv) != 4:
        print(
            "Usage:\n"
            "  uv run python scripts/detect_openings.py "
            "<pointcloud.ply> <room_geometry.json> <walls.json>"
        )
        sys.exit(1)

    pointcloud_path = Path(sys.argv[1])
    room_geometry_path = Path(sys.argv[2])
    walls_path = Path(sys.argv[3])

    print("=" * 70)
    print("OPENING DETECTION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load geometry
    # --------------------------------------------------------

    room = load_json(room_geometry_path)
    walls_data = load_json(walls_path)

    wall = find_wall(walls_data, WALL_ID)

    plane = normalize_plane(extract_plane(wall))

    print("\nWall:", WALL_ID)
    print("Plane:")
    print(plane)

    # Floor frame
    floor = room["floor_coordinate_system"]

    origin = np.asarray(
        floor["origin"],
        dtype=np.float64
    )

    U = np.asarray(
        floor["u_axis"],
        dtype=np.float64
    )

    V = np.asarray(
        floor["v_axis"],
        dtype=np.float64
    )

    N = np.asarray(
        floor["floor_normal"],
        dtype=np.float64
    )

    # Make sure frame is normalized.
    U /= np.linalg.norm(U)
    V /= np.linalg.norm(V)
    N /= np.linalg.norm(N)

    print("\nFloor frame:")
    print("Origin:", origin)
    print("U:", U)
    print("V:", V)
    print("N:", N)

    # --------------------------------------------------------
    # Load global point cloud
    # --------------------------------------------------------

    pcd = o3d.io.read_point_cloud(str(pointcloud_path))
    points = np.asarray(pcd.points)

    if len(points) == 0:
        raise RuntimeError("Point cloud is empty.")

    print("\nGlobal points:", len(points))

    # --------------------------------------------------------
    # Extract points close to Wall 5
    # --------------------------------------------------------

    distances = np.abs(
        points @ plane[:3] + plane[3]
    )

    wall_mask = distances <= PLANE_DISTANCE
    wall_points = points[wall_mask]

    print(
        f"Points within {PLANE_DISTANCE:.3f} m of wall:",
        len(wall_points)
    )

    if len(wall_points) < 1000:
        raise RuntimeError(
            "Too few wall points. Plane distance threshold may be too strict."
        )

    # --------------------------------------------------------
    # Convert wall points into local coordinates
    #
    # v = horizontal position along wall
    # h = vertical height above floor
    #
    # U is approximately the normal direction of Wall 5,
    # while V runs along the wall.
    # --------------------------------------------------------

    # u, v, h = project_to_floor(
    #     wall_points,
    #     origin,
    #     U,
    #     V,
    #     N
    # )

    # # --------------------------------------------------------
    # # Keep only points physically close to the wall's
    # # horizontal location and above the floor.
    # # --------------------------------------------------------

    # wall_u_center = np.median(u)

    # u_tolerance = 0.15

    # mask = (
    #     (np.abs(u - wall_u_center) <= u_tolerance)
    #     & (h >= -0.10)
    #     & (h <= 3.5)
    # )

    # v = v[mask]
    # h = h[mask]
    # --------------------------------------------------------
    # Build a wall-local coordinate system
    #
    # wall_normal = perpendicular to wall
    # vertical     = floor normal
    # wall_axis   = horizontal direction along wall
    # --------------------------------------------------------

    wall_normal = plane[:3].copy()
    wall_normal /= np.linalg.norm(wall_normal)

    vertical = N.copy()
    vertical /= np.linalg.norm(vertical)

    # Direction running along the wall.
    wall_axis = np.cross(vertical, wall_normal)
    wall_axis /= np.linalg.norm(wall_axis)

    # Re-orthogonalize wall normal against vertical.
    wall_normal = np.cross(wall_axis, vertical)
    wall_normal /= np.linalg.norm(wall_normal)

    print("\nWall-local frame:")
    print("Wall normal:", wall_normal)
    print("Wall axis:", wall_axis)
    print("Vertical:", vertical)

    # --------------------------------------------------------
    # Project wall points
    #
    # s = horizontal distance along wall
    # h = height above floor
    # d = distance through wall
    # --------------------------------------------------------

    relative = wall_points - origin

    s = relative @ wall_axis
    h = relative @ vertical
    d = relative @ wall_normal

    # --------------------------------------------------------
    # Keep points close to the actual wall surface.
    # --------------------------------------------------------

    wall_center_distance = np.median(d)

    mask = (
        (np.abs(d - wall_center_distance) <= 0.10)
        & (h >= -0.10)
        & (h <= 3.5)
    )

    s = s[mask]
    h = h[mask]

    print("\nWall-local points:", len(s))
    print(
        "Horizontal range:",
        f"{s.min():.3f} -> {s.max():.3f} m"
    )
    print(
        "Height range:",
        f"{h.min():.3f} -> {h.max():.3f} m"
    )

    # --------------------------------------------------------
    # Build occupancy grid
    # --------------------------------------------------------

    v_min = np.floor(s.min() / HORIZONTAL_BIN) * HORIZONTAL_BIN
    v_max = np.ceil(s.max() / HORIZONTAL_BIN) * HORIZONTAL_BIN

    h_min = 0.0
    h_max = max(3.0, np.ceil(h.max() / HEIGHT_BIN) * HEIGHT_BIN)

    v_edges = np.arange(
        v_min,
        v_max + HORIZONTAL_BIN,
        HORIZONTAL_BIN
    )

    h_edges = np.arange(
        h_min,
        h_max + HEIGHT_BIN,
        HEIGHT_BIN
    )

    occupancy, _, _ = np.histogram2d(
        s,
        h,
        bins=[v_edges, h_edges]
    )

    occupied = occupancy > 0

    # --------------------------------------------------------
    # For each horizontal location determine how much
    # vertical wall structure exists.
    # --------------------------------------------------------

    vertical_occupancy = occupied.mean(axis=1)

    # Number of occupied height cells from the floor upward.
    #
    # This is especially useful for doorways:
    #
    # normal wall:
    #
    # █████████
    # █████████
    # █████████
    # █████████
    #
    # doorway:
    #
    # █     █
    # █     █
    # █     █
    # ███████
    #
    # Therefore a doorway tends to have low occupancy from
    # floor -> ~2m while adjacent wall regions remain dense.
    # --------------------------------------------------------

    n_height_bins = len(h_edges) - 1

    floor_gap_score = np.zeros(len(v_edges) - 1)

    door_height_bins = int(
        MIN_OPENING_HEIGHT / HEIGHT_BIN
    )

    for i in range(len(floor_gap_score)):

        column = occupied[i]

        # Examine the vertical region from floor to the
        # expected door height.
        lower = column[:door_height_bins]

        floor_gap_score[i] = 1.0 - lower.mean()

    # --------------------------------------------------------
    # Candidate opening bins
    # --------------------------------------------------------

    candidate = floor_gap_score >= (
        1.0 - WALL_OCCUPANCY_THRESHOLD
    )

    # Remove tiny isolated regions.
    runs = []

    start = None

    for i, value in enumerate(candidate):

        if value and start is None:
            start = i

        elif not value and start is not None:
            runs.append((start, i - 1))
            start = None

    if start is not None:
        runs.append((start, len(candidate) - 1))

    print("\nCandidate opening regions:")

    candidates = []

    for start, end in runs:

        left = v_edges[start]
        right = v_edges[end + 1]

        width = right - left

        if width < MIN_OPENING_WIDTH:
            continue

        if width > MAX_OPENING_WIDTH:
            continue

        opening_height, height_status, height_evidence = (
            estimate_opening_height(
                occupied,
                start,
                end,
                HEIGHT_BIN,
                MIN_OPENING_HEIGHT,
                MAX_OPENING_HEIGHT,
            )
        )

        score = float(
            np.mean(floor_gap_score[start:end + 1])
        )

        candidates.append({
            "left_s": float(left),
            "right_s": float(right),
            "width_m": float(width),
            "height_m": (
                float(opening_height)
                if opening_height is not None
                else None
            ),
            "height_status": height_status,
            "height_evidence": height_evidence,
            "score": score,
        })

        height_label = (
            f"{opening_height:.2f} m"
            if opening_height is not None
            else "not observed"
        )

        print(
            f"  s={left:.2f} -> {right:.2f} "
            f"width={width:.2f} m "
            f"height={height_label} "
            f"height_status={height_status} "
            f"score={score:.3f}"
        )

    # --------------------------------------------------------
    # Save result
    # --------------------------------------------------------

    output_dir = pointcloud_path.parent
    output_path = output_dir / "openings_wall5.json"

    result = {
        "wall_id": WALL_ID,
        "plane": plane.tolist(),
        "plane_distance_threshold_m": PLANE_DISTANCE,
        "candidates": candidates,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("\nSaved:")
    print(output_path)

    # --------------------------------------------------------
    # Also save a simple diagnostic CSV-like text file
    # --------------------------------------------------------

    profile_path = output_dir / "wall5_opening_profile.csv"

    with open(profile_path, "w", encoding="utf-8") as f:

        f.write("v_center_m,floor_gap_score,vertical_occupancy\n")

        for i in range(len(floor_gap_score)):

            center = (
                v_edges[i] + v_edges[i + 1]
            ) / 2.0

            f.write(
                f"{center:.5f},"
                f"{floor_gap_score[i]:.5f},"
                f"{vertical_occupancy[i]:.5f}\n"
            )

    print(profile_path)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
