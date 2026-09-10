from pathlib import Path
import json
import sys

import numpy as np
import open3d as o3d
import pandas as pd
import matplotlib.pyplot as plt


WALL_ID = 5

# Candidate identified from trajectory + occupancy evidence.
OPENING_CENTER = -3.14

# Examine a wider region around the candidate.
SEARCH_HALF_WIDTH = 0.65

PLANE_DISTANCE = 0.05

# Wall-local resolution.
S_BIN = 0.025
H_BIN = 0.025


def normalize_plane(plane):
    plane = np.asarray(plane, dtype=np.float64)
    return plane / np.linalg.norm(plane[:3])


def find_wall(data, wall_id):
    for wall in data["walls"]:
        if int(wall["wall_id"]) == wall_id:
            return wall
    raise ValueError(f"Wall {wall_id} not found.")


def main():

    if len(sys.argv) != 5:
        print(
            "Usage:\n"
            "uv run python scripts/analyze_opening_candidate.py "
            "<pointcloud.ply> "
            "<room_geometry.json> "
            "<walls.json> "
            "<odometry.csv>"
        )
        sys.exit(1)

    pointcloud_path = Path(sys.argv[1])
    room_path = Path(sys.argv[2])
    walls_path = Path(sys.argv[3])
    odometry_path = Path(sys.argv[4])

    room = json.loads(room_path.read_text())
    walls = json.loads(walls_path.read_text())

    wall = find_wall(walls, WALL_ID)

    plane = normalize_plane(wall["plane"])

    frame = room["floor_coordinate_system"]

    origin = np.asarray(
        frame["origin"],
        dtype=np.float64
    )

    vertical = np.asarray(
        frame["floor_normal"],
        dtype=np.float64
    )
    vertical /= np.linalg.norm(vertical)

    wall_normal = plane[:3].copy()
    wall_normal /= np.linalg.norm(wall_normal)

    wall_axis = np.cross(
        vertical,
        wall_normal
    )
    wall_axis /= np.linalg.norm(wall_axis)

    print("=" * 70)
    print("OPENING CANDIDATE ANALYSIS")
    print("=" * 70)

    print("\nWall:", WALL_ID)
    print("Candidate center:", OPENING_CENTER)

    # --------------------------------------------------------
    # Point cloud
    # --------------------------------------------------------

    pcd = o3d.io.read_point_cloud(
        str(pointcloud_path)
    )

    points = np.asarray(pcd.points)

    distances = np.abs(
        points @ plane[:3] + plane[3]
    )

    mask = distances <= PLANE_DISTANCE

    wall_points = points[mask]

    relative = wall_points - origin

    s = relative @ wall_axis
    h = relative @ vertical

    valid = (
        np.isfinite(s)
        & np.isfinite(h)
        & (h >= -0.10)
        & (h <= 3.0)
    )

    s = s[valid]
    h = h[valid]

    # --------------------------------------------------------
    # Candidate region
    # --------------------------------------------------------

    region_mask = (
        (s >= OPENING_CENTER - SEARCH_HALF_WIDTH)
        & (s <= OPENING_CENTER + SEARCH_HALF_WIDTH)
    )

    sr = s[region_mask]
    hr = h[region_mask]

    print("\nCandidate region:")
    print(
        f"s = "
        f"{OPENING_CENTER - SEARCH_HALF_WIDTH:.3f}"
        f" -> "
        f"{OPENING_CENTER + SEARCH_HALF_WIDTH:.3f}"
    )

    print("Points:", len(sr))

    # --------------------------------------------------------
    # Occupancy grid
    # --------------------------------------------------------

    s_min = (
        np.floor(sr.min() / S_BIN)
        * S_BIN
    )

    s_max = (
        np.ceil(sr.max() / S_BIN)
        * S_BIN
    )

    h_min = 0.0
    h_max = 2.5

    s_edges = np.arange(
        s_min,
        s_max + S_BIN,
        S_BIN
    )

    h_edges = np.arange(
        h_min,
        h_max + H_BIN,
        H_BIN
    )

    occupancy, _, _ = np.histogram2d(
        sr,
        hr,
        bins=[s_edges, h_edges]
    )

    occupancy = occupancy.T

    # --------------------------------------------------------
    # Vertical occupancy profile
    # --------------------------------------------------------

    occupied = occupancy > 0

    vertical_fraction = occupied.mean(axis=0)

    s_centers = (
        s_edges[:-1] +
        s_edges[1:]
    ) / 2.0

    # --------------------------------------------------------
    # Print density profile around candidate
    # --------------------------------------------------------

    print("\nHorizontal occupancy profile:")

    for i in range(len(s_centers)):

        if abs(
            s_centers[i] - OPENING_CENTER
        ) <= SEARCH_HALF_WIDTH:

            print(
                f"  s={s_centers[i]:7.3f} "
                f"occupancy={vertical_fraction[i]:.3f}"
            )

    # --------------------------------------------------------
    # Load trajectory
    # --------------------------------------------------------

    df = pd.read_csv(
        odometry_path
    )

    df.columns = df.columns.str.strip()

    positions = df[
        ["x", "y", "z"]
    ].to_numpy(
        dtype=np.float64
    )

    frames = df[
        "frame"
    ].to_numpy()

    signed_distance = (
        positions @ plane[:3]
        + plane[3]
    )

    trajectory_s = (
        (positions - origin)
        @ wall_axis
    )

    trajectory_h = (
        (positions - origin)
        @ vertical
    )

    # Find crossings.
    crossings = []

    for i in range(
        len(positions) - 1
    ):

        d1 = signed_distance[i]
        d2 = signed_distance[i + 1]

        if d1 * d2 >= 0:
            continue

        alpha = -d1 / (
            d2 - d1
        )

        p = (
            positions[i]
            + alpha
            * (
                positions[i + 1]
                - positions[i]
            )
        )

        rel = p - origin

        s_cross = (
            rel @ wall_axis
        )

        h_cross = (
            rel @ vertical
        )

        if abs(
            s_cross - OPENING_CENTER
        ) <= SEARCH_HALF_WIDTH:

            crossings.append({
                "frame_before": int(frames[i]),
                "frame_after": int(frames[i + 1]),
                "s_m": float(s_cross),
                "height_m": float(h_cross),
            })

    print("\nTrajectory crossings near candidate:")

    for c in crossings:

        print(
            f"  frames "
            f"{c['frame_before']} -> "
            f"{c['frame_after']} | "
            f"s={c['s_m']:.3f} m | "
            f"h={c['height_m']:.3f} m"
        )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(14, 7)
    )

    image = np.log1p(
        occupancy
    )

    ax.imshow(
        image,
        origin="lower",
        aspect="auto",
        extent=[
            s_edges[0],
            s_edges[-1],
            h_edges[0],
            h_edges[-1],
        ],
    )

    # Candidate center.
    ax.axvline(
        OPENING_CENTER,
        linestyle="--",
        linewidth=2,
        label="Trajectory crossing center"
    )

    # Crossing points.
    for c in crossings:

        ax.scatter(
            c["s_m"],
            c["height_m"],
            s=80,
            marker="x",
            linewidths=2,
        )

    ax.set_xlabel(
        "Distance along Wall 5 (m)"
    )

    ax.set_ylabel(
        "Height above floor (m)"
    )

    ax.set_title(
        "Wall 5 — Opening Candidate at s ≈ -3.14 m"
    )

    ax.legend()

    ax.grid(
        alpha=0.15
    )

    output = (
        pointcloud_path.parent
        / "opening_candidate_wall5.png"
    )

    fig.tight_layout()

    fig.savefig(
        output,
        dpi=180
    )

    plt.close(fig)

    # --------------------------------------------------------
    # Save evidence JSON
    # --------------------------------------------------------

    evidence = {
        "wall_id": WALL_ID,
        "candidate_center_m": OPENING_CENTER,
        "search_half_width_m": SEARCH_HALF_WIDTH,
        "candidate_point_count": int(len(sr)),
        "trajectory_crossings": crossings,
        "wall_axis": wall_axis.tolist(),
        "wall_normal": wall_normal.tolist(),
        "vertical": vertical.tolist(),
    }

    json_output = (
        pointcloud_path.parent
        / "opening_candidate_wall5.json"
    )

    json_output.write_text(
        json.dumps(
            evidence,
            indent=2
        )
    )

    print("\nSaved:")
    print(output)
    print(json_output)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
