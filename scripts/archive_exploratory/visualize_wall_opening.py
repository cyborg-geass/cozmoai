from pathlib import Path
import json
import sys

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


WALL_ID = 5
PLANE_DISTANCE = 0.05


def normalize_plane(plane):
    plane = np.asarray(plane, dtype=np.float64)
    return plane / np.linalg.norm(plane[:3])


def find_wall(data, wall_id):
    for wall in data["walls"]:
        if int(wall["wall_id"]) == wall_id:
            return wall
    raise ValueError(f"Wall {wall_id} not found.")


def main():

    if len(sys.argv) != 4:
        print(
            "Usage:\n"
            "uv run python scripts/visualize_wall_opening.py "
            "<pointcloud.ply> <room_geometry.json> <walls.json>"
        )
        sys.exit(1)

    pointcloud_path = Path(sys.argv[1])
    room_path = Path(sys.argv[2])
    walls_path = Path(sys.argv[3])

    room = json.loads(room_path.read_text())
    walls = json.loads(walls_path.read_text())

    wall = find_wall(walls, WALL_ID)
    plane = normalize_plane(wall["plane"])

    frame = room["floor_coordinate_system"]

    origin = np.asarray(frame["origin"], dtype=float)
    floor_normal = np.asarray(frame["floor_normal"], dtype=float)
    floor_normal /= np.linalg.norm(floor_normal)

    wall_normal = plane[:3]
    wall_normal /= np.linalg.norm(wall_normal)

    # Horizontal direction along wall.
    wall_axis = np.cross(floor_normal, wall_normal)
    wall_axis /= np.linalg.norm(wall_axis)

    # Re-orthogonalize.
    wall_normal = np.cross(wall_axis, floor_normal)
    wall_normal /= np.linalg.norm(wall_normal)

    print("=" * 70)
    print("WALL OPENING VISUALIZATION")
    print("=" * 70)

    print("\nWall:", WALL_ID)
    print("Wall normal:", wall_normal)
    print("Wall axis:", wall_axis)
    print("Vertical:", floor_normal)

    # --------------------------------------------------------
    # Load point cloud
    # --------------------------------------------------------

    pcd = o3d.io.read_point_cloud(str(pointcloud_path))
    points = np.asarray(pcd.points)

    distances = np.abs(
        points @ plane[:3] + plane[3]
    )

    mask = distances <= PLANE_DISTANCE
    wall_points = points[mask]

    print("\nWall points:", len(wall_points))

    # --------------------------------------------------------
    # Wall-local coordinates
    # --------------------------------------------------------

    relative = wall_points - origin

    s = relative @ wall_axis
    h = relative @ floor_normal

    valid = (
        np.isfinite(s)
        & np.isfinite(h)
        & (h >= -0.10)
        & (h <= 3.0)
    )

    s = s[valid]
    h = h[valid]

    print(
        "Horizontal:",
        f"{s.min():.3f} -> {s.max():.3f}"
    )

    print(
        "Height:",
        f"{h.min():.3f} -> {h.max():.3f}"
    )

    # --------------------------------------------------------
    # Occupancy grid
    # --------------------------------------------------------

    ds = 0.025

    s_min = np.floor(s.min() / ds) * ds
    s_max = np.ceil(s.max() / ds) * ds

    h_min = 0.0
    h_max = np.ceil(h.max() / ds) * ds

    s_edges = np.arange(
        s_min,
        s_max + ds,
        ds
    )

    h_edges = np.arange(
        h_min,
        h_max + ds,
        ds
    )

    occupancy, _, _ = np.histogram2d(
        s,
        h,
        bins=[s_edges, h_edges]
    )

    occupancy = occupancy.T

    # Log scale makes sparse regions visible.
    image = np.log1p(occupancy)

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(16, 6))

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

    ax.axhline(
        1.8,
        linestyle="--",
        linewidth=1,
        label="1.8 m"
    )

    ax.axhline(
        2.0,
        linestyle="--",
        linewidth=1,
        label="2.0 m"
    )

    ax.set_xlabel("Distance along Wall 5 (m)")
    ax.set_ylabel("Height above floor (m)")
    ax.set_title(
        "Wall 5 — Point Cloud Occupancy"
    )

    ax.legend()

    ax.grid(
        alpha=0.15
    )

    output = (
        pointcloud_path.parent /
        "wall5_opening_visualization.png"
    )

    fig.tight_layout()
    fig.savefig(
        output,
        dpi=180
    )

    plt.close(fig)

    print("\nSaved:")
    print(output)


if __name__ == "__main__":
    main()
