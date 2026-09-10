from pathlib import Path
import json
import sys

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


WALL_ID = 5

OPENING_CENTER = -3.14

# Region around the detected opening.
S_HALF_WIDTH = 0.75

# Thickness around the wall plane.
# Larger than the 5 cm used for plane fitting so that
# geometry just behind/in front of the wall is visible.
DEPTH_HALF_WIDTH = 0.30

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

    if len(sys.argv) != 4:
        print(
            "Usage:\n"
            "uv run python scripts/measure_opening_3d.py "
            "<pointcloud.ply> "
            "<room_geometry.json> "
            "<walls.json>"
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

    # Horizontal direction along wall.
    wall_axis = np.cross(
        vertical,
        wall_normal
    )
    wall_axis /= np.linalg.norm(wall_axis)

    # --------------------------------------------------------
    # Load global point cloud
    # --------------------------------------------------------

    pcd = o3d.io.read_point_cloud(
        str(pointcloud_path)
    )

    points = np.asarray(pcd.points)

    # --------------------------------------------------------
    # Wall-local coordinates for EVERY point
    # --------------------------------------------------------

    relative = points - origin

    s = relative @ wall_axis
    h = relative @ vertical
    d = relative @ wall_normal

    # --------------------------------------------------------
    # Select a 3D slab around the opening.
    # --------------------------------------------------------

    mask = (
        (np.abs(s - OPENING_CENTER) <= S_HALF_WIDTH)
        & (np.abs(d) <= DEPTH_HALF_WIDTH)
        & (h >= -0.10)
        & (h <= 3.0)
    )

    s_roi = s[mask]
    h_roi = h[mask]

    print("=" * 70)
    print("3D OPENING MEASUREMENT")
    print("=" * 70)

    print("\nWall:", WALL_ID)

    print(
        "Opening center:",
        OPENING_CENTER
    )

    print(
        "ROI points:",
        len(s_roi)
    )

    print(
        "s range:",
        f"{s_roi.min():.3f} -> {s_roi.max():.3f}"
    )

    print(
        "height range:",
        f"{h_roi.min():.3f} -> {h_roi.max():.3f}"
    )

    # --------------------------------------------------------
    # Occupancy grid
    # --------------------------------------------------------

    s_min = (
        np.floor(s_roi.min() / S_BIN)
        * S_BIN
    )

    s_max = (
        np.ceil(s_roi.max() / S_BIN)
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
        s_roi,
        h_roi,
        bins=[s_edges, h_edges]
    )

    occupancy = occupancy.T

    # --------------------------------------------------------
    # Horizontal profile
    # --------------------------------------------------------

    occupied = occupancy > 0

    horizontal_profile = occupied.mean(
        axis=0
    )

    s_centers = (
        s_edges[:-1]
        + s_edges[1:]
    ) / 2.0

    # --------------------------------------------------------
    # Vertical profile
    # --------------------------------------------------------

    vertical_profile = occupied.mean(
        axis=1
    )

    h_centers = (
        h_edges[:-1]
        + h_edges[1:]
    ) / 2.0

    print("\nVertical occupancy profile:")

    for i in range(
        len(h_centers)
    ):

        # Print every 10th bin.
        if i % 10 == 0:

            print(
                f"  h={h_centers[i]:.3f} "
                f"occupancy="
                f"{vertical_profile[i]:.3f}"
            )

    # --------------------------------------------------------
    # Estimate opening width from horizontal occupancy.
    #
    # We use the lower part of the wall where a doorway
    # should be present.
    # --------------------------------------------------------

    lower_height = (
        h_centers <= 1.8
    )

    lower_occupancy = occupied[
        lower_height,
        :
    ].mean(axis=0)

    # Low occupancy indicates the opening.
    opening_mask = (
        lower_occupancy < 0.15
    )

    # Find contiguous runs.
    runs = []

    start = None

    for i, value in enumerate(
        opening_mask
    ):

        if value and start is None:
            start = i

        elif not value and start is not None:
            runs.append(
                (start, i - 1)
            )
            start = None

    if start is not None:
        runs.append(
            (start, len(opening_mask) - 1)
        )

    print("\nOpening runs:")

    candidates = []

    for start, end in runs:

        left = s_edges[start]
        right = s_edges[end + 1]

        width = right - left

        if (
            0.35
            <= width
            <= 1.5
        ):

            candidates.append(
                {
                    "left_s_m": float(left),
                    "right_s_m": float(right),
                    "width_m": float(width),
                }
            )

            print(
                f"  {left:.3f} -> "
                f"{right:.3f} "
                f"width={width:.3f} m"
            )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(15, 7)
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

    ax.axvline(
        OPENING_CENTER,
        linestyle="--",
        linewidth=2,
        label="Opening center"
    )

    for candidate in candidates:

        ax.axvline(
            candidate["left_s_m"],
            linestyle=":",
            linewidth=1.5
        )

        ax.axvline(
            candidate["right_s_m"],
            linestyle=":",
            linewidth=1.5
        )

    ax.set_xlabel(
        "Distance along Wall 5 (m)"
    )

    ax.set_ylabel(
        "Height above floor (m)"
    )

    ax.set_title(
        "3D Point-Cloud Cross-Section — Wall 5 Opening"
    )

    ax.legend()

    ax.grid(
        alpha=0.15
    )

    output = (
        pointcloud_path.parent
        / "opening_3d_measurement_wall5.png"
    )

    fig.tight_layout()

    fig.savefig(
        output,
        dpi=180
    )

    plt.close(fig)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    result = {
        "wall_id": WALL_ID,
        "candidate_center_m": OPENING_CENTER,
        "candidates": candidates,
        "measurement_status": (
            "width_candidate_only"
            if candidates
            else "not_detected"
        ),
    }

    json_output = (
        pointcloud_path.parent
        / "opening_3d_measurement_wall5.json"
    )

    json_output.write_text(
        json.dumps(
            result,
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
