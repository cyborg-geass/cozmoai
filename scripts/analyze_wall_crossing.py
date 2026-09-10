from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd


WALL_ID = 5


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
            "uv run python scripts/analyze_wall_crossing.py "
            "<odometry.csv> <room_geometry.json> <walls.json>"
        )
        sys.exit(1)

    odometry_path = Path(sys.argv[1])
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

    floor_normal = np.asarray(
        frame["floor_normal"],
        dtype=np.float64
    )
    floor_normal /= np.linalg.norm(floor_normal)

    wall_normal = plane[:3].copy()
    wall_normal /= np.linalg.norm(wall_normal)

    # Horizontal axis along the wall.
    wall_axis = np.cross(
        floor_normal,
        wall_normal
    )
    wall_axis /= np.linalg.norm(wall_axis)

    print("=" * 70)
    print("CAMERA CROSSING ANALYSIS")
    print("=" * 70)

    print("\nWall:", WALL_ID)
    print("Plane:", plane)
    print("Wall axis:", wall_axis)

    # --------------------------------------------------------
    # Load odometry
    # --------------------------------------------------------

    df = pd.read_csv(odometry_path)
    df.columns = df.columns.str.strip()

    required = ["frame", "x", "y", "z"]

    for column in required:
        if column not in df.columns:
            raise ValueError(
                f"Missing odometry column: {column}"
            )

    positions = df[["x", "y", "z"]].to_numpy(
        dtype=np.float64
    )

    frames = df["frame"].to_numpy()

    # --------------------------------------------------------
    # Signed distance to wall plane
    # --------------------------------------------------------

    signed_distance = (
        positions @ plane[:3] + plane[3]
    )

    # --------------------------------------------------------
    # Project every camera position onto wall axis.
    # --------------------------------------------------------

    relative = positions - origin

    wall_s = relative @ wall_axis
    height = relative @ floor_normal

    print("\nTrajectory:")
    print(
        "Wall coordinate:",
        f"{wall_s.min():.3f} -> {wall_s.max():.3f}"
    )

    print(
        "Signed wall distance:",
        f"{signed_distance.min():.3f} -> "
        f"{signed_distance.max():.3f}"
    )

    # --------------------------------------------------------
    # Find sign changes.
    #
    # A sign change means the camera moved from one side
    # of the wall to the other.
    # --------------------------------------------------------

    crossings = []

    for i in range(len(positions) - 1):

        d1 = signed_distance[i]
        d2 = signed_distance[i + 1]

        if d1 == 0:
            alpha = 0.0

        elif d1 * d2 < 0:
            alpha = -d1 / (d2 - d1)

        else:
            continue

        p1 = positions[i]
        p2 = positions[i + 1]

        crossing_point = (
            p1 + alpha * (p2 - p1)
        )

        crossing_relative = crossing_point - origin

        s = crossing_relative @ wall_axis
        h = crossing_relative @ floor_normal

        crossings.append({
            "frame_before": int(frames[i]),
            "frame_after": int(frames[i + 1]),
            "s_m": float(s),
            "height_m": float(h),
            "x": float(crossing_point[0]),
            "y": float(crossing_point[1]),
            "z": float(crossing_point[2]),
        })

    print("\nWall crossings:", len(crossings))

    if crossings:

        print("\nCrossing locations:")

        for crossing in crossings:

            print(
                f"  frames "
                f"{crossing['frame_before']} -> "
                f"{crossing['frame_after']} | "
                f"s={crossing['s_m']:.3f} m | "
                f"height={crossing['height_m']:.3f} m"
            )

    # --------------------------------------------------------
    # Also find positions close to wall.
    #
    # This catches cases where the camera approaches an
    # opening but doesn't numerically cross the exact plane.
    # --------------------------------------------------------

    near_mask = np.abs(signed_distance) < 0.20

    near_s = wall_s[near_mask]
    near_h = height[near_mask]

    print(
        "\nCamera positions within 20 cm of wall:",
        len(near_s)
    )

    if len(near_s):

        print(
            "Near-wall s range:",
            f"{near_s.min():.3f} -> "
            f"{near_s.max():.3f}"
        )

        print(
            "Near-wall height range:",
            f"{near_h.min():.3f} -> "
            f"{near_h.max():.3f}"
        )

    # --------------------------------------------------------
    # Save result
    # --------------------------------------------------------

    output = {
        "wall_id": WALL_ID,
        "plane": plane.tolist(),
        "wall_axis": wall_axis.tolist(),
        "crossings": crossings,
    }

    output_path = (
        Path("outputs")
        / "single_room"
        / "wall5_crossings.json"
    )

    output_path.write_text(
        json.dumps(output, indent=2)
    )

    print("\nSaved:")
    print(output_path)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
