import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_trajectory(capture_dir):

    path = Path(capture_dir) / "odometry.csv"

    odom = pd.read_csv(path)

    odom.columns = [
        c.strip()
        for c in odom.columns
    ]

    return odom[
        ["x", "y", "z"]
    ].to_numpy(dtype=float)


def main():

    if len(sys.argv) != 3:

        print(
            "Usage:\n"
            "uv run python scripts\\inspect_room_envelope.py "
            "outputs\\single_room\\room_envelope.json "
            "..\\cozmo-dataset\\raw_dataset\\single_room\\c00a170fe1"
        )

        sys.exit(1)

    envelope_path = Path(sys.argv[1])
    capture_dir = Path(sys.argv[2])

    with open(
        envelope_path,
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    corners = np.array(
        [
            [p["u"], p["v"]]
            for p in data["corners"]
        ],
        dtype=float,
    )

    trajectory = load_trajectory(
        capture_dir
    )

    frame = data[
        "floor_coordinate_system"
    ]

    origin = np.array(
        frame["origin"],
        dtype=float,
    )

    u = np.array(
        frame["u_axis"],
        dtype=float,
    )

    v = np.array(
        frame["v_axis"],
        dtype=float,
    )

    relative = (
        trajectory - origin
    )

    trajectory_uv = np.column_stack(
        [
            relative @ u,
            relative @ v,
        ]
    )

    # --------------------------------------------------------
    # Print bounds
    # --------------------------------------------------------

    print("=" * 70)
    print("ROOM ENVELOPE INSPECTION")
    print("=" * 70)

    print(
        "\nRoom bounds:"
    )

    print(
        f"  U: "
        f"{corners[:, 0].min():.3f}"
        f" -> "
        f"{corners[:, 0].max():.3f}"
    )

    print(
        f"  V: "
        f"{corners[:, 1].min():.3f}"
        f" -> "
        f"{corners[:, 1].max():.3f}"
    )

    print(
        "\nTrajectory bounds:"
    )

    print(
        f"  U: "
        f"{trajectory_uv[:, 0].min():.3f}"
        f" -> "
        f"{trajectory_uv[:, 0].max():.3f}"
    )

    print(
        f"  V: "
        f"{trajectory_uv[:, 1].min():.3f}"
        f" -> "
        f"{trajectory_uv[:, 1].max():.3f}"
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    closed = np.vstack(
        [
            corners,
            corners[0],
        ]
    )

    plt.figure(
        figsize=(12, 8)
    )

    plt.plot(
        closed[:, 0],
        closed[:, 1],
        linewidth=3,
        marker="o",
        label="Selected room boundary",
    )

    plt.plot(
        trajectory_uv[:, 0],
        trajectory_uv[:, 1],
        linewidth=1,
        label="Camera trajectory",
    )

    # Start/end markers
    plt.scatter(
        trajectory_uv[0, 0],
        trajectory_uv[0, 1],
        s=80,
        label="Start",
    )

    plt.scatter(
        trajectory_uv[-1, 0],
        trajectory_uv[-1, 1],
        s=80,
        label="End",
    )

    for i, point in enumerate(
        corners,
        start=1,
    ):

        plt.annotate(
            f"C{i}",
            point,
            xytext=(5, 5),
            textcoords="offset points",
        )

    plt.xlabel(
        "Floor U (m)"
    )

    plt.ylabel(
        "Floor V (m)"
    )

    plt.title(
        "Room Boundary vs Camera Trajectory"
    )

    plt.axis(
        "equal"
    )

    plt.grid(
        True
    )

    plt.legend()

    output = (
        envelope_path.parent
        / "room_envelope_inspection.png"
    )

    plt.savefig(
        output,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"\nSaved:"
        f"\n{output}"
    )


if __name__ == "__main__":
    main()
