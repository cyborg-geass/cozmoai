import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


MIN_FRAME_SEPARATION = 300

MAX_POSITION_DISTANCE = 0.75

MAX_ROTATION_DISTANCE_DEG = 20.0

MAX_PAIRS = 20


def load_odometry(path):

    df = pd.read_csv(path)

    df.columns = df.columns.str.strip()

    required = [
        "frame",
        "x",
        "y",
        "z",
        "qx",
        "qy",
        "qz",
        "qw",
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    return df


def get_rotation(row):

    q = np.array([
        row["qx"],
        row["qy"],
        row["qz"],
        row["qw"],
    ])

    return Rotation.from_quat(q)


def rotation_difference_deg(r1, r2):

    relative = r1.inv() * r2

    angle = relative.magnitude()

    return np.degrees(angle)


def main():

    if len(sys.argv) != 2:

        print(
            "Usage:\n"
            'uv run python scripts\\find_good_revisits.py '
            '"..\\cozmo-dataset\\raw_dataset\\single_room\\c00a170fe1"'
        )

        sys.exit(1)

    capture_dir = Path(sys.argv[1])

    odometry_path = (
        capture_dir / "odometry.csv"
    )

    df = load_odometry(
        odometry_path
    )

    n = len(df)

    positions = df[
        ["x", "y", "z"]
    ].to_numpy()

    rotations = [
        get_rotation(df.iloc[i])
        for i in range(n)
    ]

    candidates = []

    print("=" * 70)
    print("FINDING GOOD REVISIT PAIRS")
    print("=" * 70)

    print(
        f"\nFrames: {n}"
    )

    print(
        f"Minimum frame separation: "
        f"{MIN_FRAME_SEPARATION}"
    )

    print(
        f"Maximum position distance: "
        f"{MAX_POSITION_DISTANCE} m"
    )

    print(
        f"Maximum orientation difference: "
        f"{MAX_ROTATION_DISTANCE_DEG} degrees"
    )

    for i in range(n):

        for j in range(
            i + MIN_FRAME_SEPARATION,
            n,
            10,
        ):

            position_distance = np.linalg.norm(
                positions[i] - positions[j]
            )

            if (
                position_distance
                > MAX_POSITION_DISTANCE
            ):
                continue

            rotation_distance = (
                rotation_difference_deg(
                    rotations[i],
                    rotations[j],
                )
            )

            if (
                rotation_distance
                > MAX_ROTATION_DISTANCE_DEG
            ):
                continue

            candidates.append(
                (
                    position_distance,
                    rotation_distance,
                    i,
                    j,
                )
            )

    candidates.sort(
        key=lambda x: (
            x[0],
            x[1],
        )
    )

    print(
        f"\nFound {len(candidates)} "
        f"candidate pairs."
    )

    print(
        "\nBest revisit pairs:"
    )

    print(
        "\n"
        "Frame A    Frame B    "
        "Position Δ    Rotation Δ"
    )

    print(
        "-" * 55
    )

    for (
        position_distance,
        rotation_distance,
        i,
        j,
    ) in candidates[:MAX_PAIRS]:

        print(
            f"{i:8d}    "
            f"{j:8d}    "
            f"{position_distance:8.3f} m    "
            f"{rotation_distance:8.2f}°"
        )

    if not candidates:

        print(
            "\nNo good revisit pairs were found."
        )

        print(
            "\nTry relaxing:"
        )

        print(
            "MAX_ROTATION_DISTANCE_DEG "
            "from 20 → 30 or 40."
        )


if __name__ == "__main__":
    main()
