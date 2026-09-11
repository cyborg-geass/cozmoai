import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def main():

    if len(sys.argv) != 2:
        print(
            "Usage:\n"
            'uv run python scripts\\inspect_depth_calibration.py '
            '"..\\cozmo-dataset\\raw_dataset\\single_room\\c00a170fe1"'
        )
        sys.exit(1)

    capture_dir = Path(sys.argv[1])

    depth_dir = capture_dir / "depth"
    camera_matrix_path = capture_dir / "camera_matrix.csv"
    odometry_path = capture_dir / "odometry.csv"

    print("=" * 70)
    print("CALIBRATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Camera matrix
    # --------------------------------------------------------

    K = np.loadtxt(camera_matrix_path, delimiter=",")

    print("\nCamera matrix:")
    print(K)

    print(f"\nCamera matrix shape: {K.shape}")

    # --------------------------------------------------------
    # Odometry intrinsics
    # --------------------------------------------------------

    df = pd.read_csv(odometry_path)
    df.columns = df.columns.str.strip()

    print("\nOdometry intrinsic statistics:")

    cols = ["fx", "fy", "cx", "cy"]

    print(
        df[cols].describe()
    )

    # --------------------------------------------------------
    # Depth files
    # --------------------------------------------------------

    depth_files = sorted(depth_dir.glob("*.png"))

    print("\n" + "=" * 70)
    print("DEPTH DATA")
    print("=" * 70)

    print(f"\nDepth files: {len(depth_files)}")

    indices = [
        0,
        len(depth_files) // 3,
        (2 * len(depth_files)) // 3,
        len(depth_files) - 1,
    ]

    for idx in indices:

        path = depth_files[idx]

        depth = cv2.imread(
            str(path),
            cv2.IMREAD_UNCHANGED,
        )

        if depth is None:
            print(f"\nCould not read {path}")
            continue

        values = depth.astype(np.float64).ravel()

        values = values[np.isfinite(values)]
        values = values[values > 0]

        print("\n" + "-" * 60)
        print(f"Frame: {idx}")
        print(f"File:  {path.name}")
        print(f"Shape: {depth.shape}")
        print(f"Dtype: {depth.dtype}")

        print(f"Raw min:    {values.min():.2f}")
        print(f"Raw max:    {values.max():.2f}")
        print(f"Raw mean:   {values.mean():.2f}")
        print(f"Raw median: {np.median(values):.2f}")

        print("\nPercentiles:")

        for p in [1, 5, 25, 50, 75, 95, 99]:
            print(
                f"  P{p:02d}: "
                f"{np.percentile(values, p):.2f}"
            )

        print("\nCandidate physical depths:")

        for scale in [500, 1000, 2000]:
            physical = values / scale

            print(
                f"  scale={scale:4d}: "
                f"median={np.median(physical):.3f} m, "
                f"P95={np.percentile(physical, 95):.3f} m"
            )

    # --------------------------------------------------------
    # Frame correspondence
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FRAME CORRESPONDENCE")
    print("=" * 70)

    print(
        "\nFirst odometry frames:"
    )

    print(
        df[
            ["frame", "timestamp", "x", "y", "z"]
        ].head(10).to_string(index=False)
    )

    print(
        "\nLast odometry frames:"
    )

    print(
        df[
            ["frame", "timestamp", "x", "y", "z"]
        ].tail(10).to_string(index=False)
    )


if __name__ == "__main__":
    main()
