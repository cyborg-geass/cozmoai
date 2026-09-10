from pathlib import Path

import cv2
import numpy as np
import pandas as pd


CAPTURE = Path(r"..\cozmo-dataset\raw_dataset\single_room\c00a170fe1")


def inspect_depth():
    path = CAPTURE / "depth" / "000000.png"

    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    print("=" * 70)
    print("DEPTH")
    print("=" * 70)

    print("Path:", path)
    print("Shape:", depth.shape)
    print("dtype:", depth.dtype)

    valid = depth > 0

    print("Valid pixels:", valid.sum())
    print("Valid percentage:", valid.mean() * 100)

    values = depth[valid]

    print("Min:", values.min())
    print("Max:", values.max())
    print("Mean:", values.mean())
    print("Median:", np.median(values))

    print()


def inspect_confidence():
    path = CAPTURE / "confidence" / "000000.png"

    confidence = cv2.imread(
        str(path),
        cv2.IMREAD_UNCHANGED
    )

    print("=" * 70)
    print("CONFIDENCE")
    print("=" * 70)

    print("Shape:", confidence.shape)
    print("dtype:", confidence.dtype)

    values, counts = np.unique(
        confidence,
        return_counts=True
    )

    print("Values:")

    for value, count in zip(values, counts):
        print(
            f"  {value}: {count} "
            f"({count / confidence.size * 100:.2f}%)"
        )

    print()


def inspect_camera_matrix():
    path = CAPTURE / "camera_matrix.csv"

    K = np.loadtxt(
        path,
        delimiter=","
    )

    print("=" * 70)
    print("CAMERA MATRIX")
    print("=" * 70)

    print(K)
    print("Shape:", K.shape)

    print()


def inspect_odometry():
    path = CAPTURE / "odometry.csv"

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    print("=" * 70)
    print("ODOMETRY")
    print("=" * 70)

    print("Rows:", len(df))

    print("\nFirst pose:")
    print(df.iloc[0][
        [
            "timestamp",
            "frame",
            "x",
            "y",
            "z",
            "qx",
            "qy",
            "qz",
            "qw"
        ]
    ])

    print("\nLast pose:")
    print(df.iloc[-1][
        [
            "timestamp",
            "frame",
            "x",
            "y",
            "z",
            "qx",
            "qy",
            "qz",
            "qw"
        ]
    ])

    positions = df[["x", "y", "z"]].to_numpy()

    displacement = np.linalg.norm(
        positions[-1] - positions[0]
    )

    print("\nStart position:", positions[0])
    print("End position:", positions[-1])
    print("Start-to-end displacement:", displacement)

    print("\nPosition ranges:")

    for axis in ["x", "y", "z"]:
        print(
            f"{axis}: "
            f"{df[axis].min():.3f} → "
            f"{df[axis].max():.3f}"
        )


def inspect_video():
    path = CAPTURE / "rgb.mp4"

    cap = cv2.VideoCapture(str(path))

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print()
    print("=" * 70)
    print("VIDEO")
    print("=" * 70)

    print("FPS:", fps)
    print("Frames:", frames)

    cap.release()


if __name__ == "__main__":
    inspect_depth()
    inspect_confidence()
    inspect_camera_matrix()
    inspect_odometry()
    inspect_video()
