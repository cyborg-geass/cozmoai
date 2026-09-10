from pathlib import Path

import numpy as np
import pandas as pd

from cozmo_ai.geometry.pose import (
    quaternion_to_rotation_matrix,
)


def load_odometry(
    path: str | Path,
) -> pd.DataFrame:
    """
    Load and normalize odometry.csv.
    """

    path = Path(path)

    df = pd.read_csv(path)

    # Dataset contains leading spaces in some headers.
    df.columns = df.columns.str.strip()

    required_columns = [
        "frame",
        "timestamp",
        "x",
        "y",
        "z",
        "qx",
        "qy",
        "qz",
        "qw",
        "fx",
        "fy",
        "cx",
        "cy",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing odometry columns: {missing}"
        )

    return df


def get_frame_pose(
    odometry: pd.DataFrame,
    frame_index: int,
):
    """
    Return rotation and translation for a frame.
    """

    row = odometry.iloc[frame_index]

    rotation = quaternion_to_rotation_matrix(
        row["qx"],
        row["qy"],
        row["qz"],
        row["qw"],
    )

    translation = np.array(
        [
            row["x"],
            row["y"],
            row["z"],
        ],
        dtype=np.float64,
    )

    return rotation, translation
