from pathlib import Path

import numpy as np
import pandas as pd

from cozmo_ai.geometry.calibration import (
    CameraCalibration,
)


def load_camera_calibration(
    capture_dir: str | Path,
) -> CameraCalibration:

    capture_dir = Path(capture_dir)

    camera_matrix_path = (
        capture_dir / "camera_matrix.csv"
    )

    odometry_path = (
        capture_dir / "odometry.csv"
    )

    K = np.loadtxt(
        camera_matrix_path,
        delimiter=",",
    )

    if K.shape != (3, 3):
        raise ValueError(
            f"Expected 3x3 camera matrix, "
            f"got {K.shape}"
        )

    odometry = pd.read_csv(
        odometry_path
    )

    odometry.columns = (
        odometry.columns.str.strip()
    )

    # Use the dataset's supplied camera matrix.
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    return CameraCalibration(
        rgb_width=1920,
        rgb_height=1440,
        depth_width=256,
        depth_height=192,
        fx_rgb=fx,
        fy_rgb=fy,
        cx_rgb=cx,
        cy_rgb=cy,
        depth_scale=1000.0,
    )
