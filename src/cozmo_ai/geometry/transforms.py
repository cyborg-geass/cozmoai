from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

def quaternion_to_rotation_matrix(
    qx: float,
    qy: float,
    qz: float,
    qw: float,
) -> np.ndarray:
    """
    Convert quaternion (qx, qy, qz, qw)
    into a 3x3 rotation matrix.
    """

    q = np.array(
        [qx, qy, qz, qw],
        dtype=np.float64,
    )

    # Normalize defensively.
    norm = np.linalg.norm(q)

    if norm == 0:
        raise ValueError("Invalid zero quaternion.")

    q /= norm

    return Rotation.from_quat(q).as_matrix()


def transform_points(
    points: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    """
    Transform points from one coordinate frame to another.

    P' = R P + t
    """

    return points @ rotation.T + translation
