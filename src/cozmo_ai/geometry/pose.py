import numpy as np

from scipy.spatial.transform import Rotation


def quaternion_to_rotation_matrix(
    qx,
    qy,
    qz,
    qw,
):
    return Rotation.from_quat(
        np.array(
            [qx, qy, qz, qw],
            dtype=np.float64,
        )
    ).as_matrix()


def camera_to_world(
    points_camera: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    """
    Transform camera-frame points into world coordinates.

    Dataset convention selected from diagnostic evaluation:

        P_world = R @ P_camera + t
    """

    if points_camera.ndim != 2:
        raise ValueError(
            "points_camera must be Nx3"
        )

    if points_camera.shape[1] != 3:
        raise ValueError(
            "points_camera must have shape Nx3"
        )

    rotation = np.asarray(
        rotation,
        dtype=np.float64,
    )

    translation = np.asarray(
        translation,
        dtype=np.float64,
    )

    if rotation.shape != (3, 3):
        raise ValueError(
            "rotation must have shape (3, 3)"
        )

    if translation.shape != (3,):
        raise ValueError(
            "translation must have shape (3,)"
        )

    return (
        points_camera @ rotation
        + translation
    )
