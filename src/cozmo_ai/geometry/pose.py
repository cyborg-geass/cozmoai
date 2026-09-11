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
    convention: str = "rotation_transpose",
) -> np.ndarray:
    """
    Transform camera-frame points into world coordinates.

    Points are stored as row vectors, so the matrix expression is the
    transpose of the equivalent column-vector notation.

    Supported conventions:

    - ``rotation``: row-vector form of ``P_world = R @ P_camera + t``.
    - ``rotation_transpose``: row-vector form of
      ``P_world = R.T @ P_camera + t``.
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

    if convention == "rotation":
        return (
            points_camera @ rotation.T
            + translation
        )

    if convention == "rotation_transpose":
        return (
            points_camera @ rotation
            + translation
        )

    raise ValueError(
        "Unsupported pose convention: "
        f"{convention!r}"
    )
