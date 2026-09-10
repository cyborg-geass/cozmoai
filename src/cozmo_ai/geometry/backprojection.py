import numpy as np
from .calibration import CameraCalibration


def depth_to_camera_points(
    depth: np.ndarray,
    calibration: CameraCalibration,
    pixel_stride: int = 1,
) -> np.ndarray:
    """
    Backproject a depth image into 3D camera coordinates.

    Parameters
    ----------
    depth:
        H x W uint16 depth image.

    calibration:
        Camera/depth calibration.

    pixel_stride:
        Sample every Nth pixel in both dimensions.

    Returns
    -------
    np.ndarray
        Nx3 array containing points in camera coordinates.
        Coordinates are expressed in meters.
    """

    if depth.ndim != 2:
        raise ValueError(
            f"Expected a single-channel depth image, "
            f"got shape {depth.shape}"
        )

    expected_shape = (
        calibration.depth_height,
        calibration.depth_width,
    )

    if depth.shape != expected_shape:
        raise ValueError(
            f"Depth shape {depth.shape} does not match "
            f"calibration shape {expected_shape}"
        )

    if pixel_stride < 1:
        raise ValueError(
            "pixel_stride must be >= 1"
        )

    fx = calibration.fx_depth
    fy = calibration.fy_depth
    cx = calibration.cx_depth
    cy = calibration.cy_depth

    # Sample pixel coordinates.
    v, u = np.indices(depth.shape)

    u = u[::pixel_stride, ::pixel_stride]
    v = v[::pixel_stride, ::pixel_stride]

    depth_sampled = depth[
        ::pixel_stride,
        ::pixel_stride,
    ]

    # Convert raw depth units to meters.
    z = (
        depth_sampled.astype(np.float64)
        / calibration.depth_scale
    )

    # Valid depth.
    valid = (
        np.isfinite(z)
        & (z > 0.0)
    )

    u = u[valid].astype(np.float64)
    v = v[valid].astype(np.float64)
    z = z[valid]

    # Pinhole backprojection.
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    return np.column_stack(
        (x, y, z)
    )
