from __future__ import annotations

import numpy as np


def depth_to_points(
    depth: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    depth_scale: float = 1000.0,
    stride: int = 4,
) -> np.ndarray:
    """
    Convert a depth image into 3D points in camera coordinates.

    Parameters
    ----------
    depth:
        H x W depth image.

    fx, fy, cx, cy:
        Camera intrinsic parameters.

    depth_scale:
        Number of raw depth units per metre.
        1000 means depth is stored in millimetres.

    stride:
        Pixel sampling stride.

    Returns
    -------
    points:
        N x 3 array containing XYZ points in camera coordinates.
    """

    depth = depth[::stride, ::stride]

    height, width = depth.shape

    v, u = np.indices((height, width))

    # Convert sampled pixel coordinates back to
    # coordinates in the original depth image.
    u = u * stride
    v = v * stride

    z = depth.astype(np.float64) / depth_scale

    valid = np.isfinite(z) & (z > 0)

    u = u[valid]
    v = v[valid]
    z = z[valid]

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    points = np.column_stack((x, y, z))

    return points
