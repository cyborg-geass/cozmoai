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

    The supplied camera intrinsics may correspond to the RGB
    resolution rather than the depth resolution. Therefore we
    scale the intrinsics according to the depth image dimensions.

    Parameters
    ----------
    depth:
        H x W depth image.

    fx, fy, cx, cy:
        Camera intrinsics at the RGB/native camera resolution.

    depth_scale:
        Raw depth units per metre.

    stride:
        Pixel sampling stride.
    """

    height, width = depth.shape

    # ---------------------------------------------------------
    # The dataset's RGB resolution is 1920x1440 while depth
    # resolution is 256x192.
    #
    # Convert RGB-resolution intrinsics to depth resolution.
    # ---------------------------------------------------------

    # These are inferred from the supplied data.
    RGB_WIDTH = 1920
    RGB_HEIGHT = 1440

    scale_x = width / RGB_WIDTH
    scale_y = height / RGB_HEIGHT

    fx_depth = fx * scale_x
    fy_depth = fy * scale_y
    cx_depth = cx * scale_x
    cy_depth = cy * scale_y

    # ---------------------------------------------------------
    # Subsample depth
    # ---------------------------------------------------------

    depth_sampled = depth[::stride, ::stride]

    v, u = np.indices(depth_sampled.shape)

    u = u * stride
    v = v * stride

    z = depth_sampled.astype(np.float64) / depth_scale

    valid = np.isfinite(z) & (z > 0)

    u = u[valid]
    v = v[valid]
    z = z[valid]

    # ---------------------------------------------------------
    # Backproject
    # ---------------------------------------------------------

    x = (u - cx_depth) * z / fx_depth
    y = (v - cy_depth) * z / fy_depth

    points = np.column_stack((x, y, z))

    return points
