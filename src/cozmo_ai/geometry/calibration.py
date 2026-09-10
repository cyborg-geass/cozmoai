from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraCalibration:
    """
    Camera/depth calibration for one capture.
    """

    rgb_width: int
    rgb_height: int

    depth_width: int
    depth_height: int

    fx_rgb: float
    fy_rgb: float
    cx_rgb: float
    cy_rgb: float

    depth_scale: float = 1000.0

    @property
    def scale_x(self) -> float:
        return self.depth_width / self.rgb_width

    @property
    def scale_y(self) -> float:
        return self.depth_height / self.rgb_height

    @property
    def fx_depth(self) -> float:
        return self.fx_rgb * self.scale_x

    @property
    def fy_depth(self) -> float:
        return self.fy_rgb * self.scale_y

    @property
    def cx_depth(self) -> float:
        return self.cx_rgb * self.scale_x

    @property
    def cy_depth(self) -> float:
        return self.cy_rgb * self.scale_y

    @property
    def depth_intrinsics(self) -> np.ndarray:
        return np.array(
            [
                [self.fx_depth, 0.0, self.cx_depth],
                [0.0, self.fy_depth, self.cy_depth],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
