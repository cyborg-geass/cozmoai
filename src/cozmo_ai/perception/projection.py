from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cozmo_ai.geometry.calibration import CameraCalibration


@dataclass(frozen=True)
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    def clipped(
        self,
        *,
        width: int,
        height: int,
    ) -> "BoundingBox":
        return BoundingBox(
            x1=max(0.0, min(float(width), self.x1)),
            y1=max(0.0, min(float(height), self.y1)),
            x2=max(0.0, min(float(width), self.x2)),
            y2=max(0.0, min(float(height), self.y2)),
        )

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    def to_xyxy(self) -> list[float]:
        return [
            float(self.x1),
            float(self.y1),
            float(self.x2),
            float(self.y2),
        ]


def world_to_camera(
    points_world: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    *,
    convention: str = "rotation",
) -> np.ndarray:
    points_world = np.asarray(
        points_world,
        dtype=np.float64,
    )
    rotation = np.asarray(
        rotation,
        dtype=np.float64,
    )
    translation = np.asarray(
        translation,
        dtype=np.float64,
    )

    if points_world.ndim != 2 or points_world.shape[1] != 3:
        raise ValueError("points_world must have shape Nx3")
    if rotation.shape != (3, 3):
        raise ValueError("rotation must have shape 3x3")
    if translation.shape != (3,):
        raise ValueError("translation must have shape 3")

    centered = points_world - translation

    if convention == "rotation":
        return centered @ rotation

    if convention == "rotation_transpose":
        return centered @ rotation.T

    raise ValueError(f"Unsupported pose convention: {convention!r}")


def project_camera_points_to_rgb(
    points_camera: np.ndarray,
    calibration: CameraCalibration,
) -> tuple[np.ndarray, np.ndarray]:
    points_camera = np.asarray(
        points_camera,
        dtype=np.float64,
    )

    if points_camera.ndim != 2 or points_camera.shape[1] != 3:
        raise ValueError("points_camera must have shape Nx3")

    z = points_camera[:, 2]
    valid = z > 1e-6
    pixels = np.full(
        (len(points_camera), 2),
        np.nan,
        dtype=np.float64,
    )
    pixels[valid, 0] = (
        points_camera[valid, 0] * calibration.fx_rgb / z[valid]
        + calibration.cx_rgb
    )
    pixels[valid, 1] = (
        points_camera[valid, 1] * calibration.fy_rgb / z[valid]
        + calibration.cy_rgb
    )
    valid &= (
        (pixels[:, 0] >= 0)
        & (pixels[:, 0] <= calibration.rgb_width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] <= calibration.rgb_height)
    )

    return pixels, valid


def projected_bbox(
    points_world: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    calibration: CameraCalibration,
    *,
    convention: str = "rotation",
) -> BoundingBox | None:
    points_camera = world_to_camera(
        points_world,
        rotation,
        translation,
        convention=convention,
    )
    pixels, valid = project_camera_points_to_rgb(
        points_camera,
        calibration,
    )

    visible = pixels[valid]
    if len(visible) == 0:
        return None

    return BoundingBox(
        x1=float(np.min(visible[:, 0])),
        y1=float(np.min(visible[:, 1])),
        x2=float(np.max(visible[:, 0])),
        y2=float(np.max(visible[:, 1])),
    ).clipped(
        width=calibration.rgb_width,
        height=calibration.rgb_height,
    )


def bbox_iou(a: BoundingBox, b: BoundingBox) -> float:
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)

    intersection = BoundingBox(
        x1,
        y1,
        x2,
        y2,
    ).area
    union = a.area + b.area - intersection

    if union <= 0:
        return 0.0

    return float(intersection / union)
