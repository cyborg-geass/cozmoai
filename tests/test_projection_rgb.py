import unittest

import numpy as np

from cozmo_ai.geometry.calibration import CameraCalibration
from cozmo_ai.perception.projection import (
    BoundingBox,
    bbox_iou,
    project_camera_points_to_rgb,
    world_to_camera,
)


class ProjectionRgbTests(unittest.TestCase):
    def calibration(self):
        return CameraCalibration(
            rgb_width=100,
            rgb_height=80,
            depth_width=10,
            depth_height=8,
            fx_rgb=50,
            fy_rgb=50,
            cx_rgb=50,
            cy_rgb=40,
        )

    def test_project_camera_points_to_rgb(self):
        pixels, valid = project_camera_points_to_rgb(
            np.array(
                [
                    [0.0, 0.0, 1.0],
                    [0.5, 0.2, 1.0],
                    [0.0, 0.0, -1.0],
                ]
            ),
            self.calibration(),
        )

        self.assertTrue(valid[0])
        self.assertTrue(valid[1])
        self.assertFalse(valid[2])
        self.assertAlmostEqual(pixels[0, 0], 50.0)
        self.assertAlmostEqual(pixels[0, 1], 40.0)
        self.assertAlmostEqual(pixels[1, 0], 75.0)
        self.assertAlmostEqual(pixels[1, 1], 50.0)

    def test_world_to_camera_inverts_rotation_convention(self):
        rotation = np.eye(3)
        translation = np.array([1.0, 2.0, 3.0])
        camera = world_to_camera(
            np.array([[1.0, 2.0, 5.0]]),
            rotation,
            translation,
            convention="rotation",
        )

        np.testing.assert_allclose(
            camera,
            [[0.0, 0.0, 2.0]],
        )

    def test_bbox_iou(self):
        iou = bbox_iou(
            BoundingBox(0, 0, 10, 10),
            BoundingBox(5, 5, 15, 15),
        )

        self.assertAlmostEqual(iou, 25 / 175)


if __name__ == "__main__":
    unittest.main()
