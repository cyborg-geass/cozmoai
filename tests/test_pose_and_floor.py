import unittest
from pathlib import Path

import numpy as np

from cozmo_ai.geometry.pose import camera_to_world
from scripts.build_pointcloud import resolve_pose_convention
from scripts.detect_floor import select_floor_candidate


class PoseConventionTests(unittest.TestCase):
    def test_camera_to_world_rotation_convention(self):
        points = np.array([[1.0, 0.0, 0.0]])
        rotation = np.array(
            [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        translation = np.array([10.0, 20.0, 30.0])

        transformed = camera_to_world(
            points,
            rotation,
            translation,
            convention="rotation",
        )

        np.testing.assert_allclose(
            transformed,
            np.array([[10.0, 21.0, 30.0]]),
        )

    def test_camera_to_world_rotation_transpose_convention(self):
        points = np.array([[1.0, 0.0, 0.0]])
        rotation = np.array(
            [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        translation = np.array([10.0, 20.0, 30.0])

        transformed = camera_to_world(
            points,
            rotation,
            translation,
            convention="rotation_transpose",
        )

        np.testing.assert_allclose(
            transformed,
            np.array([[10.0, 19.0, 30.0]]),
        )

    def test_single_room_uses_validated_rotation_convention(self):
        convention = resolve_pose_convention(
            Path("cozmo-dataset")
            / "raw_dataset"
            / "single_room"
            / "c00a170fe1"
        )

        self.assertEqual(convention, "rotation")

    def test_ceiling_capture_keeps_transpose_convention(self):
        convention = resolve_pose_convention(
            Path("cozmo-dataset")
            / "raw_dataset"
            / "single_scan_with_ceiling"
            / "c7d28f72c6"
        )

        self.assertEqual(convention, "rotation_transpose")


class FloorCandidateSelectionTests(unittest.TestCase):
    def candidate(self, *, y, points, area, normal_y=1.0):
        return {
            "normal": np.array([0.0, normal_y, 0.0]),
            "points": points,
            "footprint_area_m2": area,
            "median_y": y,
        }

    def test_selects_lowest_large_horizontal_surface(self):
        floor = self.candidate(y=-1.45, points=58_000, area=40.0)
        ceiling = self.candidate(y=1.55, points=62_000, area=38.0)
        shelf = self.candidate(y=-1.8, points=8_000, area=2.0)

        selected, horizontal = select_floor_candidate(
            [ceiling, shelf, floor]
        )

        self.assertIs(selected, floor)
        self.assertEqual(len(horizontal), 3)

    def test_rejects_non_horizontal_candidates(self):
        selected, horizontal = select_floor_candidate(
            [
                self.candidate(
                    y=-1.45,
                    points=80_000,
                    area=50.0,
                    normal_y=0.2,
                )
            ]
        )

        self.assertIsNone(selected)
        self.assertEqual(horizontal, [])


if __name__ == "__main__":
    unittest.main()
