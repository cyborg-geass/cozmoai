import contextlib
import json
import shutil
import unittest
from pathlib import Path

from scripts.finalize_pipeline import (
    MIN_POINTCLOUD_BYTES,
    PipelineValidationError,
    REPO,
    validate_capture,
    validate_file,
    validate_measurements,
    validate_openings,
    validate_room_envelope,
    validate_walls,
)


class FinalizePipelineValidationTests(unittest.TestCase):
    @contextlib.contextmanager
    def temp_dir(self):
        root = REPO / "outputs" / "test_validation_fixtures"
        path = root / self._testMethodName
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(
            parents=True,
        )
        try:
            yield path
        finally:
            shutil.rmtree(
                path,
                ignore_errors=True,
            )

    def write_json(self, path: Path, payload: dict) -> Path:
        path.write_text(
            json.dumps(payload),
            encoding="utf-8",
        )
        return path

    def test_validate_capture_requires_matching_depth_confidence_frames(self):
        with self.temp_dir() as directory:
            capture = Path(directory)

            for name in (
                "camera_matrix.csv",
                "odometry.csv",
                "imu.csv",
                "rgb.mp4",
            ):
                (capture / name).write_text(
                    "placeholder",
                    encoding="utf-8",
                )

            (capture / "depth").mkdir()
            (capture / "confidence").mkdir()
            (capture / "depth" / "000000.png").write_bytes(b"png")

            with self.assertRaises(PipelineValidationError):
                validate_capture(capture)

            (capture / "confidence" / "000000.png").write_bytes(b"png")

            summary = validate_capture(capture)

            self.assertEqual(summary["status"], "validated")
            self.assertEqual(
                summary["directories"]["depth"]["png_count"],
                1,
            )

    def test_validate_file_rejects_tiny_pointcloud(self):
        with self.temp_dir() as directory:
            path = Path(directory) / "pointcloud_production.ply"
            path.write_bytes(b"ply")

            with self.assertRaises(PipelineValidationError):
                validate_file(path, MIN_POINTCLOUD_BYTES)

    def test_validate_walls_requires_four_or_more_walls(self):
        with self.temp_dir() as directory:
            path = self.write_json(
                Path(directory) / "walls.json",
                {
                    "walls": [
                        {
                            "wall_id": 1,
                            "plane": [1, 0, 0, 0],
                            "saved_points": 10,
                        }
                    ]
                },
            )

            with self.assertRaises(PipelineValidationError):
                validate_walls(path)

    def test_validate_room_envelope_reports_measurement_summary(self):
        with self.temp_dir() as directory:
            path = self.write_json(
                Path(directory) / "room_envelope.json",
                {
                    "dimensions_m": {
                        "dimension_a": 2.0,
                        "dimension_b": 6.0,
                    },
                    "area_m2": 12.0,
                    "trajectory_inside_ratio": 0.75,
                    "selected_wall_pairs": [
                        {"walls": [1, 2]},
                        {"walls": [3, 4]},
                    ],
                },
            )

            summary = validate_room_envelope(path)

            self.assertEqual(summary["status"], "validated")
            self.assertEqual(summary["area_m2"], 12.0)

    def test_validate_measurements_requires_dimension_intervals(self):
        with self.temp_dir() as directory:
            path = self.write_json(
                Path(directory) / "measurement_evaluation.json",
                {
                    "floor": {
                        "point_count": 25_000,
                        "residuals": {
                            "p95_m": 0.02,
                        },
                    },
                    "room": {
                        "dimensions": {
                            "dimension_a": {
                                "value_m": 2.0,
                                "uncertainty": {
                                    "status": "model_based_uncertainty",
                                    "ci95_m": [1.9, 2.1],
                                },
                            },
                            "dimension_b": {
                                "value_m": 6.0,
                                "uncertainty": {
                                    "status": "model_based_uncertainty",
                                    "ci95_m": [5.9, 6.1],
                                },
                            },
                        },
                        "area": {
                            "reconstructed_footprint": {
                                "value_m2": 12.0,
                            }
                        },
                    },
                },
            )

            summary = validate_measurements(path)

            self.assertEqual(summary["status"], "validated")
            self.assertEqual(summary["dimension_b_m"], 6.0)

    def test_validate_openings_allows_null_not_observed_heights(self):
        with self.temp_dir() as directory:
            path = self.write_json(
                Path(directory) / "openings_wall5.json",
                {
                    "candidates": [
                        {
                            "width_m": 0.5,
                            "height_m": None,
                            "height_status": "not_observed",
                        }
                    ]
                },
            )

            summary = validate_openings(path)

            self.assertEqual(summary["status"], "validated")
            self.assertEqual(summary["candidate_count"], 1)


if __name__ == "__main__":
    unittest.main()
