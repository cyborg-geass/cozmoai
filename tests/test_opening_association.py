import unittest

from cozmo_ai.perception.opening_semantics import (
    Detection2D,
    best_associated_detection,
)
from cozmo_ai.perception.projection import BoundingBox


class OpeningAssociationTests(unittest.TestCase):
    def test_accepts_best_detection_above_thresholds(self):
        detection, iou = best_associated_detection(
            geometry_bbox=BoundingBox(0, 0, 100, 100),
            detections=[
                Detection2D("door", 0.9, [10, 10, 90, 90], "mock"),
                Detection2D("window", 0.8, [200, 200, 260, 260], "mock"),
            ],
            min_iou=0.5,
            min_confidence=0.5,
        )

        self.assertIsNotNone(detection)
        self.assertEqual(detection.class_name, "door")
        self.assertGreaterEqual(iou, 0.5)

    def test_rejects_low_iou_or_low_confidence(self):
        detection, iou = best_associated_detection(
            geometry_bbox=BoundingBox(0, 0, 100, 100),
            detections=[
                Detection2D("door", 0.1, [0, 0, 100, 100], "mock"),
                Detection2D("window", 0.9, [150, 150, 200, 200], "mock"),
            ],
            min_iou=0.5,
            min_confidence=0.5,
        )

        self.assertIsNone(detection)
        self.assertEqual(iou, 0.0)


if __name__ == "__main__":
    unittest.main()
