import unittest

from cozmo_ai.perception.room_classifier import aggregate_frame_probabilities
from cozmo_ai.perception.schemas import RoomSemantics


class RoomSemanticTests(unittest.TestCase):
    def test_aggregation_uses_mean_probability_and_stability(self):
        result = aggregate_frame_probabilities(
            [
                {"bedroom": 0.7, "office": 0.3},
                {"bedroom": 0.6, "office": 0.4},
                {"bedroom": 0.3, "office": 0.7},
            ]
        )

        self.assertEqual(result["label"], "bedroom")
        self.assertAlmostEqual(result["confidence"], 0.5333333333333333)
        self.assertAlmostEqual(result["stability"], 2 / 3)
        self.assertEqual(result["top_classes"][0].label, "bedroom")

    def test_empty_aggregation_returns_unknown(self):
        result = aggregate_frame_probabilities([])

        self.assertEqual(result["label"], "unknown")
        self.assertEqual(result["confidence"], 0.0)
        self.assertEqual(result["stability"], 0.0)

    def test_schema_rejects_unknown_status(self):
        semantics = RoomSemantics(
            label="unknown",
            confidence=0.0,
            stability=0.0,
            method="zero_shot_vlm",
            model="mock",
            sampled_frames=[],
            top_classes=[],
            status="mystery",
        )

        with self.assertRaises(ValueError):
            semantics.to_dict()


if __name__ == "__main__":
    unittest.main()
