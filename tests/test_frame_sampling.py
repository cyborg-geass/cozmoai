import unittest

from cozmo_ai.perception.frame_sampling import uniform_sample_indices


class FrameSamplingTests(unittest.TestCase):
    def test_uniform_sample_indices_are_deterministic_and_bounded(self):
        indices = uniform_sample_indices(
            frame_count=100,
            target_count=5,
        )

        self.assertEqual(indices, [0, 25, 50, 74, 99])
        self.assertEqual(indices, sorted(indices))
        self.assertEqual(len(indices), 5)
        self.assertGreaterEqual(indices[0], 0)
        self.assertLess(indices[-1], 100)

    def test_uniform_sample_indices_return_all_frames_when_short(self):
        self.assertEqual(
            uniform_sample_indices(
                frame_count=3,
                target_count=8,
            ),
            [0, 1, 2],
        )

    def test_uniform_sample_indices_reject_non_positive_target(self):
        with self.assertRaises(ValueError):
            uniform_sample_indices(
                frame_count=10,
                target_count=0,
            )


if __name__ == "__main__":
    unittest.main()
