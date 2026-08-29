import unittest

from src.task2_elastic_tuning import (
    ALPHAS,
    L1_RATIOS,
    candidate_grid,
    candidate_name,
    make_candidate,
)


class Task2ElasticTuningTests(unittest.TestCase):
    def test_grid_is_bounded_unique_and_contains_control(self):
        grid = candidate_grid()
        self.assertEqual(len(grid), len(ALPHAS) * len(L1_RATIOS))
        self.assertEqual(len({item["model"] for item in grid}), len(grid))
        self.assertIn(
            {"model": candidate_name(0.001, 0.5), "alpha": 0.001, "l1_ratio": 0.5},
            grid,
        )

    def test_candidate_factory_preserves_requested_parameters(self):
        model = make_candidate(alpha=0.003, l1_ratio=0.7, seed=2026)
        self.assertEqual(model.alpha, 0.003)
        self.assertEqual(model.l1_ratio, 0.7)
        self.assertEqual(model.max_iter, 20000)


if __name__ == "__main__":
    unittest.main()
