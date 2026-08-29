import unittest

from src.task3_linear_tuning import (
    ELASTIC_ALPHAS,
    L1_RATIOS,
    RIDGE_ALPHAS,
    candidate_grid,
    elastic_name,
    make_candidate,
    ridge_name,
)


class Task3LinearTuningTests(unittest.TestCase):
    def test_grid_is_bounded_unique_and_contains_baseline_control(self):
        grid = candidate_grid()
        expected_count = len(RIDGE_ALPHAS) + len(ELASTIC_ALPHAS) * len(L1_RATIOS)
        self.assertEqual(len(grid), expected_count)
        self.assertEqual(len({item["model"] for item in grid}), expected_count)
        self.assertIn(ridge_name(10.0), {item["model"] for item in grid})
        self.assertIn(elastic_name(0.001, 0.5), {item["model"] for item in grid})

    def test_ridge_factory_preserves_requested_alpha(self):
        item = {"family": "ridge", "model": ridge_name(3.0), "alpha": 3.0, "l1_ratio": float("nan")}
        model = make_candidate(item, seed=2026)
        self.assertEqual(model.alpha, 3.0)

    def test_elastic_factory_preserves_requested_parameters(self):
        item = {
            "family": "elastic_net",
            "model": elastic_name(0.003, 0.7),
            "alpha": 0.003,
            "l1_ratio": 0.7,
        }
        model = make_candidate(item, seed=2026)
        self.assertEqual(model.alpha, 0.003)
        self.assertEqual(model.l1_ratio, 0.7)
        self.assertEqual(model.max_iter, 20000)

    def test_unknown_family_is_rejected(self):
        with self.assertRaises(ValueError):
            make_candidate(
                {"family": "unknown", "model": "bad", "alpha": 1.0, "l1_ratio": 0.5},
                seed=2026,
            )


if __name__ == "__main__":
    unittest.main()
