import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.task2_baseline import model_factories, predictor_count


class Task2BaselineTests(unittest.TestCase):
    def test_required_multicollinearity_baselines_are_present(self):
        self.assertEqual(
            set(model_factories()),
            {"dummy_mean", "ordinary_least_squares", "ridge_fixed", "elastic_net_fixed", "pls_8"},
        )

    def test_dummy_uses_zero_predictors_for_adjusted_r2(self):
        self.assertEqual(predictor_count("dummy_mean", 62), 0)
        self.assertEqual(predictor_count("ridge_fixed", 62), 62)

    def test_models_are_fresh_instances(self):
        first = model_factories()
        second = model_factories()
        for name in first:
            self.assertIsNot(first[name], second[name])


if __name__ == "__main__":
    unittest.main()
