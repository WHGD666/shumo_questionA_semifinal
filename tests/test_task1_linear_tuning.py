import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.task1_linear_tuning import ELASTIC_ALPHAS, ELASTIC_L1_RATIOS, RIDGE_ALPHAS, model_factories
from src.task1_regression_baseline import FORBIDDEN, TARGET


class Task1LinearTuningTests(unittest.TestCase):
    def test_grid_has_expected_size(self):
        expected = len(RIDGE_ALPHAS) + len(ELASTIC_ALPHAS) * len(ELASTIC_L1_RATIOS)
        self.assertEqual(len(model_factories()), expected)

    def test_baseline_ridge_is_in_grid(self):
        self.assertIn("ridge_alpha_10p0", model_factories())

    def test_target_and_direct_sleep_fields_remain_forbidden(self):
        self.assertIn(TARGET, FORBIDDEN)
        self.assertIn("Sleep_Duration_Hours", FORBIDDEN)
        self.assertIn("Wake_Up_Time", FORBIDDEN)

    def test_model_names_are_unique(self):
        names = list(model_factories())
        self.assertEqual(len(names), len(set(names)))


if __name__ == "__main__":
    unittest.main()
