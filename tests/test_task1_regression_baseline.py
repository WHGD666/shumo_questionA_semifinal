import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.task1_regression_baseline import FORBIDDEN, TARGET, model_factories


class Task1RegressionBaselineTests(unittest.TestCase):
    def test_target_is_forbidden(self):
        self.assertIn(TARGET, FORBIDDEN)

    def test_direct_sleep_fields_are_forbidden(self):
        self.assertIn("Sleep_Duration_Hours", FORBIDDEN)
        self.assertIn("Sleep_Time", FORBIDDEN)
        self.assertIn("Wake_Up_Time", FORBIDDEN)

    def test_model_factory_contains_required_baselines(self):
        self.assertEqual(set(model_factories()), {"dummy_mean", "ridge", "elastic_net", "extra_trees"})


if __name__ == "__main__":
    unittest.main()
