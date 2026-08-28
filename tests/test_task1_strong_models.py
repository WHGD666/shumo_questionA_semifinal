import unittest
from pathlib import Path
import sys
from sklearn.base import is_regressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.task1_regression_baseline import FORBIDDEN, TARGET
from src.task1_strong_models import model_factories


class Task1StrongModelTests(unittest.TestCase):
    def test_factory_has_fixed_candidates(self):
        self.assertEqual(set(model_factories()), {"lightgbm_fixed", "catboost_fixed"})

    def test_models_are_regressors(self):
        for model in model_factories().values():
            self.assertTrue(is_regressor(model))

    def test_feature_contract_is_unchanged(self):
        self.assertIn(TARGET, FORBIDDEN)
        self.assertIn("Sleep_Duration_Hours", FORBIDDEN)
        self.assertIn("Sleep_Time", FORBIDDEN)
        self.assertIn("Wake_Up_Time", FORBIDDEN)


if __name__ == "__main__":
    unittest.main()
