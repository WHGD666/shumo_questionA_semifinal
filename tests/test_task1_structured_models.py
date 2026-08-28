import unittest
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.task1_structured_models import candidate_names, prepare_native_catboost


class Task1StructuredModelTests(unittest.TestCase):
    def test_candidate_family_is_fixed(self):
        self.assertEqual(
            set(candidate_names()),
            {"pls_8", "pls_16", "spline_ridge", "catboost_native"},
        )

    def test_native_catboost_preprocessing_is_fold_local(self):
        train = pd.DataFrame({"num": [1.0, None, 3.0], "cat": ["A", None, "B"]})
        valid = pd.DataFrame({"num": [None], "cat": [None]})
        train_out, valid_out, indices = prepare_native_catboost(train, valid)
        self.assertEqual(float(valid_out.loc[0, "num"]), 2.0)
        self.assertEqual(valid_out.loc[0, "cat"], "Missing_Unknown")
        self.assertEqual(indices, [1])


if __name__ == "__main__":
    unittest.main()
