import unittest
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.task1_residual_correction import (
    WEIGHTS,
    blend_prediction,
    candidate_name,
    nested_inner_splits,
)


class Task1ResidualCorrectionTests(unittest.TestCase):
    def test_blend_formula(self):
        actual = blend_prediction([5.0, 6.0], [0.4, -0.2], 0.5)
        np.testing.assert_allclose(actual, [5.2, 5.9])

    def test_candidate_names_and_frozen_weights(self):
        self.assertEqual(WEIGHTS, (0.25, 0.5, 0.75, 1.0))
        self.assertEqual(candidate_name("ridge", 0.25), "ridge_residual_w0p25")

    def test_inner_validation_rows_are_disjoint_and_complete(self):
        row_count = 31
        validation_rows = []
        for train_idx, valid_idx in nested_inner_splits(row_count, outer_fold=2):
            self.assertFalse(set(train_idx) & set(valid_idx))
            validation_rows.extend(valid_idx.tolist())
        self.assertEqual(sorted(validation_rows), list(range(row_count)))


if __name__ == "__main__":
    unittest.main()
