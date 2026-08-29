import unittest

import numpy as np

from src.task2_residual_correction import (
    CONTRACT,
    WEIGHTS,
    _base_pipeline,
    blend_prediction,
    candidate_name,
    nested_inner_splits,
)


class Task2ResidualCorrectionTests(unittest.TestCase):
    def test_blend_formula(self):
        actual = blend_prediction([5.0, 6.0], [0.4, -0.2], 0.5)
        np.testing.assert_allclose(actual, [5.2, 5.9])

    def test_candidate_names_and_frozen_weights(self):
        self.assertEqual(WEIGHTS, (0.25, 0.5, 0.75, 1.0))
        self.assertEqual(candidate_name("lightgbm", 0.75), "lightgbm_residual_w0p75")

    def test_inner_validation_rows_are_disjoint_and_complete(self):
        validation_rows = []
        for train_idx, valid_idx in nested_inner_splits(31, outer_fold=2):
            self.assertFalse(set(train_idx) & set(valid_idx))
            validation_rows.extend(valid_idx.tolist())
        self.assertEqual(sorted(validation_rows), list(range(31)))

    def test_base_and_contract_are_frozen(self):
        import pandas as pd

        frame = pd.DataFrame({"x": [1.0, 2.0]})
        model = _base_pipeline(frame, seed=2037).named_steps["model"]
        self.assertEqual(CONTRACT, "scientific_proxy_removed")
        self.assertEqual(model.alpha, 0.01)
        self.assertEqual(model.l1_ratio, 0.9)


if __name__ == "__main__":
    unittest.main()
