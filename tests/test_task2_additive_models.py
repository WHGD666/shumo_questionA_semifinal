import unittest

import pandas as pd

from src.task2_additive_models import (
    SPLINE_FEATURES,
    make_estimator,
    make_preprocessor,
    parameter_grid,
)
from src.task2_protocol import TARGET


class Task2AdditiveModelTests(unittest.TestCase):
    def test_grid_is_bounded_and_unique(self):
        grid = parameter_grid()
        self.assertEqual(len(grid), 24)
        self.assertEqual(len({item["model"] for item in grid}), len(grid))
        self.assertEqual({item["penalty"] for item in grid}, {"ridge", "elastic"})

    def test_spline_features_are_substantive_and_exclude_target(self):
        self.assertNotIn(TARGET, SPLINE_FEATURES)
        self.assertEqual(len(SPLINE_FEATURES), 9)
        self.assertEqual(len(set(SPLINE_FEATURES)), len(SPLINE_FEATURES))

    def test_preprocessor_keeps_spline_linear_and_categorical_blocks(self):
        frame = pd.DataFrame({
            **{column: [1.0, 2.0, 3.0] for column in SPLINE_FEATURES},
            "Other_Numeric": [4.0, 5.0, 6.0],
            "Category": ["a", "b", "a"],
        })
        transformed = make_preprocessor(frame, n_knots=3, degree=2).fit_transform(frame)
        self.assertEqual(transformed.shape[0], 3)
        self.assertGreater(transformed.shape[1], frame.shape[1])

    def test_estimator_factory_preserves_penalty(self):
        ridge = make_estimator("ridge", alpha=1.0, seed=2026)
        elastic = make_estimator("elastic", alpha=0.01, seed=2026)
        self.assertEqual(ridge.alpha, 1.0)
        self.assertEqual(elastic.l1_ratio, 0.9)


if __name__ == "__main__":
    unittest.main()
