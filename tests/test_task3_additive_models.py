import unittest

import pandas as pd

from src.task3_additive_models import (
    COMMON_SPLINE_FEATURES,
    COMPETITION_ONLY_SPLINE_FEATURES,
    make_estimator,
    make_preprocessor,
    parameter_grid,
    spline_features_for_contract,
)
from src.task3_protocol import TARGET


class Task3AdditiveModelTests(unittest.TestCase):
    def test_grid_is_bounded_and_unique(self):
        grid = parameter_grid()
        self.assertEqual(len(grid), 10)
        self.assertEqual(len({item["model"] for item in grid}), len(grid))
        self.assertEqual({item["penalty"] for item in grid}, {"ridge", "elastic"})

    def test_spline_features_follow_proxy_contract(self):
        competition = spline_features_for_contract("competition_proxy_inclusive")
        scientific = spline_features_for_contract("scientific_proxy_removed")
        self.assertEqual(set(competition) - set(scientific), set(COMPETITION_ONLY_SPLINE_FEATURES))
        self.assertEqual(scientific, COMMON_SPLINE_FEATURES)
        self.assertNotIn(TARGET, competition)

    def test_unknown_contract_is_rejected(self):
        with self.assertRaises(ValueError):
            spline_features_for_contract("unknown")

    def test_preprocessor_keeps_spline_linear_and_categorical_blocks(self):
        spline_features = ("BMI", "Age")
        frame = pd.DataFrame({
            "BMI": [20.0, 25.0, 30.0],
            "Age": [20, 40, 60],
            "Other_Numeric": [4.0, 5.0, 6.0],
            "Category": ["a", "b", "a"],
        })
        transformed = make_preprocessor(
            frame, spline_features, n_knots=3, degree=2
        ).fit_transform(frame)
        self.assertEqual(transformed.shape[0], 3)
        self.assertGreater(transformed.shape[1], frame.shape[1])

    def test_estimator_factory_preserves_penalty(self):
        ridge = make_estimator("ridge", alpha=1.0, seed=2026)
        elastic = make_estimator("elastic", alpha=0.01, seed=2026)
        self.assertEqual(ridge.alpha, 1.0)
        self.assertEqual(elastic.l1_ratio, 0.9)


if __name__ == "__main__":
    unittest.main()
