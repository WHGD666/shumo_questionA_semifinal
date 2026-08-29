import unittest

import numpy as np

from src.task2_learning_curve import (
    FROZEN_ALPHA,
    FROZEN_L1_RATIO,
    SUBSAMPLE_SEEDS,
    TRAINING_FRACTIONS,
    diagnose_plateau,
    fraction_label,
    nested_training_positions,
)
from src.task2_elastic_tuning import make_candidate


class Task2LearningCurveTests(unittest.TestCase):
    def test_curve_is_frozen_bounded_and_includes_full_training_data(self):
        self.assertEqual(TRAINING_FRACTIONS, tuple(sorted(set(TRAINING_FRACTIONS))))
        self.assertEqual(TRAINING_FRACTIONS[-1], 1.0)
        self.assertEqual(len(SUBSAMPLE_SEEDS), 5)
        self.assertEqual(len(set(SUBSAMPLE_SEEDS)), len(SUBSAMPLE_SEEDS))

    def test_nested_subsamples_are_deterministic_exact_and_nested(self):
        positions = np.arange(100)
        first = nested_training_positions(positions, TRAINING_FRACTIONS, seed=2026)
        second = nested_training_positions(positions, TRAINING_FRACTIONS, seed=2026)
        for fraction in TRAINING_FRACTIONS:
            np.testing.assert_array_equal(first[fraction], second[fraction])
            self.assertEqual(len(first[fraction]), round(100 * fraction))
            self.assertEqual(len(np.unique(first[fraction])), len(first[fraction]))
        for smaller, larger in zip(TRAINING_FRACTIONS, TRAINING_FRACTIONS[1:]):
            self.assertTrue(set(first[smaller]).issubset(set(first[larger])))

    def test_frozen_candidate_matches_tuning_winner(self):
        model = make_candidate(FROZEN_ALPHA, FROZEN_L1_RATIO, seed=2026)
        self.assertEqual(model.alpha, 0.01)
        self.assertEqual(model.l1_ratio, 0.9)
        self.assertEqual(model.max_iter, 20000)

    def test_plateau_diagnosis_requires_small_gain_and_small_gap(self):
        self.assertTrue(diagnose_plateau(0.004, 0.02))
        self.assertFalse(diagnose_plateau(0.006, 0.02))
        self.assertFalse(diagnose_plateau(0.004, 0.04))

    def test_fraction_label_is_filename_safe_and_sortable(self):
        self.assertEqual([fraction_label(value) for value in TRAINING_FRACTIONS], [
            "020pct", "040pct", "060pct", "080pct", "100pct",
        ])


if __name__ == "__main__":
    unittest.main()
