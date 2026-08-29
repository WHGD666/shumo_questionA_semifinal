import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.task2_candidate_freeze import EXPECTED_CONTRACT, EXPECTED_MODEL, build_registry


class Task2CandidateFreezeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry, cls.manifest, cls.oof = build_registry(ROOT)

    def test_exactly_one_candidate_is_selected(self):
        selected = self.registry.loc[self.registry["decision"] == "selected"]
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected.iloc[0]["best_model"], EXPECTED_MODEL)
        self.assertEqual(selected.iloc[0]["feature_contract"], EXPECTED_CONTRACT)

    def test_all_registered_runs_keep_holdout_sealed(self):
        self.assertFalse(self.registry["holdout_evaluated"].any())
        self.assertFalse(self.manifest["holdout_evaluated"])
        self.assertEqual(self.manifest["sealed_holdout_rows"], 2000)

    def test_frozen_metrics_reconcile_with_source_results(self):
        metrics = self.manifest["selected_oof_metrics"]
        self.assertAlmostEqual(metrics["adjusted_r2_raw"], 0.6201624354997604)
        self.assertAlmostEqual(metrics["r2"], 0.6230115733757466)
        self.assertAlmostEqual(metrics["mae"], 0.559852657967001)
        self.assertAlmostEqual(metrics["rmse"], 0.7030530944052009)

    def test_learning_curve_confirmation_is_preserved(self):
        curve = self.manifest["learning_curve_confirmation"]
        self.assertTrue(curve["practical_plateau_supported"])
        self.assertLess(curve["gain_oof_r2_80_to_100"], 0.005)

    def test_proxy_comparator_and_compact_oof_are_preserved(self):
        comparator = self.registry.loc[self.registry["decision"] == "scientific_comparator"]
        self.assertEqual(len(comparator), 1)
        self.assertEqual(comparator.iloc[0]["feature_contract"], "competition")
        self.assertEqual(list(self.oof.columns), ["Person_ID", "true_value", "predicted_value", "residual"])
        self.assertEqual(len(self.oof), 8000)


if __name__ == "__main__":
    unittest.main()
