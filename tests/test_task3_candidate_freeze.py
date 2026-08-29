import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.task3_candidate_freeze import (
    EXPECTED_CONTRACT,
    EXPECTED_MODEL,
    EXPECTED_SCIENTIFIC_CONTRACT,
    EXPECTED_SCIENTIFIC_MODEL,
    build_registry,
)


class Task3CandidateFreezeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry, cls.manifest, cls.selected_oof, cls.scientific_oof = build_registry(ROOT)

    def test_exactly_one_competition_candidate_is_selected(self):
        selected = self.registry.loc[self.registry["decision"] == "selected"]
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected.iloc[0]["best_model"], EXPECTED_MODEL)
        self.assertEqual(selected.iloc[0]["feature_contract"], EXPECTED_CONTRACT)

    def test_scientific_comparator_is_preserved(self):
        scientific = self.registry.loc[self.registry["decision"] == "scientific_comparator"]
        self.assertEqual(len(scientific), 1)
        self.assertEqual(scientific.iloc[0]["best_model"], EXPECTED_SCIENTIFIC_MODEL)
        self.assertEqual(scientific.iloc[0]["feature_contract"], EXPECTED_SCIENTIFIC_CONTRACT)

    def test_all_registered_runs_keep_holdout_sealed(self):
        self.assertFalse(self.registry["holdout_evaluated"].any())
        self.assertFalse(self.manifest["holdout_evaluated"])
        self.assertEqual(self.manifest["sealed_holdout_rows"], 2000)

    def test_frozen_metrics_reconcile_with_source_results(self):
        metrics = self.manifest["selected_oof_metrics"]
        self.assertAlmostEqual(metrics["r2"], 0.9807002146476884)
        self.assertAlmostEqual(metrics["mae"], 1.3529337096156853)
        self.assertAlmostEqual(metrics["rmse"], 1.701476424199776)

    def test_boundary_confirmation_preserves_stop_rule(self):
        stop = self.manifest["stop_confirmation"]
        self.assertFalse(stop["material_improvement"])
        self.assertLess(stop["delta_vs_selected_oof_r2"], stop["minimum_material_improvement"])

    def test_both_compact_oof_frames_are_complete(self):
        expected_columns = ["Person_ID", "true_value", "predicted_value", "residual"]
        self.assertEqual(list(self.selected_oof.columns), expected_columns)
        self.assertEqual(list(self.scientific_oof.columns), expected_columns)
        self.assertEqual(len(self.selected_oof), 8000)
        self.assertEqual(len(self.scientific_oof), 8000)


if __name__ == "__main__":
    unittest.main()
