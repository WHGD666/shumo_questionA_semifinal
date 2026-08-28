import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.task1_candidate_freeze import EXPECTED_MODEL, build_registry


class Task1CandidateFreezeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry, cls.manifest = build_registry(ROOT)

    def test_exactly_one_candidate_is_selected(self):
        selected = self.registry.loc[self.registry["decision"] == "selected"]
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected.iloc[0]["best_model"], EXPECTED_MODEL)

    def test_all_registered_runs_keep_holdout_sealed(self):
        self.assertFalse(self.registry["holdout_evaluated"].any())
        self.assertFalse(self.manifest["holdout_evaluated"])
        self.assertEqual(self.manifest["sealed_holdout_rows"], 2000)

    def test_frozen_metrics_reconcile_with_source_results(self):
        metrics = self.manifest["selected_oof_metrics"]
        self.assertAlmostEqual(metrics["r2"], 0.7935867590718252)
        self.assertAlmostEqual(metrics["mae"], 0.5743946660285861)
        self.assertAlmostEqual(metrics["rmse"], 0.721730891558355)

    def test_proxy_comparator_is_preserved(self):
        comparator = self.registry.loc[self.registry["decision"] == "scientific_comparator"]
        self.assertEqual(len(comparator), 1)
        self.assertEqual(comparator.iloc[0]["feature_contract"], "proxy_removed")


if __name__ == "__main__":
    unittest.main()
