import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.final_results_registry import (
    _assert_close,
    _development_details,
    load_json_strict,
    registry_output_paths,
)


ROOT = Path(__file__).resolve().parents[1]


class FinalResultsRegistryTests(unittest.TestCase):
    def test_strict_json_loader_rejects_nan(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text('{"metric": NaN}', encoding="utf-8")
            with self.assertRaises(ValueError):
                load_json_strict(path)

    def test_metric_reconciliation_uses_tight_tolerance(self):
        _assert_close(0.5, 0.5 + 1e-13, "metric")
        with self.assertRaises(ValueError):
            _assert_close(0.5, 0.500001, "metric")

    def test_registry_outputs_are_machine_readable_not_paper_artifacts(self):
        paths = registry_output_paths(ROOT)
        self.assertEqual(
            str(paths["registry"].relative_to(ROOT)).replace("\\", "/"),
            "outputs/tables/final_results_registry.csv",
        )
        for path in paths.values():
            normalized = str(path).replace("\\", "/").lower()
            self.assertNotIn("paper/", normalized)
            self.assertNotIn("figure", normalized)
            self.assertNotIn("evidence_index", normalized)

    def test_task2_primary_delta_is_marked_sample_size_sensitive(self):
        frozen = load_json_strict(
            ROOT / "outputs" / "logs" / "task2_frozen_candidate_manifest.json"
        )
        details = _development_details("task2", frozen)
        self.assertEqual(details["primary_metric"], "adjusted_r2_raw")
        self.assertEqual(
            details["primary_delta_comparability"],
            "descriptive_only_sample_size_sensitive",
        )
        self.assertEqual(details["feature_contract"], "scientific_proxy_removed")

    def test_proxy_disclosures_are_preserved_for_all_tasks(self):
        for task in ("task1", "task2", "task3"):
            with self.subTest(task=task):
                frozen = load_json_strict(
                    ROOT / "outputs" / "logs" / f"{task}_frozen_candidate_manifest.json"
                )
                details = _development_details(task, frozen)
                self.assertTrue(details["proxy_policy"])
                self.assertTrue(details["proxy_variables"])
                self.assertTrue(details["scientific_comparator_name"])
                self.assertGreaterEqual(float(details["scientific_comparator_value"]), 0.0)

    def test_internal_results_never_claim_official_hidden_score(self):
        holdout = load_json_strict(
            ROOT / "outputs" / "logs" / "final_holdout_evaluation_manifest.json"
        )
        self.assertFalse(holdout["official_hidden_test_score"])
        metrics = pd.read_csv(ROOT / "outputs" / "tables" / "final_holdout_metrics.csv")
        self.assertFalse(metrics["official_hidden_test_score"].any())

    def test_consumed_marker_prohibits_repeat_evaluation(self):
        marker = load_json_strict(
            ROOT / "outputs" / "logs" / "final_holdout_consumed.json"
        )
        self.assertTrue(marker["holdout_consumed"])
        self.assertFalse(marker["repeat_evaluation_allowed"])


if __name__ == "__main__":
    unittest.main()
