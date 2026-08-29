import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.task3_stage_report import (
    build_evidence_text,
    build_proxy_comparison,
    build_report_text,
    validate_stage_sources,
)


class Task3StageReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry, cls.freeze, cls.protocol, _ = validate_stage_sources(ROOT)
        cls.report = build_report_text(cls.registry, cls.freeze, cls.protocol)
        cls.evidence = build_evidence_text(cls.registry, cls.freeze)
        cls.proxy = build_proxy_comparison(cls.registry)

    def test_report_uses_frozen_metrics(self):
        self.assertIn("0.980700", self.report)
        self.assertIn("1.352934", self.report)
        self.assertIn("1.701476", self.report)
        self.assertIn("0.938393", self.report)

    def test_report_preserves_holdout_disclosure(self):
        self.assertIn("尚未训练最终部署模型", self.report)
        self.assertIn("没有启封2000条最终留出集", self.report)

    def test_report_preserves_proxy_disclosure(self):
        self.assertIn("Fitness_Level", self.report)
        self.assertIn("Wellness_Category", self.report)
        self.assertIn("Healthy_Aging_Score", self.report)
        self.assertIn("0.042308", self.report)

    def test_report_preserves_stop_evidence(self):
        self.assertIn("0.000096", self.report)
        self.assertIn("0.000200", self.report)
        self.assertIn("保留更简单的4结点模型", self.report)

    def test_proxy_table_keeps_both_contracts(self):
        self.assertEqual(len(self.proxy), 2)
        self.assertEqual(set(self.proxy["role"]), {"competition_candidate", "scientific_comparator"})
        self.assertAlmostEqual(self.proxy.iloc[0]["r2_gap_vs_scientific"], 0.0423076270675413)

    def test_evidence_index_points_to_machine_readable_sources(self):
        self.assertIn("task3_experiment_registry.csv", self.evidence)
        self.assertIn("task3_frozen_candidate_manifest.json", self.evidence)
        self.assertIn("task3_frozen_candidate_oof_predictions.csv", self.evidence)
        self.assertIn(self.freeze["selection_run_id"], self.evidence)


if __name__ == "__main__":
    unittest.main()
