import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.task1_stage_report import build_evidence_text, build_report_text, validate_stage_sources


class Task1StageReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry, cls.freeze, cls.stability = validate_stage_sources(ROOT)
        cls.report = build_report_text(cls.registry, cls.freeze, cls.stability)
        cls.evidence = build_evidence_text(cls.registry, cls.freeze)

    def test_report_uses_frozen_metrics(self):
        self.assertIn("0.793587", self.report)
        self.assertIn("0.574395", self.report)
        self.assertIn("0.721731", self.report)

    def test_report_preserves_holdout_disclosure(self):
        self.assertIn("尚未训练最终部署模型", self.report)
        self.assertIn("没有启封2000行最终留出集", self.report)

    def test_report_preserves_proxy_comparison(self):
        self.assertIn("0.680178", self.report)
        self.assertIn("0.110424", self.report)

    def test_evidence_index_points_to_machine_readable_sources(self):
        self.assertIn("task1_experiment_registry.csv", self.evidence)
        self.assertIn("task1_frozen_candidate_manifest.json", self.evidence)
        self.assertIn(self.freeze["selection_run_id"], self.evidence)


if __name__ == "__main__":
    unittest.main()
