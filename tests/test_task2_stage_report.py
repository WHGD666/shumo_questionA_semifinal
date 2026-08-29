import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.task2_stage_report import build_evidence_text, build_report_text, validate_stage_sources


class Task2StageReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry, cls.freeze, cls.learning, cls.curve = validate_stage_sources(ROOT)
        cls.report = build_report_text(cls.registry, cls.freeze, cls.learning, cls.curve)
        cls.evidence = build_evidence_text(cls.registry, cls.freeze)

    def test_report_uses_frozen_metrics(self):
        self.assertIn("0.620162", self.report)
        self.assertIn("0.623012", self.report)
        self.assertIn("0.559853", self.report)
        self.assertIn("0.703053", self.report)

    def test_report_preserves_holdout_disclosure(self):
        self.assertIn("尚未训练最终部署模型", self.report)
        self.assertIn("没有启封2000条最终留出集", self.report)

    def test_report_preserves_proxy_comparison(self):
        self.assertIn("0.619912", self.report)
        self.assertIn("均未超过代理剔除契约", self.report)

    def test_report_preserves_learning_curve_stop_evidence(self):
        self.assertIn("0.000410", self.report)
        self.assertIn("0.003369", self.report)
        self.assertIn("实际性能平台诊断通过", self.report)

    def test_evidence_index_points_to_machine_readable_sources(self):
        self.assertIn("task2_experiment_registry.csv", self.evidence)
        self.assertIn("task2_frozen_candidate_manifest.json", self.evidence)
        self.assertIn("task2_frozen_candidate_oof_predictions.csv", self.evidence)
        self.assertIn(self.freeze["selection_run_id"], self.evidence)


if __name__ == "__main__":
    unittest.main()
