import unittest
from pathlib import Path

from src.final_evaluation_protocol import (
    PAPER_ARTIFACT_PROHIBITIONS,
    load_protocol,
    require_holdout_confirmation,
    validate_final_evaluation_protocol,
)


ROOT = Path(__file__).resolve().parents[1]


class FinalEvaluationProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = load_protocol(ROOT)
        cls.validation = validate_final_evaluation_protocol(ROOT)

    def test_all_three_frozen_candidates_are_registered(self):
        self.assertEqual(tuple(self.validation["tasks"]), ("task1", "task2", "task3"))
        self.assertEqual(
            [self.validation["tasks"][task]["holdout_evaluated"] for task in self.validation["tasks"]],
            [False, False, False],
        )
        self.assertEqual(
            [self.validation["tasks"][task]["final_model_trained"] for task in self.validation["tasks"]],
            [False, False, False],
        )

    def test_split_is_complete_disjoint_and_balanced(self):
        split = self.validation["split"]
        self.assertEqual(split["rows"], 10000)
        self.assertEqual(split["unique_ids"], 10000)
        self.assertEqual(split["development_rows"], 8000)
        self.assertEqual(split["holdout_rows"], 2000)
        self.assertEqual(split["fold_counts"], {str(index): 1600 for index in range(5)})

    def test_holdout_gate_is_not_preauthorized(self):
        self.assertTrue(self.validation["evaluation_model_training_authorized"])
        self.assertFalse(self.validation["holdout_evaluation_authorized"])
        self.assertFalse(self.protocol["protocol"]["allow_repeated_holdout_evaluation"])

    def test_holdout_confirmation_requires_exact_token(self):
        token = self.protocol["protocol"]["holdout_confirmation_token"]
        require_holdout_confirmation(ROOT, token)
        with self.assertRaises(ValueError):
            require_holdout_confirmation(ROOT, token.lower())
        with self.assertRaises(ValueError):
            require_holdout_confirmation(ROOT, "TASK_FINAL_HOLDOUT")

    def test_paper_artifact_gate_forces_manual_pause(self):
        gate = self.protocol["paper_artifact_gate"]
        self.assertEqual(gate["status"], "blocked_until_manual_confirmation")
        self.assertTrue(gate["requires_user_confirmation"])
        self.assertTrue(gate["stop_and_remind_user"])
        self.assertEqual(set(gate["prohibited_before_confirmation"]), PAPER_ARTIFACT_PROHIBITIONS)

    def test_candidate_hashes_and_run_ids_are_present(self):
        for task in self.validation["tasks"].values():
            self.assertEqual(len(task["candidate_config_sha256"]), 64)
            self.assertEqual(len(task["candidate_manifest_sha256"]), 64)
            self.assertTrue(task["selection_run_id"])
            self.assertTrue(task["confirmation_run_id"])


if __name__ == "__main__":
    unittest.main()
