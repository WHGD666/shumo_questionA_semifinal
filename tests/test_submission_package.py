import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.final_evaluation_models import ID_COLUMN, TARGETS
from src.submission_package import (
    build_submission_package,
    verify_submission_package,
)

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "submission" / "model"
PYTHON = sys.executable


class SubmissionPackageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        build_submission_package(ROOT, overwrite=True)
        cls.manifest = json.loads(
            (PACKAGE_ROOT / "model_manifest.json").read_text(encoding="utf-8")
        )
        cls.example_input = PACKAGE_ROOT / "example_input.csv"

    def test_package_manifest_matches_release_training_manifest(self):
        training = json.loads(
            (
                ROOT / "outputs" / "logs" / "final_release_model_training_manifest.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(self.manifest["run_id"], training["run_id"])
        for task in TARGETS:
            entry = self.manifest["tasks"][task]
            self.assertEqual(
                entry["model_sha256"], training["tasks"][task]["model_sha256"]
            )
            self.assertNotIn(TARGETS[task], entry["required_columns"])
            self.assertNotIn(ID_COLUMN, entry["required_columns"])

    def test_verify_package_reports_verified_contract(self):
        findings = verify_submission_package(ROOT)
        self.assertEqual(findings["status"], "verified")
        self.assertEqual(set(findings["tasks"]), set(TARGETS))

    def test_predict_entrypoint_accepts_masked_input_with_extra_columns(self):
        frame = pd.read_csv(self.example_input)
        with tempfile.TemporaryDirectory() as directory:
            masked = frame.drop(columns=[TARGETS["task1"]])
            masked["extra_note"] = "ignored"
            masked_path = Path(directory) / "masked.csv"
            masked.to_csv(masked_path, index=False, encoding="utf-8")
            output_path = Path(directory) / "task1_predictions.csv"
            completed = subprocess.run(
                [
                    PYTHON,
                    str(PACKAGE_ROOT / "predict.py"),
                    "--task",
                    "task1",
                    "--input",
                    str(masked_path),
                    "--output",
                    str(output_path),
                ],
                capture_output=True,
                text=True,
                timeout=300,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            output = pd.read_csv(output_path)
            self.assertEqual(list(output.columns), [ID_COLUMN, "Sleep_Quality_Score_prediction"])
            self.assertEqual(len(output), len(frame))
            self.assertTrue(output[ID_COLUMN].equals(frame[ID_COLUMN]))
            self.assertTrue(np.isfinite(output["Sleep_Quality_Score_prediction"]).all())

    def test_predict_all_writes_three_outputs_in_order(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "outputs"
            completed = subprocess.run(
                [
                    PYTHON,
                    str(PACKAGE_ROOT / "predict.py"),
                    "--task",
                    "all",
                    "--input",
                    str(self.example_input),
                    "--output-dir",
                    str(output_dir),
                ],
                capture_output=True,
                text=True,
                timeout=300,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            for task, target in TARGETS.items():
                output = pd.read_csv(output_dir / f"{task}_predictions.csv")
                self.assertEqual(list(output.columns), [ID_COLUMN, f"{target}_prediction"])
                self.assertEqual(len(output), 20)

    def test_missing_required_column_reports_clear_error(self):
        frame = pd.read_csv(self.example_input)
        required = self.manifest["tasks"]["task1"]["required_columns"]
        victim = next(column for column in required if column in frame.columns)
        with tempfile.TemporaryDirectory() as directory:
            masked_path = Path(directory) / "missing.csv"
            frame.drop(columns=[victim]).to_csv(masked_path, index=False, encoding="utf-8")
            completed = subprocess.run(
                [
                    PYTHON,
                    str(PACKAGE_ROOT / "predict.py"),
                    "--task",
                    "task1",
                    "--input",
                    str(masked_path),
                    "--output",
                    str(Path(directory) / "out.csv"),
                ],
                capture_output=True,
                text=True,
                timeout=300,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("缺少必需列", completed.stderr + completed.stdout)
            self.assertIn(victim, completed.stderr + completed.stdout)

    def test_duplicate_person_id_is_rejected(self):
        frame = pd.read_csv(self.example_input)
        duplicated = pd.concat([frame.head(3), frame.head(3)], ignore_index=True)
        with tempfile.TemporaryDirectory() as directory:
            duplicated_path = Path(directory) / "duplicated.csv"
            duplicated.to_csv(duplicated_path, index=False, encoding="utf-8")
            completed = subprocess.run(
                [
                    PYTHON,
                    str(PACKAGE_ROOT / "predict.py"),
                    "--task",
                    "task2",
                    "--input",
                    str(duplicated_path),
                    "--output",
                    str(Path(directory) / "out.csv"),
                ],
                capture_output=True,
                text=True,
                timeout=300,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("唯一", completed.stderr + completed.stdout)

    def test_verify_package_script_passes_in_package_directory(self):
        completed = subprocess.run(
            [PYTHON, str(PACKAGE_ROOT / "verify_package.py")],
            capture_output=True,
            text=True,
            timeout=300,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("全部通过", completed.stdout)

    def test_example_results_cover_all_tasks(self):
        for task, target in TARGETS.items():
            path = ROOT / "submission" / "results" / f"example_{task}_predictions.csv"
            output = pd.read_csv(path)
            self.assertEqual(list(output.columns), [ID_COLUMN, f"{target}_prediction"])
            self.assertEqual(len(output), 20)

    def test_masked_example_inputs_exclude_only_own_target(self):
        for task, target in TARGETS.items():
            masked = pd.read_csv(PACKAGE_ROOT / f"example_input_{task}.csv")
            self.assertNotIn(target, masked.columns)
            self.assertIn(ID_COLUMN, masked.columns)
        full = pd.read_csv(PACKAGE_ROOT / "example_input.csv")
        for target in TARGETS.values():
            self.assertIn(target, full.columns)


if __name__ == "__main__":
    unittest.main()
