import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.final_holdout_evaluation import (
    build_holdout_prediction_frame,
    configured_output_paths,
    load_partition_frame,
    regression_metric_row,
    require_exact_confirmation,
    require_fresh_holdout_outputs,
    validate_task_output_consistency,
)
from src.task2_protocol import adjusted_r2


ROOT = Path(__file__).resolve().parents[1]


class FinalHoldoutEvaluationTests(unittest.TestCase):
    def test_confirmation_requires_exact_token(self):
        with self.assertRaises(ValueError):
            require_exact_confirmation(ROOT, "SEMIFINAL_FINAL_HOLDOUT")
        require_exact_confirmation(ROOT, "SEMIFINAL_FINAL_HOLDOUT_ONCE")

    def test_output_paths_are_isolated_and_named(self):
        paths = configured_output_paths(ROOT)
        relative = {name: str(path.relative_to(ROOT)).replace("\\", "/") for name, path in paths.items()}
        self.assertEqual(len(relative), len(set(relative.values())))
        self.assertEqual(
            relative["consumed_marker"],
            "outputs/logs/final_holdout_consumed.json",
        )
        for task in ("task1", "task2", "task3"):
            self.assertIn("final_holdout", relative[f"{task}_predictions"])

    def test_existing_consumed_marker_blocks_repeat_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_dir = root / "configs"
            config_dir.mkdir()
            config_dir.joinpath("final_holdout_evaluation.toml").write_text(
                (ROOT / "configs" / "final_holdout_evaluation.toml").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            marker = root / "outputs" / "logs" / "final_holdout_consumed.json"
            marker.parent.mkdir(parents=True)
            marker.write_text(json.dumps({"holdout_consumed": True}), encoding="utf-8")
            with self.assertRaises(FileExistsError):
                require_fresh_holdout_outputs(root)

    def test_partition_loader_is_ordered_complete_and_rejects_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "data.csv"
            split_path = root / "splits.csv"
            pd.DataFrame({
                "Person_ID": ["P3", "P1", "P2"],
                "target": [3.0, 1.0, 2.0],
            }).to_csv(data_path, index=False)
            pd.DataFrame({
                "Person_ID": ["P2", "P3", "P1"],
                "split": ["holdout", "development", "holdout"],
            }).to_csv(split_path, index=False)
            frame = load_partition_frame(data_path, split_path, "holdout", 2)
            self.assertEqual(frame["Person_ID"].tolist(), ["P2", "P1"])
            duplicated = pd.DataFrame({
                "Person_ID": ["P1", "P1"],
                "split": ["holdout", "holdout"],
            })
            duplicated.to_csv(split_path, index=False)
            with self.assertRaises(ValueError):
                load_partition_frame(data_path, split_path, "holdout", 2)

    def test_prediction_frame_has_required_columns_and_residual(self):
        output = build_holdout_prediction_frame(
            pd.Series(["P1", "P2"]),
            np.array([2.0, 4.0]),
            np.array([1.5, 4.5]),
        )
        self.assertEqual(
            output.columns.tolist(),
            ["Person_ID", "true_value", "predicted_value", "residual"],
        )
        np.testing.assert_allclose(output["residual"], [0.5, -0.5])

    def test_task2_primary_metric_uses_raw_adjusted_r2(self):
        truth = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        prediction = np.array([1.1, 1.8, 3.2, 3.9, 5.1, 5.8])
        row = regression_metric_row(
            "task2",
            truth,
            prediction,
            raw_predictor_count=2,
        )
        self.assertEqual(row["primary_metric"], "adjusted_r2_raw")
        self.assertAlmostEqual(
            row["primary_value"],
            adjusted_r2(row["r2"], n=6, p=2),
        )
        self.assertFalse(row["official_hidden_test_score"])

    def test_task1_and_task3_primary_metric_is_r2(self):
        truth = np.array([1.0, 2.0, 3.0])
        prediction = np.array([1.0, 2.1, 2.9])
        for task in ("task1", "task3"):
            with self.subTest(task=task):
                row = regression_metric_row(task, truth, prediction, 4)
                self.assertEqual(row["primary_metric"], "r2")
                self.assertEqual(row["primary_value"], row["r2"])
                self.assertTrue(np.isnan(row["adjusted_r2_raw"]))

    def test_prediction_and_metric_consistency_is_recomputed(self):
        ids = pd.Series(["P1", "P2", "P3", "P4"])
        truth = np.array([1.0, 2.0, 3.0, 4.0])
        prediction = np.array([1.1, 1.9, 3.2, 3.8])
        output = build_holdout_prediction_frame(ids, truth, prediction)
        metrics = regression_metric_row("task1", truth, prediction, 2)
        validate_task_output_consistency("task1", output, metrics, ids, 2)
        corrupted = output.copy()
        corrupted.loc[0, "residual"] = 999.0
        with self.assertRaises(ValueError):
            validate_task_output_consistency("task1", corrupted, metrics, ids, 2)


if __name__ == "__main__":
    unittest.main()
