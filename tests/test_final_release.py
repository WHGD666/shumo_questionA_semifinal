import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.final_evaluation_models import build_evaluation_model
from src.final_release import (
    load_all_labeled_rows,
    load_release_config,
    validate_release_gate,
)


ROOT = Path(__file__).resolve().parents[1]


class FinalReleaseTests(unittest.TestCase):
    def test_release_gate_requires_locked_holdout_and_no_post_holdout_tuning(self):
        gate = validate_release_gate(ROOT)
        release = gate["config"]["release"]
        self.assertEqual(release["training_rows"], 10000)
        self.assertTrue(release["holdout_labels_used_for_refit"])
        self.assertFalse(release["holdout_used_for_model_selection"])
        self.assertFalse(release["post_holdout_tuning"])

    def test_release_data_are_complete_and_identified(self):
        config = load_release_config(ROOT)
        frame = load_all_labeled_rows(ROOT, config)
        self.assertEqual(len(frame), 10000)
        self.assertEqual(frame["Person_ID"].nunique(), 10000)

    def test_release_models_keep_frozen_parameters(self):
        task1 = build_evaluation_model("task1")
        task2 = build_evaluation_model("task2")
        task3 = build_evaluation_model("task3")
        self.assertEqual((task1.residual_weight, task1.inner_folds), (0.75, 4))
        self.assertEqual((task2.alpha, task2.l1_ratio), (0.01, 0.9))
        self.assertEqual((task3.n_knots, task3.degree, task3.alpha), (4, 2, 0.003))

    def test_independent_task_contract_excludes_own_target(self):
        config = load_release_config(ROOT)
        frame = load_all_labeled_rows(ROOT, config).iloc[:80].copy()
        for task, target in {
            "task1": "Sleep_Quality_Score",
            "task2": "Productivity_Score",
            "task3": "Health_Score",
        }.items():
            with self.subTest(task=task):
                model = build_evaluation_model(task).fit(frame, frame[target])
                self.assertNotIn(target, model.required_columns_)
                self.assertNotIn("Person_ID", model.required_columns_)
                masked = frame.drop(columns=[target])
                prediction = model.predict(masked.iloc[:5])
                self.assertTrue(np.isfinite(prediction).all())
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / f"{task}.joblib"
                    joblib.dump(model, path)
                    restored = joblib.load(path)
                    np.testing.assert_allclose(
                        restored.predict(masked.iloc[:5]), prediction, atol=1e-12, rtol=0.0
                    )


if __name__ == "__main__":
    unittest.main()
