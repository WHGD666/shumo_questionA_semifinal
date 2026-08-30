import tempfile
import unittest
from pathlib import Path
import tomllib

import joblib
import numpy as np
import pandas as pd

from src.final_evaluation_models import (
    Task1EvaluationRegressor,
    Task2EvaluationRegressor,
    Task3EvaluationRegressor,
    build_evaluation_model,
    load_development_frame,
    prediction_frame,
)


ROOT = Path(__file__).resolve().parents[1]


def synthetic_frame(rows: int = 40) -> pd.DataFrame:
    index = np.arange(rows)
    health = 55.0 + 0.6 * index
    frame = pd.DataFrame({
        "Person_ID": [f"S{value:04d}" for value in index],
        "Age": 20 + index % 50,
        "BMI": 18.0 + (index % 20) * 0.5,
        "Weight_kg": 50.0 + index,
        "Gender": np.where(index % 2 == 0, "Male", "Female").astype(object),
        "Sleep_Time": np.where(index % 2 == 0, "22:30", "23:15").astype(object),
        "Wake_Up_Time": np.where(index % 2 == 0, "06:30", "07:15").astype(object),
        "Sleep_Duration_Hours": 7.0 + (index % 5) * 0.1,
        "Number_of_Night_Awakenings": index % 3,
        "Weekend_Sleep_Difference_Hours": (index % 4) * 0.2,
        "Nap_Frequency_Per_Week": index % 4,
        "Screen_Time_Before_Bed_Hours": 0.5 + (index % 3) * 0.2,
        "Sleep_Disorder_Risk": np.where(index % 3 == 0, "Low", "Medium").astype(object),
        "Exercise_Frequency_Per_Week": index % 7,
        "Immune_Health_Score": 50.0 + index * 0.4,
        "Protein_Intake_Grams": 60.0 + index,
        "Sitting_Hours_Per_Day": 4.0 + (index % 8) * 0.5,
        "Daily_Steps": 5000 + index * 150,
        "Daily_Calorie_Intake": 1800 + index * 10,
        "Water_Intake_Liters": 1.5 + (index % 5) * 0.2,
        "Systolic_BP": 105 + index % 30,
        "Healthy_Aging_Score": 50.0 + index * 0.5,
        "Fitness_Level": np.where(health >= 65, "Good", "Average").astype(object),
        "Wellness_Category": np.where(health >= 65, "Good", "Average").astype(object),
        "Sleep_Quality_Score": 4.0 + index * 0.08,
        "Productivity_Score": 3.0 + index * 0.07,
        "Health_Score": health,
    })
    return frame


class FinalEvaluationModelTests(unittest.TestCase):
    def test_builders_preserve_frozen_model_parameters(self):
        task1 = build_evaluation_model("task1")
        task2 = build_evaluation_model("task2")
        task3 = build_evaluation_model("task3")
        self.assertIsInstance(task1, Task1EvaluationRegressor)
        self.assertEqual(task1.residual_weight, 0.75)
        self.assertEqual(task1.inner_folds, 4)
        self.assertIsInstance(task2, Task2EvaluationRegressor)
        self.assertEqual((task2.alpha, task2.l1_ratio), (0.01, 0.9))
        self.assertIsInstance(task3, Task3EvaluationRegressor)
        self.assertEqual((task3.n_knots, task3.degree, task3.alpha), (4, 2, 0.003))

    def test_model_outputs_are_isolated_from_release_artifacts(self):
        with (ROOT / "configs" / "final_evaluation_models.toml").open("rb") as stream:
            config = tomllib.load(stream)
        self.assertEqual(
            config["training"]["model_directory"],
            "outputs/models/evaluation",
        )
        self.assertNotIn("submission", config["training"]["model_directory"])

    def test_unknown_task_is_rejected(self):
        with self.assertRaises(ValueError):
            build_evaluation_model("task4")

    def test_all_models_round_trip_from_raw_columns(self):
        frame = synthetic_frame()
        targets = {
            "task1": frame["Sleep_Quality_Score"],
            "task2": frame["Productivity_Score"],
            "task3": frame["Health_Score"],
        }
        for task in ("task1", "task2", "task3"):
            with self.subTest(task=task):
                model = build_evaluation_model(task)
                model.fit(frame, targets[task])
                expected = model.predict(frame.iloc[:6])
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / f"{task}.joblib"
                    joblib.dump(model, path)
                    restored = joblib.load(path)
                    actual = restored.predict(frame.iloc[:6])
                np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)
                self.assertNotIn(targets[task].name, model.required_columns_)
                self.assertNotIn("Person_ID", model.required_columns_)

    def test_missing_required_column_is_rejected_but_extra_column_is_allowed(self):
        frame = synthetic_frame()
        model = Task2EvaluationRegressor().fit(frame, frame["Productivity_Score"])
        extra = frame.copy()
        extra["Unused_Extra"] = 1
        self.assertEqual(len(model.predict(extra.iloc[:3])), 3)
        missing = frame.drop(columns=[model.required_columns_[0]])
        with self.assertRaises(ValueError):
            model.predict(missing)

    def test_prediction_frame_preserves_id_and_rejects_duplicates(self):
        frame = synthetic_frame()
        model = Task3EvaluationRegressor().fit(frame, frame["Health_Score"])
        output = prediction_frame(model, frame.iloc[:5])
        self.assertEqual(output.columns.tolist(), ["Person_ID", "predicted_value"])
        self.assertEqual(output["Person_ID"].tolist(), frame["Person_ID"].iloc[:5].tolist())
        duplicate = frame.iloc[:5].copy()
        duplicate.loc[duplicate.index[1], "Person_ID"] = duplicate.iloc[0]["Person_ID"]
        with self.assertRaises(ValueError):
            prediction_frame(model, duplicate)

    def test_actual_development_loader_excludes_holdout_rows(self):
        development = load_development_frame(ROOT)
        assignments = pd.read_csv(ROOT / "data" / "splits" / "split_assignments.csv")
        holdout_ids = set(assignments.loc[assignments["split"] == "holdout", "Person_ID"])
        self.assertEqual(len(development), 8000)
        self.assertEqual(development["Person_ID"].nunique(), 8000)
        self.assertTrue(set(development["Person_ID"]).isdisjoint(holdout_ids))


if __name__ == "__main__":
    unittest.main()
