from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import lightgbm
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

from src.final_evaluation_protocol import (
    PROTOCOL_PATH,
    sha256_file,
    validate_final_evaluation_protocol,
)
from src.task1_regression_baseline import FORBIDDEN as TASK1_FORBIDDEN
from src.task1_residual_correction import (
    _base_pipeline as task1_base_pipeline,
    _residual_pipeline as task1_residual_pipeline,
    blend_prediction,
)
from src.task2_baseline import make_preprocessor as task2_preprocessor
from src.task2_protocol import feature_contracts as task2_feature_contracts
from src.task2_time_encoding import engineer_cyclic_times
from src.task3_additive_models import (
    make_estimator as task3_estimator,
    make_preprocessor as task3_preprocessor,
    spline_features_for_contract,
)
from src.task3_protocol import feature_contracts as task3_feature_contracts


ID_COLUMN = "Person_ID"
TASKS = ("task1", "task2", "task3")
TARGETS = {
    "task1": "Sleep_Quality_Score",
    "task2": "Productivity_Score",
    "task3": "Health_Score",
}
MODEL_CONFIG_PATH = Path("configs/final_evaluation_models.toml")


def _as_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("模型输入必须是pandas DataFrame")
    if frame.columns.duplicated().any():
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        raise ValueError(f"模型输入包含重复字段: {duplicates}")
    return frame


def _validated_target(y: Any, expected_rows: int) -> np.ndarray:
    values = pd.to_numeric(pd.Series(y).reset_index(drop=True), errors="coerce").to_numpy(dtype=float)
    if len(values) != expected_rows:
        raise ValueError("目标变量长度与输入样本数不一致")
    if not np.isfinite(values).all():
        raise ValueError("目标变量包含缺失值或非有限数值")
    return values


def _select_required(frame: pd.DataFrame, required: tuple[str, ...]) -> pd.DataFrame:
    frame = _as_frame(frame)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"模型输入缺少必需字段: {missing}")
    return frame.loc[:, list(required)].copy()


class Task1EvaluationRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, residual_weight: float = 0.75, inner_folds: int = 4, seed: int = 2026):
        self.residual_weight = residual_weight
        self.inner_folds = inner_folds
        self.seed = seed

    def fit(self, X: pd.DataFrame, y: Any):
        frame = _as_frame(X)
        self.required_columns_ = tuple(
            column for column in frame.columns if column not in TASK1_FORBIDDEN
        )
        if not self.required_columns_:
            raise ValueError("任务一没有可用的非睡眠特征")
        selected = _select_required(frame, self.required_columns_).reset_index(drop=True)
        target = _validated_target(y, len(selected))
        if len(selected) < self.inner_folds:
            raise ValueError("任务一样本数不足以生成交叉拟合残差")

        splitter = KFold(n_splits=self.inner_folds, shuffle=True, random_state=self.seed)
        inner_oof = np.full(len(selected), np.nan)
        for train_positions, valid_positions in splitter.split(selected):
            inner_model = task1_base_pipeline(selected.iloc[train_positions])
            inner_model.fit(selected.iloc[train_positions], target[train_positions])
            inner_oof[valid_positions] = inner_model.predict(selected.iloc[valid_positions])
        if not np.isfinite(inner_oof).all():
            raise ValueError("任务一内层折外主模型预测不完整")

        self.base_model_ = task1_base_pipeline(selected)
        self.base_model_.fit(selected, target)
        residual_target = target - inner_oof
        residual_frame = selected.copy()
        residual_frame["Base_Prediction"] = inner_oof
        self.residual_model_ = task1_residual_pipeline(
            residual_frame, "lightgbm", self.seed
        )
        self.residual_model_.fit(residual_frame, residual_target)
        self.n_features_in_ = len(self.required_columns_)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "base_model_"):
            raise ValueError("任务一模型尚未拟合")
        selected = _select_required(X, self.required_columns_).reset_index(drop=True)
        base_prediction = np.asarray(self.base_model_.predict(selected), dtype=float).reshape(-1)
        residual_frame = selected.copy()
        residual_frame["Base_Prediction"] = base_prediction
        residual_prediction = np.asarray(
            self.residual_model_.predict(residual_frame), dtype=float
        ).reshape(-1)
        prediction = blend_prediction(base_prediction, residual_prediction, self.residual_weight)
        if not np.isfinite(prediction).all():
            raise ValueError("任务一模型产生了非有限预测")
        return prediction


class Task2EvaluationRegressor(RegressorMixin, BaseEstimator):
    def __init__(
        self,
        alpha: float = 0.01,
        l1_ratio: float = 0.9,
        max_iter: int = 20000,
        seed: int = 2026,
    ):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.seed = seed

    def fit(self, X: pd.DataFrame, y: Any):
        frame = _as_frame(X)
        self.required_columns_ = tuple(
            task2_feature_contracts(frame.columns.tolist())["scientific_proxy_removed"]
        )
        selected = _select_required(frame, self.required_columns_).reset_index(drop=True)
        engineered = engineer_cyclic_times(selected)
        target = _validated_target(y, len(engineered))
        self.preprocessor_ = task2_preprocessor(engineered)
        train_matrix = self.preprocessor_.fit_transform(engineered)
        self.model_ = ElasticNet(
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            max_iter=self.max_iter,
            random_state=self.seed,
        )
        self.model_.fit(train_matrix, target)
        self.engineered_feature_count_ = int(engineered.shape[1])
        self.transformed_width_ = int(train_matrix.shape[1])
        self.n_features_in_ = len(self.required_columns_)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise ValueError("任务二模型尚未拟合")
        selected = _select_required(X, self.required_columns_).reset_index(drop=True)
        engineered = engineer_cyclic_times(selected)
        prediction = np.asarray(
            self.model_.predict(self.preprocessor_.transform(engineered)), dtype=float
        ).reshape(-1)
        if not np.isfinite(prediction).all():
            raise ValueError("任务二模型产生了非有限预测")
        return prediction


class Task3EvaluationRegressor(RegressorMixin, BaseEstimator):
    def __init__(
        self,
        n_knots: int = 4,
        degree: int = 2,
        alpha: float = 0.003,
        l1_ratio: float = 0.9,
        seed: int = 2026,
    ):
        self.n_knots = n_knots
        self.degree = degree
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.seed = seed

    def fit(self, X: pd.DataFrame, y: Any):
        frame = _as_frame(X)
        contract = "competition_proxy_inclusive"
        self.required_columns_ = tuple(
            task3_feature_contracts(frame.columns.tolist())[contract]
        )
        selected = _select_required(frame, self.required_columns_).reset_index(drop=True)
        engineered = engineer_cyclic_times(selected)
        target = _validated_target(y, len(engineered))
        self.spline_features_ = tuple(spline_features_for_contract(contract))
        self.preprocessor_ = task3_preprocessor(
            engineered,
            self.spline_features_,
            n_knots=self.n_knots,
            degree=self.degree,
        )
        train_matrix = self.preprocessor_.fit_transform(engineered)
        self.model_ = task3_estimator("elastic", self.alpha, self.seed)
        if isinstance(self.model_, ElasticNet):
            self.model_.set_params(l1_ratio=self.l1_ratio)
        self.model_.fit(train_matrix, target)
        self.engineered_feature_count_ = int(engineered.shape[1])
        self.transformed_width_ = int(train_matrix.shape[1])
        self.n_features_in_ = len(self.required_columns_)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise ValueError("任务三模型尚未拟合")
        selected = _select_required(X, self.required_columns_).reset_index(drop=True)
        engineered = engineer_cyclic_times(selected)
        prediction = np.asarray(
            self.model_.predict(self.preprocessor_.transform(engineered)), dtype=float
        ).reshape(-1)
        if not np.isfinite(prediction).all():
            raise ValueError("任务三模型产生了非有限预测")
        return prediction


def build_evaluation_model(task: str):
    if task == "task1":
        return Task1EvaluationRegressor()
    if task == "task2":
        return Task2EvaluationRegressor()
    if task == "task3":
        return Task3EvaluationRegressor()
    raise ValueError(f"未知任务: {task}")


def validate_prediction_input(frame: pd.DataFrame) -> None:
    frame = _as_frame(frame)
    if ID_COLUMN not in frame.columns:
        raise ValueError("预测输入缺少Person_ID")
    if frame[ID_COLUMN].isna().any():
        raise ValueError("Person_ID不能为空")
    if frame[ID_COLUMN].duplicated().any():
        raise ValueError("Person_ID不能重复")


def prediction_frame(model, frame: pd.DataFrame) -> pd.DataFrame:
    validate_prediction_input(frame)
    prediction = np.asarray(model.predict(frame), dtype=float).reshape(-1)
    if len(prediction) != len(frame):
        raise ValueError("模型预测行数与输入不一致")
    return pd.DataFrame({
        ID_COLUMN: frame[ID_COLUMN].to_numpy(),
        "predicted_value": prediction,
    })


def load_development_frame(root: Path) -> pd.DataFrame:
    data_path = root / "data" / "raw" / "A题数据集.csv"
    split_path = root / "data" / "splits" / "split_assignments.csv"
    raw = pd.read_csv(data_path)
    assignments = pd.read_csv(split_path)
    development_ids = assignments.loc[
        assignments["split"] == "development", [ID_COLUMN]
    ]
    development = development_ids.merge(
        raw,
        on=ID_COLUMN,
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if len(development) != 8000 or development[ID_COLUMN].duplicated().any():
        raise ValueError("最终评估模型开发集划分不完整")
    if not development["_merge"].eq("both").all():
        raise ValueError("开发集ID无法映射到原始数据")
    return development.drop(columns="_merge")


def _git_state(root: Path) -> dict[str, object]:
    def run(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args], cwd=root, check=False, capture_output=True, text=True
        )
        return completed.stdout.strip()

    return {
        "branch": run("branch", "--show-current"),
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(run("status", "--porcelain")),
    }


def _load_model_config(root: Path) -> dict[str, Any]:
    with (root / MODEL_CONFIG_PATH).open("rb") as stream:
        return tomllib.load(stream)


def _task_manifest(
    root: Path,
    task: str,
    model,
    model_path: Path,
    training_seconds: float,
    round_trip_max_abs_difference: float,
) -> dict[str, Any]:
    protocol = validate_final_evaluation_protocol(root)
    frozen = protocol["tasks"][task]
    return {
        "task": task,
        "target": TARGETS[task],
        "status": "evaluation_model_trained",
        "role": "internal_holdout_evaluation",
        "selected_model": frozen["selected_model"],
        "selection_run_id": frozen["selection_run_id"],
        "confirmation_run_id": frozen["confirmation_run_id"],
        "training_split": "development",
        "training_rows": 8000,
        "sealed_holdout_rows": 2000,
        "holdout_labels_used": False,
        "holdout_evaluated": False,
        "required_columns": list(model.required_columns_),
        "required_column_count": len(model.required_columns_),
        "accepts_extra_columns": True,
        "model_path": str(model_path.relative_to(root)).replace("\\", "/"),
        "model_sha256": sha256_file(model_path),
        "candidate_config_sha256": frozen["candidate_config_sha256"],
        "candidate_manifest_sha256": frozen["candidate_manifest_sha256"],
        "data_sha256": protocol["data_sha256"],
        "split_sha256": protocol["split_sha256"],
        "protocol_sha256": sha256_file(root / PROTOCOL_PATH),
        "training_seconds": training_seconds,
        "serialization_round_trip_max_abs_difference": round_trip_max_abs_difference,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "dependencies": {
            "joblib": joblib.__version__,
            "lightgbm": lightgbm.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }


def train_final_evaluation_models(root: Path, overwrite: bool = False) -> list[Path]:
    protocol = validate_final_evaluation_protocol(root)
    if protocol["holdout_evaluation_authorized"]:
        raise ValueError("训练脚本不得在留出集评估已授权状态下运行")
    config = _load_model_config(root)
    development = load_development_frame(root)
    source_git_state = _git_state(root)
    model_root = root / config["training"]["model_directory"]
    model_root.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []
    manifests: dict[str, dict[str, Any]] = {}
    started_at = datetime.now().astimezone()
    total_started = time.perf_counter()

    planned_paths: list[Path] = []
    for task in TASKS:
        artifact = config["artifacts"][task]
        planned_paths.extend([
            model_root / artifact["filename"],
            root / artifact["manifest"],
        ])
    combined_path = root / config["training"]["combined_manifest"]
    planned_paths.append(combined_path)
    if not overwrite and any(path.exists() for path in planned_paths):
        raise FileExistsError("评估模型产物已存在；确认重建时请使用 --overwrite")

    for task in TASKS:
        artifact = config["artifacts"][task]
        target = TARGETS[task]
        model_path = model_root / artifact["filename"]
        manifest_path = root / artifact["manifest"]
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        model = build_evaluation_model(task)
        fit_started = time.perf_counter()
        model.fit(development, pd.to_numeric(development[target]))
        training_seconds = time.perf_counter() - fit_started
        sample = development.iloc[:32].copy()
        before = np.asarray(model.predict(sample), dtype=float)
        joblib.dump(model, model_path, compress=3)
        loaded = joblib.load(model_path)
        after = np.asarray(loaded.predict(sample), dtype=float)
        maximum_difference = float(np.max(np.abs(before - after)))
        if maximum_difference > 1e-12:
            raise ValueError(f"{task}模型保存前后预测不一致")
        manifest = _task_manifest(
            root,
            task,
            loaded,
            model_path,
            training_seconds,
            maximum_difference,
        )
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        manifests[task] = manifest
        output_paths.extend([model_path, manifest_path])
        print(f"{task}: evaluation model trained and serialization verified")

    run_identity = hashlib.sha256(
        (
            "final_evaluation_model_training|"
            f"{protocol['data_sha256']}|{protocol['split_sha256']}|"
            f"{sha256_file(root / MODEL_CONFIG_PATH)}"
        ).encode()
    ).hexdigest()[:8]
    run_id = f"{started_at.strftime('%Y%m%dT%H%M%S%z')}_evaluation_models_{run_identity}"
    combined = {
        "run_id": run_id,
        "status": "completed",
        "experiment": "final_evaluation_model_training",
        "experiment_role": "pre_holdout_model_fit",
        "command": "python scripts/train_final_evaluation_models.py",
        "git": source_git_state,
        "training_split": "development",
        "training_rows": 8000,
        "sealed_holdout_rows": 2000,
        "holdout_labels_used": False,
        "holdout_evaluated": False,
        "tasks": manifests,
        "data_sha256": protocol["data_sha256"],
        "split_sha256": protocol["split_sha256"],
        "models_config_sha256": sha256_file(root / MODEL_CONFIG_PATH),
        "protocol_sha256": sha256_file(root / PROTOCOL_PATH),
        "duration_seconds": time.perf_counter() - total_started,
    }
    combined_text = json.dumps(combined, ensure_ascii=False, indent=2)
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    combined_path.write_text(combined_text, encoding="utf-8")
    history_path = root / "outputs" / "logs" / "history" / f"{run_id}_manifest.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(combined_text, encoding="utf-8")
    output_paths.extend([combined_path, history_path])
    return output_paths
