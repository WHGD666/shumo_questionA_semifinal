from __future__ import annotations

import hashlib
import json
import subprocess
import time
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.final_evaluation_models import ID_COLUMN, TARGETS, TASKS
from src.final_evaluation_protocol import (
    PROTOCOL_PATH,
    sha256_file,
    validate_final_evaluation_protocol,
)
from src.task2_protocol import adjusted_r2


HOLDOUT_CONFIG_PATH = Path("configs/final_holdout_evaluation.toml")
MODEL_TRAINING_CONFIG_PATH = Path("configs/final_evaluation_models.toml")


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def require_exact_confirmation(root: Path, confirmation: str) -> None:
    protocol = _load_toml(root / PROTOCOL_PATH)
    expected = protocol["protocol"]["holdout_confirmation_token"]
    if confirmation != expected:
        raise ValueError("最终留出集评估需要精确确认口令")
    if protocol["protocol"]["allow_repeated_holdout_evaluation"]:
        raise ValueError("一次性留出集协议不得允许重复评估")


def configured_output_paths(root: Path) -> dict[str, Path]:
    config = _load_toml(root / HOLDOUT_CONFIG_PATH)
    paths = {
        "metrics": root / config["evaluation"]["metrics_path"],
        "manifest": root / config["evaluation"]["manifest_path"],
        "consumed_marker": root / config["evaluation"]["consumed_marker_path"],
    }
    for task in TASKS:
        paths[f"{task}_predictions"] = root / config["predictions"][task]
    return paths


def require_fresh_holdout_outputs(root: Path) -> dict[str, Path]:
    paths = configured_output_paths(root)
    existing = [str(path.relative_to(root)) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "最终留出集已经评估或存在部分产物，禁止覆盖: " + ", ".join(existing)
        )
    return paths


def load_partition_frame(
    data_path: Path,
    split_path: Path,
    split_name: str,
    expected_rows: int,
) -> pd.DataFrame:
    raw = pd.read_csv(data_path)
    assignments = pd.read_csv(split_path)
    if ID_COLUMN not in raw.columns or ID_COLUMN not in assignments.columns:
        raise ValueError("数据或划分文件缺少Person_ID")
    if raw[ID_COLUMN].duplicated().any() or assignments[ID_COLUMN].duplicated().any():
        raise ValueError("数据或划分文件包含重复Person_ID")
    selected_ids = assignments.loc[
        assignments["split"] == split_name,
        [ID_COLUMN],
    ]
    frame = selected_ids.merge(
        raw,
        on=ID_COLUMN,
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if len(frame) != expected_rows:
        raise ValueError(f"{split_name}样本数不等于{expected_rows}")
    if not frame["_merge"].eq("both").all():
        raise ValueError(f"{split_name}中的Person_ID无法映射到原始数据")
    if frame[ID_COLUMN].duplicated().any():
        raise ValueError(f"{split_name}包含重复Person_ID")
    return frame.drop(columns="_merge")


def verify_evaluation_model_artifacts(root: Path) -> dict[str, dict[str, Any]]:
    training_config = _load_toml(root / MODEL_TRAINING_CONFIG_PATH)
    holdout_config = _load_toml(root / HOLDOUT_CONFIG_PATH)
    combined_path = root / holdout_config["evaluation"]["model_training_manifest"]
    if not combined_path.exists():
        raise FileNotFoundError(combined_path)
    combined = json.loads(combined_path.read_text(encoding="utf-8"))
    if combined.get("status") != "completed":
        raise ValueError("最终评估模型训练清单状态不是completed")
    if combined.get("training_split") != "development":
        raise ValueError("最终评估模型不是仅由开发集训练")
    if combined.get("holdout_labels_used") or combined.get("holdout_evaluated"):
        raise ValueError("模型训练清单显示留出集已经被使用")

    verified: dict[str, dict[str, Any]] = {}
    for task in TASKS:
        artifact = training_config["artifacts"][task]
        model_path = root / training_config["training"]["model_directory"] / artifact["filename"]
        manifest_path = root / artifact["manifest"]
        if not model_path.exists() or not manifest_path.exists():
            raise FileNotFoundError(f"{task}评估模型或清单不存在")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("training_split") != "development":
            raise ValueError(f"{task}模型不是仅由开发集训练")
        if manifest.get("holdout_labels_used") or manifest.get("holdout_evaluated"):
            raise ValueError(f"{task}模型清单显示留出集已经被使用")
        actual_hash = sha256_file(model_path)
        if actual_hash != manifest.get("model_sha256"):
            raise ValueError(f"{task}模型哈希与训练清单不一致")
        combined_task = combined.get("tasks", {}).get(task, {})
        if combined_task.get("model_sha256") != actual_hash:
            raise ValueError(f"{task}模型哈希与汇总训练清单不一致")
        verified[task] = {
            "model_path": model_path,
            "manifest_path": manifest_path,
            "manifest": manifest,
            "model_sha256": actual_hash,
        }
    return verified


def build_holdout_prediction_frame(
    person_ids: pd.Series,
    truth: np.ndarray,
    prediction: np.ndarray,
) -> pd.DataFrame:
    truth = np.asarray(truth, dtype=float).reshape(-1)
    prediction = np.asarray(prediction, dtype=float).reshape(-1)
    if len(person_ids) != len(truth) or len(truth) != len(prediction):
        raise ValueError("留出集ID、真值和预测行数不一致")
    if person_ids.isna().any() or person_ids.duplicated().any():
        raise ValueError("留出集Person_ID为空或重复")
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("留出集真值或预测包含非有限数值")
    return pd.DataFrame({
        ID_COLUMN: person_ids.to_numpy(),
        "true_value": truth,
        "predicted_value": prediction,
        "residual": truth - prediction,
    })


def regression_metric_row(
    task: str,
    truth: np.ndarray,
    prediction: np.ndarray,
    raw_predictor_count: int,
) -> dict[str, Any]:
    truth = np.asarray(truth, dtype=float).reshape(-1)
    prediction = np.asarray(prediction, dtype=float).reshape(-1)
    if len(truth) != len(prediction) or len(truth) < 2:
        raise ValueError("指标计算所需真值和预测长度无效")
    r2 = float(r2_score(truth, prediction))
    mae = float(mean_absolute_error(truth, prediction))
    rmse = float(mean_squared_error(truth, prediction) ** 0.5)
    adjusted = np.nan
    primary_metric = "r2"
    primary_value = r2
    if task == "task2":
        adjusted = adjusted_r2(r2, n=len(truth), p=raw_predictor_count)
        primary_metric = "adjusted_r2_raw"
        primary_value = adjusted
    return {
        "task": task,
        "target": TARGETS[task],
        "evaluation_role": "internal_frozen_holdout",
        "sample_count": len(truth),
        "raw_predictor_count": raw_predictor_count,
        "primary_metric": primary_metric,
        "primary_value": primary_value,
        "r2": r2,
        "adjusted_r2_raw": adjusted,
        "mae": mae,
        "rmse": rmse,
        "official_hidden_test_score": False,
    }


def validate_task_output_consistency(
    task: str,
    prediction_frame: pd.DataFrame,
    metric_row: dict[str, Any],
    expected_ids: pd.Series,
    raw_predictor_count: int,
) -> None:
    expected_columns = [ID_COLUMN, "true_value", "predicted_value", "residual"]
    if prediction_frame.columns.tolist() != expected_columns:
        raise ValueError(f"{task}留出集预测文件字段不符合约定")
    if prediction_frame[ID_COLUMN].tolist() != expected_ids.tolist():
        raise ValueError(f"{task}留出集预测文件的Person_ID顺序不一致")
    if prediction_frame[ID_COLUMN].duplicated().any():
        raise ValueError(f"{task}留出集预测文件包含重复Person_ID")
    truth = pd.to_numeric(prediction_frame["true_value"], errors="raise").to_numpy()
    prediction = pd.to_numeric(
        prediction_frame["predicted_value"], errors="raise"
    ).to_numpy()
    residual = pd.to_numeric(prediction_frame["residual"], errors="raise").to_numpy()
    if not np.allclose(residual, truth - prediction, rtol=0.0, atol=1e-12):
        raise ValueError(f"{task}留出集预测文件残差与真值、预测不一致")
    recalculated = regression_metric_row(
        task,
        truth,
        prediction,
        raw_predictor_count=raw_predictor_count,
    )
    for name in ("primary_value", "r2", "mae", "rmse"):
        if not np.isclose(
            float(metric_row[name]),
            float(recalculated[name]),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(f"{task}留出集指标{name}与预测文件不一致")
    expected_adjusted = recalculated["adjusted_r2_raw"]
    actual_adjusted = metric_row["adjusted_r2_raw"]
    if np.isnan(expected_adjusted):
        if not pd.isna(actual_adjusted):
            raise ValueError(f"{task}不应保存调整R²")
    elif not np.isclose(
        float(actual_adjusted),
        float(expected_adjusted),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(f"{task}调整R²与预测文件不一致")


def _git_state(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    return {
        "branch": run("branch", "--show-current"),
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(run("status", "--porcelain")),
    }


def evaluate_final_holdout(root: Path, confirmation: str) -> list[Path]:
    require_exact_confirmation(root, confirmation)
    protocol = validate_final_evaluation_protocol(root)
    if protocol["holdout_evaluation_authorized"]:
        raise ValueError("静态协议不得预授权一次性留出集评估")
    paths = require_fresh_holdout_outputs(root)
    artifacts = verify_evaluation_model_artifacts(root)
    source_git_state = _git_state(root)
    config = _load_toml(root / HOLDOUT_CONFIG_PATH)
    data_path = root / "data" / "raw" / "A题数据集.csv"
    split_path = root / "data" / "splits" / "split_assignments.csv"
    started_at = datetime.now().astimezone()
    started = time.perf_counter()

    holdout = load_partition_frame(
        data_path,
        split_path,
        split_name=config["evaluation"]["split"],
        expected_rows=int(config["evaluation"]["expected_rows"]),
    )
    metric_rows: list[dict[str, Any]] = []
    task_records: dict[str, dict[str, Any]] = {}
    pending_task_records: dict[str, dict[str, Any]] = {}
    written: list[Path] = []
    for task in TASKS:
        target = TARGETS[task]
        truth = pd.to_numeric(holdout[target], errors="raise").to_numpy(dtype=float)
        model = joblib.load(artifacts[task]["model_path"])
        prediction = np.asarray(model.predict(holdout), dtype=float).reshape(-1)
        prediction_frame = build_holdout_prediction_frame(
            holdout[ID_COLUMN], truth, prediction
        )
        prediction_path = paths[f"{task}_predictions"]
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        prediction_frame.to_csv(prediction_path, index=False, encoding="utf-8")
        raw_predictor_count = int(
            artifacts[task]["manifest"]["required_column_count"]
        )
        metric = regression_metric_row(
            task,
            truth,
            prediction,
            raw_predictor_count=raw_predictor_count,
        )
        metric_rows.append(metric)
        pending_task_records[task] = {
            "target": target,
            "selected_model": artifacts[task]["manifest"]["selected_model"],
            "model_sha256": artifacts[task]["model_sha256"],
            "model_training_manifest": str(
                artifacts[task]["manifest_path"].relative_to(root)
            ).replace("\\", "/"),
            "prediction_path": str(prediction_path.relative_to(root)).replace("\\", "/"),
            "metrics": metric,
        }
        written.append(prediction_path)

    metrics_frame = pd.DataFrame(metric_rows)
    paths["metrics"].parent.mkdir(parents=True, exist_ok=True)
    metrics_frame.to_csv(paths["metrics"], index=False, encoding="utf-8")
    written.append(paths["metrics"])

    persisted_metrics = pd.read_csv(paths["metrics"])
    if persisted_metrics["task"].tolist() != list(TASKS):
        raise ValueError("最终留出集指标文件的任务顺序或数量不一致")
    for task in TASKS:
        persisted_prediction = pd.read_csv(paths[f"{task}_predictions"])
        persisted_metric = persisted_metrics.loc[
            persisted_metrics["task"] == task
        ].iloc[0].to_dict()
        raw_predictor_count = int(
            artifacts[task]["manifest"]["required_column_count"]
        )
        validate_task_output_consistency(
            task,
            persisted_prediction,
            persisted_metric,
            holdout[ID_COLUMN],
            raw_predictor_count,
        )
        task_records[task] = {
            **pending_task_records[task],
            "prediction_sha256": sha256_file(paths[f"{task}_predictions"]),
            "metrics": persisted_metric,
        }

    identity = hashlib.sha256(
        (
            "final_holdout|"
            f"{protocol['data_sha256']}|{protocol['split_sha256']}|"
            + "|".join(artifacts[task]["model_sha256"] for task in TASKS)
        ).encode()
    ).hexdigest()[:8]
    run_id = f"{started_at.strftime('%Y%m%dT%H%M%S%z')}_final_holdout_{identity}"
    manifest = {
        "run_id": run_id,
        "status": "completed",
        "experiment": "one_time_final_internal_holdout_evaluation",
        "experiment_role": "final_internal_holdout",
        "official_hidden_test_score": False,
        "confirmation_verified": True,
        "confirmation_token_stored": False,
        "holdout_evaluated": True,
        "holdout_consumed": True,
        "holdout_rows": len(holdout),
        "git": source_git_state,
        "data_sha256": protocol["data_sha256"],
        "split_sha256": protocol["split_sha256"],
        "protocol_sha256": sha256_file(root / PROTOCOL_PATH),
        "evaluation_config_sha256": sha256_file(root / HOLDOUT_CONFIG_PATH),
        "metrics_path": str(paths["metrics"].relative_to(root)).replace("\\", "/"),
        "metrics_sha256": sha256_file(paths["metrics"]),
        "tasks": task_records,
        "duration_seconds": time.perf_counter() - started,
    }
    manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2)
    paths["manifest"].parent.mkdir(parents=True, exist_ok=True)
    paths["manifest"].write_text(manifest_text, encoding="utf-8")
    history_path = (
        root
        / config["evaluation"]["history_directory"]
        / f"{run_id}_manifest.json"
    )
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(manifest_text, encoding="utf-8")
    written.extend([paths["manifest"], history_path])

    marker = {
        "status": "consumed",
        "holdout_consumed": True,
        "run_id": run_id,
        "consumed_at": started_at.isoformat(),
        "evaluation_manifest": str(paths["manifest"].relative_to(root)).replace("\\", "/"),
        "evaluation_manifest_sha256": sha256_file(paths["manifest"]),
        "repeat_evaluation_allowed": False,
    }
    paths["consumed_marker"].write_text(
        json.dumps(marker, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    written.append(paths["consumed_marker"])
    return written
