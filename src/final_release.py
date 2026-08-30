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

from src.final_evaluation_models import (
    ID_COLUMN,
    TARGETS,
    TASKS,
    build_evaluation_model,
)
from src.final_evaluation_protocol import sha256_file


RELEASE_CONFIG_PATH = Path("configs/final_release.toml")


def _strict_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"JSON包含非标准常量: {value}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


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


def load_release_config(root: Path) -> dict[str, Any]:
    with (root / RELEASE_CONFIG_PATH).open("rb") as stream:
        return tomllib.load(stream)


def validate_release_gate(root: Path) -> dict[str, Any]:
    config = load_release_config(root)
    release = config["release"]
    if release["training_data"] != "all_labeled_rows":
        raise ValueError("最终发布重训必须明确使用全部有标签数据")
    if release["holdout_used_for_model_selection"]:
        raise ValueError("一次性留出集不得参与模型选择")
    if release["post_holdout_tuning"]:
        raise ValueError("一次性留出集评估后禁止继续调参")

    sources = config["sources"]
    consumed = _strict_json(root / sources["holdout_consumed_marker"])
    holdout = _strict_json(root / sources["holdout_manifest"])
    registry = _strict_json(root / sources["results_registry_manifest"])
    if consumed.get("status") != "consumed" or not consumed.get("holdout_consumed"):
        raise ValueError("最终发布重训前必须完成一次性留出集评估")
    if not holdout.get("holdout_consumed"):
        raise ValueError("留出集评估清单未记录消费状态")
    if not registry.get("all_source_checks_passed"):
        raise ValueError("最终结果注册表审计尚未通过")
    if registry.get("official_hidden_test_score"):
        raise ValueError("内部结果不得标记为官方隐藏测试成绩")
    if consumed.get("run_id") != holdout.get("run_id"):
        raise ValueError("留出集消费标记与评估清单运行编号不一致")
    return {
        "config": config,
        "holdout_run_id": holdout["run_id"],
        "registry_run_id": registry["run_id"],
    }


def load_all_labeled_rows(root: Path, config: dict[str, Any]) -> pd.DataFrame:
    data_path = root / config["sources"]["data_path"]
    frame = pd.read_csv(data_path)
    if len(frame) != int(config["release"]["training_rows"]):
        raise ValueError("最终发布重训数据行数与冻结配置不一致")
    if ID_COLUMN not in frame.columns:
        raise ValueError("最终发布重训数据缺少Person_ID")
    if frame[ID_COLUMN].isna().any() or frame[ID_COLUMN].duplicated().any():
        raise ValueError("最终发布重训数据的Person_ID必须唯一且非空")
    for target in TARGETS.values():
        values = pd.to_numeric(frame[target], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"最终发布重训目标无效: {target}")
    return frame


def _task_manifest(
    root: Path,
    gate: dict[str, Any],
    task: str,
    model: Any,
    model_path: Path,
    training_seconds: float,
    round_trip_max_abs_difference: float,
) -> dict[str, Any]:
    config = gate["config"]
    artifact = config["artifacts"][task]
    required = list(model.required_columns_)
    target = artifact["target"]
    if target in required or ID_COLUMN in required:
        raise ValueError(f"{task}最终发布模型字段契约包含目标或标识符")
    return {
        "task": task,
        "target": target,
        "status": "release_model_trained",
        "training_role": config["release"]["training_role"],
        "selected_model": artifact["selected_model"],
        "training_rows": int(config["release"]["training_rows"]),
        "training_data": config["release"]["training_data"],
        "holdout_labels_used_for_refit": True,
        "holdout_used_for_model_selection": False,
        "post_holdout_tuning": False,
        "input_contract_mode": config["input_contract"]["mode"],
        "required_columns": required,
        "required_column_count": len(required),
        "accepts_extra_columns": bool(config["input_contract"]["accepts_extra_columns"]),
        "model_path": str(model_path.relative_to(root)).replace("\\", "/"),
        "model_sha256": sha256_file(model_path),
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


def train_final_release_models(root: Path, overwrite: bool = False) -> list[Path]:
    gate = validate_release_gate(root)
    config = gate["config"]
    frame = load_all_labeled_rows(root, config)
    model_root = root / config["release"]["model_directory"]
    model_root.mkdir(parents=True, exist_ok=True)
    combined_path = root / config["release"]["combined_manifest"]
    planned = [
        model_root / config["artifacts"][task]["filename"] for task in TASKS
    ] + [combined_path]
    if not overwrite and any(path.exists() for path in planned):
        raise FileExistsError("最终发布模型已存在；确认重建时请使用 --overwrite")

    started_at = datetime.now().astimezone()
    total_started = time.perf_counter()
    source_git = _git_state(root)
    manifests: dict[str, dict[str, Any]] = {}
    written: list[Path] = []
    for task in TASKS:
        artifact = config["artifacts"][task]
        target = artifact["target"]
        model = build_evaluation_model(task)
        fit_started = time.perf_counter()
        model.fit(frame, pd.to_numeric(frame[target]))
        training_seconds = time.perf_counter() - fit_started
        model_path = model_root / artifact["filename"]
        joblib.dump(model, model_path, compress=3)
        restored = joblib.load(model_path)
        sample = frame.iloc[:32]
        before = np.asarray(model.predict(sample), dtype=float)
        after = np.asarray(restored.predict(sample), dtype=float)
        difference = float(np.max(np.abs(before - after)))
        if difference > 1e-12:
            raise ValueError(f"{task}最终发布模型保存前后预测不一致")
        manifests[task] = _task_manifest(
            root, gate, task, restored, model_path, training_seconds, difference
        )
        written.append(model_path)
        print(f"{task}: final release model trained and verified")

    identity = hashlib.sha256(
        (
            "final_release_refit|"
            f"{sha256_file(root / config['sources']['data_path'])}|"
            f"{sha256_file(root / RELEASE_CONFIG_PATH)}|{gate['holdout_run_id']}"
        ).encode()
    ).hexdigest()[:8]
    run_id = f"{started_at.strftime('%Y%m%dT%H%M%S%z')}_final_release_{identity}"
    combined = {
        "run_id": run_id,
        "status": "completed",
        "experiment": "final_release_model_training",
        "training_role": config["release"]["training_role"],
        "command": "python scripts/train_final_release_models.py",
        "git": source_git,
        "training_rows": len(frame),
        "training_data": config["release"]["training_data"],
        "holdout_labels_used_for_refit": True,
        "holdout_used_for_model_selection": False,
        "post_holdout_tuning": False,
        "holdout_run_id": gate["holdout_run_id"],
        "registry_run_id": gate["registry_run_id"],
        "input_contract": config["input_contract"],
        "tasks": manifests,
        "data_sha256": sha256_file(root / config["sources"]["data_path"]),
        "release_config_sha256": sha256_file(root / RELEASE_CONFIG_PATH),
        "results_registry_sha256": sha256_file(root / config["sources"]["results_registry"]),
        "duration_seconds": time.perf_counter() - total_started,
    }
    text = json.dumps(combined, ensure_ascii=False, indent=2, allow_nan=False)
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    combined_path.write_text(text, encoding="utf-8")
    history_path = root / "outputs" / "logs" / "history" / f"{run_id}_manifest.json"
    history_path.write_text(text, encoding="utf-8")
    written.extend([combined_path, history_path])
    return written
