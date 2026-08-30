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

import pandas as pd


PROTOCOL_PATH = Path("configs/final_evaluation_protocol.toml")
TASK_NAMES = ("task1", "task2", "task3")
EXPECTED_TARGETS = {
    "task1": "Sleep_Quality_Score",
    "task2": "Productivity_Score",
    "task3": "Health_Score",
}
PAPER_ARTIFACT_PROHIBITIONS = {
    "paper_table_code",
    "paper_figure_code",
    "paper_evidence_index_code",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_protocol(root: Path) -> dict[str, Any]:
    path = root / PROTOCOL_PATH
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("rb") as stream:
        return tomllib.load(stream)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _load_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("rb") as stream:
        return tomllib.load(stream)


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


def require_holdout_confirmation(root: Path, supplied_token: str) -> None:
    protocol = load_protocol(root)
    expected = protocol["protocol"]["holdout_confirmation_token"]
    if supplied_token != expected:
        raise ValueError("最终留出集确认口令不匹配")
    if protocol["evaluation_gate"]["holdout_evaluation_authorized"]:
        raise ValueError("协议文件不得预先授权留出集评估")


def _validate_split(assignments: pd.DataFrame, protocol: dict[str, Any]) -> dict[str, object]:
    settings = protocol["protocol"]
    required = {settings["id_column"], "split", "cv_fold"}
    missing = required - set(assignments.columns)
    if missing:
        raise ValueError(f"冻结划分缺少字段: {sorted(missing)}")
    id_column = settings["id_column"]
    if assignments[id_column].isna().any() or assignments[id_column].duplicated().any():
        raise ValueError("冻结划分中的Person_ID必须非空且唯一")

    development = assignments.loc[assignments["split"] == "development"]
    holdout = assignments.loc[assignments["split"] == "holdout"]
    if len(development) != settings["development_rows"]:
        raise ValueError("开发集样本数与最终评估协议不一致")
    if len(holdout) != settings["holdout_rows"]:
        raise ValueError("留出集样本数与最终评估协议不一致")
    if len(assignments) != settings["development_rows"] + settings["holdout_rows"]:
        raise ValueError("冻结划分包含未知split或样本总数异常")
    if not (holdout["cv_fold"] == -1).all():
        raise ValueError("留出集cv_fold必须全部为-1")

    fold_counts = development["cv_fold"].value_counts().sort_index()
    expected_folds = list(range(settings["cv_folds"]))
    if fold_counts.index.tolist() != expected_folds:
        raise ValueError("开发集五折编号不完整")
    if fold_counts.nunique() != 1:
        raise ValueError("开发集五折样本数不均衡")
    return {
        "rows": int(len(assignments)),
        "development_rows": int(len(development)),
        "holdout_rows": int(len(holdout)),
        "fold_counts": {str(int(key)): int(value) for key, value in fold_counts.items()},
        "unique_ids": int(assignments[id_column].nunique()),
    }


def validate_final_evaluation_protocol(root: Path) -> dict[str, Any]:
    protocol = load_protocol(root)
    settings = protocol["protocol"]
    if settings["status"] != "pre_holdout_ready":
        raise ValueError("最终评估协议状态必须为pre_holdout_ready")
    if settings["holdout_policy"] != "sealed_one_time":
        raise ValueError("最终留出集必须采用sealed_one_time策略")
    if settings["allow_repeated_holdout_evaluation"]:
        raise ValueError("最终评估协议禁止重复留出集评估")

    evaluation_gate = protocol["evaluation_gate"]
    if not evaluation_gate["evaluation_model_training_authorized"]:
        raise ValueError("当前协议应允许开发集评估模型训练")
    if evaluation_gate["holdout_evaluation_authorized"]:
        raise ValueError("当前阶段不得授权留出集评估")
    for field in (
        "require_serialization_round_trip",
        "require_raw_schema_prediction_test",
        "require_model_hash_verification",
        "require_exact_confirmation_token",
    ):
        if not evaluation_gate[field]:
            raise ValueError(f"最终评估安全门禁未启用: {field}")

    paper_gate = protocol["paper_artifact_gate"]
    if paper_gate["status"] != "blocked_until_manual_confirmation":
        raise ValueError("论文材料门禁必须保持人工确认前阻塞")
    if not paper_gate["requires_user_confirmation"] or not paper_gate["stop_and_remind_user"]:
        raise ValueError("论文材料门禁必须要求人工确认并主动提醒")
    if set(paper_gate["prohibited_before_confirmation"]) != PAPER_ARTIFACT_PROHIBITIONS:
        raise ValueError("论文材料门禁的禁止项不完整")

    data_path = root / settings["data_path"]
    split_path = root / settings["split_path"]
    if not data_path.exists():
        raise FileNotFoundError(data_path)
    assignments = pd.read_csv(split_path)
    split_summary = _validate_split(assignments, protocol)
    data_hash = sha256_file(data_path)
    split_hash = sha256_file(split_path)

    task_results: dict[str, dict[str, Any]] = {}
    configured_tasks = protocol.get("tasks", {})
    if tuple(configured_tasks) != TASK_NAMES:
        raise ValueError("最终评估协议必须按task1、task2、task3完整声明任务")

    for task in TASK_NAMES:
        task_protocol = configured_tasks[task]
        if task_protocol["target"] != EXPECTED_TARGETS[task]:
            raise ValueError(f"{task}目标字段与冻结定义不一致")
        candidate_config_path = root / task_protocol["candidate_config"]
        candidate_manifest_path = root / task_protocol["candidate_manifest"]
        candidate_config = _load_toml(candidate_config_path)
        candidate_manifest = _load_json(candidate_manifest_path)

        if candidate_manifest["task"] != task:
            raise ValueError(f"{task}冻结清单任务名不一致")
        if candidate_manifest["target"] != task_protocol["target"]:
            raise ValueError(f"{task}冻结清单目标字段不一致")
        if candidate_manifest["status"] != "development_candidate_frozen":
            raise ValueError(f"{task}尚未冻结开发集候选")
        if candidate_manifest["selected_model"] != task_protocol["selected_model"]:
            raise ValueError(f"{task}最终协议模型与冻结清单不一致")
        if candidate_manifest["holdout_evaluated"]:
            raise ValueError(f"{task}留出集已被消费，不能建立首次最终评估协议")
        if candidate_manifest["final_model_trained"]:
            raise ValueError(f"{task}已存在最终模型训练标记，状态顺序异常")
        if candidate_manifest["data_sha256"].lower() != data_hash.lower():
            raise ValueError(f"{task}冻结清单数据哈希不一致")
        if candidate_manifest["split_sha256"].lower() != split_hash.lower():
            raise ValueError(f"{task}冻结清单划分哈希不一致")
        if candidate_manifest["candidate_config_sha256"].lower() != sha256_file(candidate_config_path).lower():
            raise ValueError(f"{task}冻结候选配置哈希不一致")

        if task in {"task1", "task2"}:
            candidate_section = candidate_config["candidate"]
            if candidate_section["status"] != "development_candidate_frozen":
                raise ValueError(f"{task}候选配置状态异常")
            if candidate_section["holdout_evaluated"]:
                raise ValueError(f"{task}候选配置错误标记留出集已评估")
        else:
            experiment = candidate_config["experiment"]
            if experiment["status"] != "development_candidate_frozen":
                raise ValueError("task3候选配置状态异常")
            if experiment["holdout_evaluated"]:
                raise ValueError("task3候选配置错误标记留出集已评估")

        task_results[task] = {
            "target": task_protocol["target"],
            "selected_model": task_protocol["selected_model"],
            "primary_metric": task_protocol["primary_metric"],
            "selection_run_id": candidate_manifest["selection_run_id"],
            "confirmation_run_id": candidate_manifest["confirmation_run_id"],
            "candidate_config_sha256": sha256_file(candidate_config_path),
            "candidate_manifest_sha256": sha256_file(candidate_manifest_path),
            "holdout_evaluated": False,
            "final_model_trained": False,
        }

    return {
        "protocol_status": settings["status"],
        "data_sha256": data_hash,
        "split_sha256": split_hash,
        "split": split_summary,
        "tasks": task_results,
        "evaluation_model_training_authorized": True,
        "holdout_evaluation_authorized": False,
        "paper_artifact_gate": paper_gate["status"],
    }


def write_protocol_audit(root: Path, overwrite: bool = False) -> list[Path]:
    output_path = root / "outputs" / "logs" / "final_evaluation_protocol_manifest.json"
    history_root = root / "outputs" / "logs" / "history"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    history_root.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise FileExistsError("最终评估协议清单已存在；确认重建时请使用 --overwrite")

    started_at = datetime.now().astimezone()
    started = time.perf_counter()
    validation = validate_final_evaluation_protocol(root)
    protocol_path = root / PROTOCOL_PATH
    identity = hashlib.sha256(
        (
            "final_evaluation_protocol|"
            f"{validation['data_sha256']}|{validation['split_sha256']}|{sha256_file(protocol_path)}"
        ).encode()
    ).hexdigest()[:8]
    run_id = f"{started_at.strftime('%Y%m%dT%H%M%S%z')}_final_evaluation_protocol_{identity}"
    manifest = {
        "run_id": run_id,
        "status": "completed",
        "experiment": "final_evaluation_protocol_audit",
        "experiment_role": "pre_holdout_gate",
        "command": "python scripts/check_final_evaluation_protocol.py",
        "git": _git_state(root),
        "protocol_sha256": sha256_file(protocol_path),
        **validation,
        "duration_seconds": time.perf_counter() - started,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
    }
    text = json.dumps(manifest, ensure_ascii=False, indent=2)
    output_path.write_text(text, encoding="utf-8")
    history_path = history_root / f"{run_id}_manifest.json"
    history_path.write_text(text, encoding="utf-8")
    return [output_path, history_path]
