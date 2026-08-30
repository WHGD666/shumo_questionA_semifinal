from __future__ import annotations

import hashlib
import json
import subprocess
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.final_evaluation_models import TASKS
from src.final_evaluation_protocol import sha256_file
from src.final_holdout_evaluation import validate_task_output_consistency


REGISTRY_CONFIG_PATH = Path("configs/final_results_registry.toml")
FROZEN_CONFIG_PATHS = {
    "task1": Path("configs/task1_frozen_candidate.toml"),
    "task2": Path("configs/task2_frozen_candidate.toml"),
    "task3": Path("configs/task3_frozen_candidate.toml"),
}
FROZEN_MANIFEST_PATHS = {
    task: Path(f"outputs/logs/{task}_frozen_candidate_manifest.json")
    for task in TASKS
}


def load_json_strict(path: Path) -> dict[str, Any]:
    def reject_constant(value: str):
        raise ValueError(f"非标准JSON常量: {value}")

    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_constant,
    )


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def _assert_close(actual: Any, expected: Any, label: str) -> None:
    if pd.isna(actual) and pd.isna(expected):
        return
    if not np.isclose(float(actual), float(expected), rtol=0.0, atol=1e-12):
        raise ValueError(f"{label}不一致: {actual} != {expected}")


def registry_output_paths(root: Path) -> dict[str, Path]:
    config = _load_toml(root / REGISTRY_CONFIG_PATH)
    return {
        "registry": root / config["registry"]["output_path"],
        "manifest": root / config["registry"]["manifest_path"],
    }


def _development_details(task: str, frozen: dict[str, Any]) -> dict[str, Any]:
    metrics = frozen["selected_oof_metrics"]
    if task == "task2":
        primary_metric = "adjusted_r2_raw"
        feature_contract = frozen["selected_feature_contract"]
        proxy = frozen["proxy_disclosure"]
        return {
            "primary_metric": primary_metric,
            "primary_value": metrics[primary_metric],
            "r2": metrics["r2"],
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "feature_contract": feature_contract,
            "proxy_policy": "removed_composite_proxies",
            "proxy_variables": "|".join(proxy["removed_composite_proxies"]),
            "scientific_comparator_name": proxy["competition_contract_model"],
            "scientific_comparator_metric": "adjusted_r2_raw",
            "scientific_comparator_value": proxy["competition_oof_adjusted_r2_raw"],
            "proxy_gap": proxy["proxy_removed_minus_competition_adjusted_r2"],
            "primary_delta_comparability": "descriptive_only_sample_size_sensitive",
        }
    if task == "task1":
        proxy = frozen["proxy_disclosure"]
        return {
            "primary_metric": "r2",
            "primary_value": metrics["r2"],
            "r2": metrics["r2"],
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "feature_contract": "full_non_sleep",
            "proxy_policy": "retained_rules_permitted_composite_proxies",
            "proxy_variables": "|".join(proxy["permitted_composite_proxies"]),
            "scientific_comparator_name": "proxy_removed_spline_ridge",
            "scientific_comparator_metric": "r2",
            "scientific_comparator_value": proxy["proxy_removed_oof_r2"],
            "proxy_gap": proxy["proxy_r2_gap"],
            "primary_delta_comparability": "direct_same_metric",
        }
    comparator = frozen["scientific_comparator"]
    proxy = frozen["proxy_disclosure"]
    return {
        "primary_metric": "r2",
        "primary_value": metrics["r2"],
        "r2": metrics["r2"],
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "feature_contract": frozen["selected_feature_contract"],
        "proxy_policy": "excluded_deterministic_proxies_retained_strong_composite",
        "proxy_variables": "|".join(
            proxy["excluded_deterministic_target_proxies"]
            + [proxy["retained_strong_composite_proxy"]]
        ),
        "scientific_comparator_name": comparator["model"],
        "scientific_comparator_metric": "r2",
        "scientific_comparator_value": comparator["oof_r2"],
        "proxy_gap": proxy["competition_minus_scientific_oof_r2"],
        "primary_delta_comparability": "direct_same_metric",
    }


def _validate_holdout_sources(
    root: Path,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any], dict[str, Any]]:
    sources = config["sources"]
    metrics_path = root / sources["holdout_metrics"]
    holdout_manifest_path = root / sources["holdout_manifest"]
    marker_path = root / sources["holdout_consumed_marker"]
    training_manifest_path = root / sources["evaluation_model_training_manifest"]
    for path in (
        metrics_path,
        holdout_manifest_path,
        marker_path,
        training_manifest_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    metrics = pd.read_csv(metrics_path)
    holdout = load_json_strict(holdout_manifest_path)
    marker = load_json_strict(marker_path)
    training = load_json_strict(training_manifest_path)
    if metrics["task"].tolist() != list(TASKS):
        raise ValueError("最终留出集指标文件必须恰好包含task1、task2、task3")
    if not holdout.get("holdout_consumed") or not marker.get("holdout_consumed"):
        raise ValueError("最终留出集尚未被正式标记为已消费")
    if holdout.get("official_hidden_test_score"):
        raise ValueError("内部留出集不得标记为官方隐藏测试成绩")
    if training.get("status") != "completed":
        raise ValueError("最终评估模型训练清单状态不是completed")
    if training.get("holdout_labels_used") or training.get("holdout_evaluated"):
        raise ValueError("最终评估模型训练阶段不得使用留出集")
    if marker.get("repeat_evaluation_allowed"):
        raise ValueError("消费标记不得允许重复留出集评估")
    if marker.get("run_id") != holdout.get("run_id"):
        raise ValueError("消费标记与留出集运行编号不一致")
    if sha256_file(holdout_manifest_path) != marker.get("evaluation_manifest_sha256"):
        raise ValueError("消费标记中的留出集清单哈希不一致")
    if sha256_file(metrics_path) != holdout.get("metrics_sha256"):
        raise ValueError("留出集指标文件哈希与清单不一致")
    return metrics, holdout, marker, training


def build_registry_frame(root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    config = _load_toml(root / REGISTRY_CONFIG_PATH)
    metrics, holdout, marker, training = _validate_holdout_sources(root, config)
    rows: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {}
    reference_holdout_ids: list[Any] | None = None

    for task in TASKS:
        frozen_path = root / FROZEN_MANIFEST_PATHS[task]
        frozen_config_path = root / FROZEN_CONFIG_PATHS[task]
        frozen = load_json_strict(frozen_path)
        model_manifest_path = root / holdout["tasks"][task]["model_training_manifest"]
        model_manifest = load_json_strict(model_manifest_path)
        prediction_path = root / holdout["tasks"][task]["prediction_path"]
        model_path = root / model_manifest["model_path"]
        for path in (
            frozen_path,
            frozen_config_path,
            model_manifest_path,
            prediction_path,
            model_path,
        ):
            if not path.exists():
                raise FileNotFoundError(path)

        holdout_row = metrics.loc[metrics["task"] == task].iloc[0].to_dict()
        holdout_task = holdout["tasks"][task]
        if frozen["target"] != holdout_row["target"]:
            raise ValueError(f"{task}冻结目标与留出集指标目标不一致")
        if frozen["selected_model"] != holdout_task["selected_model"]:
            raise ValueError(f"{task}冻结模型与留出集模型不一致")
        if model_manifest["selected_model"] != frozen["selected_model"]:
            raise ValueError(f"{task}模型训练清单与冻结模型不一致")
        if model_manifest["model_sha256"] != holdout_task["model_sha256"]:
            raise ValueError(f"{task}模型哈希跨清单不一致")
        if sha256_file(model_path) != model_manifest["model_sha256"]:
            raise ValueError(f"{task}本地模型哈希不一致")
        if sha256_file(prediction_path) != holdout_task["prediction_sha256"]:
            raise ValueError(f"{task}留出集预测文件哈希不一致")
        if training["tasks"][task]["model_sha256"] != model_manifest["model_sha256"]:
            raise ValueError(f"{task}汇总模型训练清单哈希不一致")
        data_hashes = {
            frozen["data_sha256"],
            model_manifest["data_sha256"],
            training["data_sha256"],
            holdout["data_sha256"],
        }
        split_hashes = {
            frozen["split_sha256"],
            model_manifest["split_sha256"],
            training["split_sha256"],
            holdout["split_sha256"],
        }
        if len(data_hashes) != 1 or len(split_hashes) != 1:
            raise ValueError(f"{task}数据或划分哈希跨阶段不一致")
        for name in ("primary_value", "r2", "mae", "rmse"):
            _assert_close(
                holdout_row[name],
                holdout_task["metrics"][name],
                f"{task}留出集{name}",
            )
        prediction = pd.read_csv(prediction_path)
        current_ids = prediction["Person_ID"].tolist()
        if reference_holdout_ids is None:
            reference_holdout_ids = current_ids
        elif current_ids != reference_holdout_ids:
            raise ValueError(f"{task}留出集ID或顺序与其他任务不一致")
        validate_task_output_consistency(
            task,
            prediction,
            holdout_row,
            prediction["Person_ID"],
            int(holdout_row["raw_predictor_count"]),
        )

        development = _development_details(task, frozen)
        row = {
            "task": task,
            "target": frozen["target"],
            "selected_model": frozen["selected_model"],
            "feature_contract": development["feature_contract"],
            "development_evaluation_role": "development_oof",
            "development_primary_metric": development["primary_metric"],
            "development_primary_value": development["primary_value"],
            "development_r2": development["r2"],
            "development_mae": development["mae"],
            "development_rmse": development["rmse"],
            "holdout_evaluation_role": "internal_frozen_holdout",
            "holdout_sample_count": int(holdout_row["sample_count"]),
            "holdout_primary_metric": holdout_row["primary_metric"],
            "holdout_primary_value": holdout_row["primary_value"],
            "holdout_r2": holdout_row["r2"],
            "holdout_adjusted_r2_raw": holdout_row["adjusted_r2_raw"],
            "holdout_mae": holdout_row["mae"],
            "holdout_rmse": holdout_row["rmse"],
            "development_to_holdout_primary_delta": (
                holdout_row["primary_value"] - development["primary_value"]
            ),
            "development_to_holdout_r2_delta": holdout_row["r2"] - development["r2"],
            "development_to_holdout_mae_delta": holdout_row["mae"] - development["mae"],
            "development_to_holdout_rmse_delta": holdout_row["rmse"] - development["rmse"],
            "primary_delta_comparability": development["primary_delta_comparability"],
            "proxy_policy": development["proxy_policy"],
            "proxy_variables": development["proxy_variables"],
            "scientific_comparator_name": development["scientific_comparator_name"],
            "scientific_comparator_metric": development["scientific_comparator_metric"],
            "scientific_comparator_value": development["scientific_comparator_value"],
            "proxy_or_contract_gap": development["proxy_gap"],
            "model_sha256": model_manifest["model_sha256"],
            "prediction_sha256": holdout_task["prediction_sha256"],
            "selection_run_id": frozen["selection_run_id"],
            "confirmation_run_id": frozen["confirmation_run_id"],
            "model_training_run_id": training["run_id"],
            "holdout_run_id": holdout["run_id"],
            "holdout_consumed": True,
            "official_hidden_test_score": False,
            "frozen_candidate_manifest": str(frozen_path.relative_to(root)).replace("\\", "/"),
            "model_training_manifest": str(model_manifest_path.relative_to(root)).replace("\\", "/"),
            "holdout_prediction_path": str(prediction_path.relative_to(root)).replace("\\", "/"),
            "holdout_manifest": config["sources"]["holdout_manifest"],
        }
        rows.append(row)
        for label, path in {
            f"{task}_frozen_manifest": frozen_path,
            f"{task}_frozen_config": frozen_config_path,
            f"{task}_model_manifest": model_manifest_path,
            f"{task}_model": model_path,
            f"{task}_holdout_predictions": prediction_path,
        }.items():
            source_hashes[label] = sha256_file(path)

    registry = pd.DataFrame(rows)
    if registry["task"].tolist() != list(TASKS):
        raise ValueError("最终结果注册表任务顺序不一致")
    if registry["official_hidden_test_score"].any():
        raise ValueError("最终结果注册表不得声称官方隐藏测试成绩")
    source_hashes.update({
        "holdout_metrics": sha256_file(root / config["sources"]["holdout_metrics"]),
        "holdout_manifest": sha256_file(root / config["sources"]["holdout_manifest"]),
        "holdout_consumed_marker": sha256_file(root / config["sources"]["holdout_consumed_marker"]),
        "evaluation_model_training_manifest": sha256_file(
            root / config["sources"]["evaluation_model_training_manifest"]
        ),
    })
    audit = {
        "holdout_run_id": holdout["run_id"],
        "holdout_consumed": marker["holdout_consumed"],
        "task_count": len(registry),
        "all_source_checks_passed": True,
        "official_hidden_test_score": False,
        "source_hashes": source_hashes,
    }
    return registry, audit


def _git_state(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    return {
        "branch": run("branch", "--show-current"),
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(run("status", "--porcelain")),
    }


def build_final_results_registry(root: Path, overwrite: bool = False) -> list[Path]:
    config = _load_toml(root / REGISTRY_CONFIG_PATH)
    paths = registry_output_paths(root)
    if not overwrite and any(path.exists() for path in paths.values()):
        raise FileExistsError("最终结果注册表已存在；确认重建时请使用 --overwrite")
    source_git_state = _git_state(root)
    registry, audit = build_registry_frame(root)
    paths["registry"].parent.mkdir(parents=True, exist_ok=True)
    registry.to_csv(paths["registry"], index=False, encoding="utf-8")

    persisted = pd.read_csv(paths["registry"])
    if persisted["task"].tolist() != list(TASKS) or len(persisted) != 3:
        raise ValueError("写盘后的最终结果注册表任务不完整")
    identity = hashlib.sha256(
        (
            "final_results_registry|"
            + "|".join(audit["source_hashes"].values())
            + f"|{sha256_file(root / REGISTRY_CONFIG_PATH)}"
        ).encode()
    ).hexdigest()[:8]
    now = datetime.now().astimezone()
    run_id = f"{now.strftime('%Y%m%dT%H%M%S%z')}_final_registry_{identity}"
    manifest = {
        "run_id": run_id,
        "status": "completed",
        "experiment": "final_results_machine_registry",
        "paper_artifacts_generated": False,
        "git": source_git_state,
        "registry_path": str(paths["registry"].relative_to(root)).replace("\\", "/"),
        "registry_sha256": sha256_file(paths["registry"]),
        "registry_config_sha256": sha256_file(root / REGISTRY_CONFIG_PATH),
        **audit,
    }
    text = json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False)
    paths["manifest"].parent.mkdir(parents=True, exist_ok=True)
    paths["manifest"].write_text(text, encoding="utf-8")
    history_path = (
        root
        / config["registry"]["history_directory"]
        / f"{run_id}_manifest.json"
    )
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(text, encoding="utf-8")
    return [paths["registry"], paths["manifest"], history_path]
