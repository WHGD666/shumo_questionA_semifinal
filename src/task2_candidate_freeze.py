from __future__ import annotations

import hashlib
import json
import tomllib
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_SELECTION_RUN = "20260829T173352+0800_task2_elastic_tuning_59b765b7"
EXPECTED_CONFIRMATION_RUN = "20260829T190021+0800_task2_learning_curve_3d139646"
EXPECTED_MODEL = "elastic_a0p01_l1_0p9"
EXPECTED_CONTRACT = "scientific_proxy_removed"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_close(actual: float, expected: float, label: str, tolerance: float = 1e-12) -> None:
    if not np.isclose(float(actual), float(expected), rtol=0.0, atol=tolerance):
        raise ValueError(f"{label} 不一致: {actual} != {expected}")


def _select_result(tables: Path, spec: dict[str, object]) -> pd.Series:
    frame = pd.read_csv(tables / str(spec["table"]))
    for column, value in spec.get("filters", {}).items():
        frame = frame.loc[frame[column] == value]
    if frame.empty:
        raise ValueError(f"实验结果为空: {spec['experiment']}")
    return frame.sort_values(str(spec["metric"]), ascending=False).iloc[0]


def _specifications() -> list[dict[str, object]]:
    return [
        {
            "order": 1,
            "experiment": "multicollinearity_baselines",
            "role": "baseline",
            "contract": EXPECTED_CONTRACT,
            "table": "task2_multicollinearity_baselines_summary_metrics.csv",
            "metric": "oof_adjusted_r2_raw",
            "filters": {"contract": EXPECTED_CONTRACT},
            "manifest": "task2_multicollinearity_baselines_manifest.json",
            "decision": "baseline_control",
            "reason": "PLS established the strongest raw-time baseline under the proxy-removed contract.",
        },
        {
            "order": 2,
            "experiment": "cyclic_time_encoding",
            "role": "controlled_feature_engineering",
            "contract": EXPECTED_CONTRACT,
            "table": "task2_cyclic_time_encoding_summary_metrics.csv",
            "metric": "oof_adjusted_r2_raw",
            "filters": {"contract": EXPECTED_CONTRACT},
            "manifest": "task2_cyclic_time_encoding_manifest.json",
            "decision": "promoted",
            "reason": "Cyclic time encoding improved the adjusted R2 control on identical folds.",
        },
        {
            "order": 3,
            "experiment": "elastic_tuning",
            "role": "candidate_selection",
            "contract": EXPECTED_CONTRACT,
            "table": "task2_cyclic_elastic_tuning_summary_metrics.csv",
            "metric": "oof_adjusted_r2_raw",
            "filters": {"contract": EXPECTED_CONTRACT},
            "manifest": "task2_cyclic_elastic_tuning_manifest.json",
            "decision": "selected",
            "reason": "The bounded grid selected alpha=0.01 and l1_ratio=0.9 without holdout feedback.",
        },
        {
            "order": 4,
            "experiment": "competition_contract_comparator",
            "role": "proxy_sensitivity",
            "contract": "competition",
            "table": "task2_cyclic_elastic_tuning_summary_metrics.csv",
            "metric": "oof_adjusted_r2_raw",
            "filters": {"contract": "competition"},
            "manifest": "task2_cyclic_elastic_tuning_manifest.json",
            "decision": "scientific_comparator",
            "reason": "The four disclosed composite proxies did not improve adjusted or raw OOF R2.",
        },
        {
            "order": 5,
            "experiment": "fixed_strong_models",
            "role": "model_family_comparison",
            "contract": EXPECTED_CONTRACT,
            "table": "task2_cyclic_strong_models_fixed_summary_metrics.csv",
            "metric": "oof_adjusted_r2_raw",
            "filters": {"contract": EXPECTED_CONTRACT},
            "manifest": "task2_cyclic_strong_models_fixed_manifest.json",
            "decision": "rejected",
            "reason": "CatBoost and LightGBM both underperformed the tuned Elastic Net.",
        },
        {
            "order": 6,
            "experiment": "nested_residual_correction",
            "role": "controlled_residual_improvement",
            "contract": EXPECTED_CONTRACT,
            "table": "task2_cyclic_nested_residual_correction_summary_metrics.csv",
            "metric": "oof_adjusted_r2_raw",
            "filters": {"contract": EXPECTED_CONTRACT},
            "manifest": "task2_cyclic_nested_residual_correction_manifest.json",
            "decision": "rejected",
            "reason": "Leakage-safe residual correction did not exceed the unchanged base Elastic Net.",
        },
        {
            "order": 7,
            "experiment": "targeted_additive_models",
            "role": "controlled_structured_nonlinearity",
            "contract": EXPECTED_CONTRACT,
            "table": "task2_cyclic_targeted_additive_models_summary_metrics.csv",
            "metric": "oof_adjusted_r2_raw",
            "filters": {"contract": EXPECTED_CONTRACT},
            "manifest": "task2_cyclic_targeted_additive_models_manifest.json",
            "decision": "rejected",
            "reason": "Targeted spline nonlinearities did not provide a material improvement.",
        },
        {
            "order": 8,
            "experiment": "latent_factor_models",
            "role": "controlled_multicollinearity_structure",
            "contract": EXPECTED_CONTRACT,
            "table": "task2_cyclic_latent_factor_models_summary_metrics.csv",
            "metric": "oof_adjusted_r2_raw",
            "filters": {"contract": EXPECTED_CONTRACT},
            "manifest": "task2_cyclic_latent_factor_models_manifest.json",
            "decision": "rejected",
            "reason": "PLS and PCR compression did not outperform sparse regularization.",
        },
        {
            "order": 9,
            "experiment": "kernel_approximation_models",
            "role": "controlled_smooth_interactions",
            "contract": EXPECTED_CONTRACT,
            "table": "task2_cyclic_kernel_approximation_models_summary_metrics.csv",
            "metric": "oof_adjusted_r2_raw",
            "filters": {"contract": EXPECTED_CONTRACT},
            "manifest": "task2_cyclic_kernel_approximation_models_manifest.json",
            "decision": "rejected",
            "reason": "RBF and Nystroem approximations lost substantial adjusted R2.",
        },
    ]


def build_registry(root: Path) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    tables = root / "outputs" / "tables"
    logs = root / "outputs" / "logs"
    config_path = root / "configs" / "task2_frozen_candidate.toml"
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    selection_manifest = _read_json(logs / "task2_cyclic_elastic_tuning_manifest.json")
    confirmation_manifest = _read_json(logs / "task2_cyclic_elastic_learning_curve_manifest.json")
    if selection_manifest.get("run_id") != EXPECTED_SELECTION_RUN:
        raise ValueError("任务二候选选择运行编号发生变化")
    if confirmation_manifest.get("run_id") != EXPECTED_CONFIRMATION_RUN:
        raise ValueError("任务二学习曲线确认运行编号发生变化")
    if selection_manifest.get("holdout_evaluated") is not False or confirmation_manifest.get("holdout_evaluated") is not False:
        raise ValueError("任务二候选选择或确认实验错误地使用了留出集")
    if confirmation_manifest.get("diagnosis", {}).get("practical_plateau_supported") is not True:
        raise ValueError("任务二学习曲线尚未支持停止规则")
    frozen = config["candidate"]
    frozen_candidate = confirmation_manifest.get("frozen_candidate", {})
    if frozen_candidate.get("model") != EXPECTED_MODEL:
        raise ValueError("学习曲线没有确认冻结候选")
    _assert_close(frozen_candidate["alpha"], config["model"]["alpha"], "学习曲线 alpha")
    _assert_close(frozen_candidate["l1_ratio"], config["model"]["l1_ratio"], "学习曲线 l1_ratio")

    common_data_sha = selection_manifest["data_sha256"]
    common_split_sha = selection_manifest["split_sha256"]
    rows: list[dict[str, object]] = []
    for spec in _specifications():
        result = _select_result(tables, spec)
        manifest = _read_json(logs / str(spec["manifest"]))
        if manifest.get("holdout_evaluated") is not False:
            raise ValueError(f"实验使用了留出集: {spec['experiment']}")
        if manifest.get("data_sha256") != common_data_sha or manifest.get("split_sha256") != common_split_sha:
            raise ValueError(f"实验数据或划分指纹不一致: {spec['experiment']}")
        rows.append({
            "order": spec["order"],
            "experiment": spec["experiment"],
            "role": spec["role"],
            "feature_contract": spec["contract"],
            "best_model": result["model"],
            "oof_adjusted_r2_raw": float(result["oof_adjusted_r2_raw"]),
            "fold_adjusted_r2_raw_mean": float(result["fold_adjusted_r2_raw_mean"]),
            "fold_adjusted_r2_raw_std": float(result["fold_adjusted_r2_raw_std"]),
            "oof_r2": float(result["oof_r2"]),
            "oof_mae": float(result["oof_mae"]),
            "oof_rmse": float(result["oof_rmse"]),
            "decision": spec["decision"],
            "decision_reason": spec["reason"],
            "run_id": manifest.get("run_id", "missing_run_id"),
            "summary_path": f"outputs/tables/{spec['table']}",
            "manifest_path": f"outputs/logs/{spec['manifest']}",
            "holdout_evaluated": False,
        })

    curve = pd.read_csv(tables / "task2_cyclic_elastic_learning_curve_aggregate.csv")
    full_curve = curve.loc[np.isclose(curve["training_fraction"], 1.0)].iloc[0]
    rows.append({
        "order": 10,
        "experiment": "learning_curve",
        "role": "diagnostic_stop_confirmation",
        "feature_contract": EXPECTED_CONTRACT,
        "best_model": EXPECTED_MODEL,
        "oof_adjusted_r2_raw": float(full_curve["mean_oof_adjusted_r2_raw"]),
        "fold_adjusted_r2_raw_mean": float(frozen["fold_adjusted_r2_raw_mean"]),
        "fold_adjusted_r2_raw_std": float(frozen["fold_adjusted_r2_raw_std"]),
        "oof_r2": float(full_curve["mean_oof_r2"]),
        "oof_mae": float(full_curve["mean_oof_mae"]),
        "oof_rmse": float(full_curve["mean_oof_rmse"]),
        "decision": "confirmed_plateau",
        "decision_reason": "The 80%-to-100% gain and full-data generalization gap satisfied the frozen stop rule.",
        "run_id": confirmation_manifest["run_id"],
        "summary_path": "outputs/tables/task2_cyclic_elastic_learning_curve_aggregate.csv",
        "manifest_path": "outputs/logs/task2_cyclic_elastic_learning_curve_manifest.json",
        "holdout_evaluated": False,
    })
    registry = pd.DataFrame(rows).sort_values("order")
    selected = registry.loc[registry["decision"] == "selected"].iloc[0]
    if selected["best_model"] != EXPECTED_MODEL or selected["feature_contract"] != EXPECTED_CONTRACT:
        raise ValueError("任务二登记表中的候选模型或特征契约不一致")
    _assert_close(selected["oof_adjusted_r2_raw"], frozen["selected_oof_adjusted_r2_raw"], "冻结调整 R2")
    _assert_close(selected["oof_r2"], frozen["selected_oof_r2"], "冻结 R2")
    _assert_close(selected["oof_mae"], frozen["selected_oof_mae"], "冻结 MAE")
    _assert_close(selected["oof_rmse"], frozen["selected_oof_rmse"], "冻结 RMSE")

    source_oof = pd.read_csv(root / "outputs" / "predictions" / "task2_cyclic_elastic_tuning_oof_predictions.csv")
    prediction_column = f"pred_{EXPECTED_CONTRACT}_{EXPECTED_MODEL}"
    if prediction_column not in source_oof.columns:
        raise ValueError("任务二候选 OOF 源文件缺少冻结预测列")
    selected_oof = source_oof[["Person_ID", "true_value", prediction_column]].rename(
        columns={prediction_column: "predicted_value"}
    )
    selected_oof["residual"] = selected_oof["true_value"] - selected_oof["predicted_value"]
    if len(selected_oof) != 8000 or selected_oof["Person_ID"].duplicated().any() or selected_oof.isna().any().any():
        raise ValueError("任务二冻结 OOF 预测不完整或包含重复 ID")

    split_assignments = pd.read_csv(root / "data" / "splits" / "split_assignments.csv")
    if split_assignments["split"].value_counts().to_dict() != {"development": 8000, "holdout": 2000}:
        raise ValueError("任务二冻结划分数量异常")
    if split_assignments["Person_ID"].duplicated().any():
        raise ValueError("任务二冻结划分包含重复 Person_ID")

    competition = registry.loc[registry["decision"] == "scientific_comparator"].iloc[0]
    freeze_manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "task": "task2",
        "target": "Productivity_Score",
        "status": "development_candidate_frozen",
        "candidate_config": "configs/task2_frozen_candidate.toml",
        "candidate_config_sha256": _sha256(config_path),
        "selection_run_id": selection_manifest["run_id"],
        "confirmation_run_id": confirmation_manifest["run_id"],
        "selected_model": EXPECTED_MODEL,
        "selected_feature_contract": EXPECTED_CONTRACT,
        "selected_oof_metrics": {
            "adjusted_r2_raw": float(selected["oof_adjusted_r2_raw"]),
            "r2": float(selected["oof_r2"]),
            "mae": float(selected["oof_mae"]),
            "rmse": float(selected["oof_rmse"]),
        },
        "proxy_disclosure": {
            **config["proxy_disclosure"],
            "competition_contract_model": str(competition["best_model"]),
        },
        "learning_curve_confirmation": {
            **config["learning_curve_confirmation"],
            "run_id": confirmation_manifest["run_id"],
        },
        "data_sha256": common_data_sha,
        "split_sha256": common_split_sha,
        "development_rows": 8000,
        "sealed_holdout_rows": 2000,
        "holdout_evaluated": False,
        "final_model_trained": False,
        "selected_oof_path": "outputs/predictions/task2_frozen_candidate_oof_predictions.csv",
    }
    return registry, freeze_manifest, selected_oof


def freeze_task2_candidate(root: Path, overwrite: bool = False) -> list[Path]:
    registry, manifest, selected_oof = build_registry(root)
    table_path = root / "outputs" / "tables" / "task2_experiment_registry.csv"
    manifest_path = root / "outputs" / "logs" / "task2_frozen_candidate_manifest.json"
    oof_path = root / "outputs" / "predictions" / "task2_frozen_candidate_oof_predictions.csv"
    paths = [table_path, manifest_path, oof_path]
    if not overwrite and any(path.exists() for path in paths):
        raise FileExistsError("任务二冻结登记已存在；确认重建时请使用 --overwrite")
    table_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    oof_path.parent.mkdir(parents=True, exist_ok=True)
    registry.to_csv(table_path, index=False, encoding="utf-8-sig")
    selected_oof.to_csv(oof_path, index=False, encoding="utf-8-sig")
    manifest["selected_oof_sha256"] = _sha256(oof_path)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return paths
