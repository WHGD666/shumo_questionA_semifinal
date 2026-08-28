from __future__ import annotations

import hashlib
import json
import tomllib
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_SELECTION_RUN = "20260829T001923+0800_task1_nested_residual_1daa2d59"
EXPECTED_CONFIRMATION_RUN = "20260829T002532+0800_task1_residual_stability_fe7d4957"
EXPECTED_MODEL = "lightgbm_residual_w0p75"


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


def _best_row(path: Path, metric: str) -> pd.Series:
    frame = pd.read_csv(path)
    if frame.empty or metric not in frame.columns:
        raise ValueError(f"结果表缺少 {metric}: {path}")
    return frame.sort_values(metric, ascending=False).iloc[0]


def build_registry(root: Path) -> tuple[pd.DataFrame, dict]:
    tables = root / "outputs" / "tables"
    logs = root / "outputs" / "logs"
    config_path = root / "configs" / "task1_frozen_candidate.toml"
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))

    selection_manifest_path = logs / "task1_non_sleep_nested_residual_correction_manifest.json"
    confirmation_manifest_path = logs / "task1_non_sleep_residual_stability_manifest.json"
    selection_manifest = _read_json(selection_manifest_path)
    confirmation_manifest = _read_json(confirmation_manifest_path)
    if selection_manifest.get("run_id") != EXPECTED_SELECTION_RUN:
        raise ValueError("候选选择运行编号发生变化")
    if confirmation_manifest.get("run_id") != EXPECTED_CONFIRMATION_RUN:
        raise ValueError("稳定性确认运行编号发生变化")
    if confirmation_manifest.get("selection_run_id") != selection_manifest.get("run_id"):
        raise ValueError("稳定性确认未指向候选选择运行")
    if selection_manifest.get("holdout_evaluated") is not False or confirmation_manifest.get("holdout_evaluated") is not False:
        raise ValueError("候选选择或稳定性实验错误地使用了留出集")

    specs = [
        {
            "order": 1,
            "experiment": "linear_tuning",
            "role": "baseline_improvement",
            "contract": "full_non_sleep",
            "table": "task1_non_sleep_linear_tuning_summary_metrics.csv",
            "metric": "oof_r2",
            "decision": "rejected",
            "reason": "Served as the tuned linear control; lower OOF R2 than nonlinear spline models.",
            "manifest": "task1_non_sleep_linear_tuning_manifest.json",
        },
        {
            "order": 2,
            "experiment": "fixed_strong_models",
            "role": "model_family_comparison",
            "contract": "full_non_sleep",
            "table": "task1_non_sleep_strong_models_fixed_summary_metrics.csv",
            "metric": "oof_r2",
            "decision": "rejected",
            "reason": "Tree boosting underperformed the linear control on identical frozen folds.",
            "manifest": "task1_non_sleep_strong_models_fixed_manifest.json",
        },
        {
            "order": 3,
            "experiment": "structured_models",
            "role": "model_family_comparison",
            "contract": "full_non_sleep",
            "table": "task1_non_sleep_structured_models_summary_metrics.csv",
            "metric": "oof_r2",
            "decision": "promoted",
            "reason": "Spline Ridge established a reproducible nonlinear improvement.",
            "manifest": "task1_non_sleep_structured_models_manifest.json",
        },
        {
            "order": 4,
            "experiment": "spline_tuning",
            "role": "bounded_tuning",
            "contract": "full_non_sleep",
            "table": "task1_non_sleep_spline_tuning_summary_metrics.csv",
            "metric": "oof_r2",
            "decision": "promoted",
            "reason": "Bounded grid selected the local spline region without using holdout data.",
            "manifest": "task1_non_sleep_spline_tuning_manifest.json",
        },
        {
            "order": 5,
            "experiment": "spline_refinement",
            "role": "bounded_refinement",
            "contract": "full_non_sleep",
            "table": "task1_non_sleep_spline_refinement_summary_metrics.csv",
            "metric": "oof_r2",
            "decision": "promoted",
            "reason": "Frozen stop rule selected k=4, degree=2, alpha=1.0.",
            "manifest": "task1_non_sleep_spline_refinement_manifest.json",
        },
        {
            "order": 6,
            "experiment": "spline_proxy_validation",
            "role": "scientific_sensitivity",
            "contract": "proxy_removed",
            "table": "task1_spline_proxy_validation_summary_metrics.csv",
            "metric": "oof_r2",
            "filter_column": "contract",
            "filter_value": "proxy_removed",
            "decision": "scientific_comparator",
            "reason": "Quantifies dependence on four disclosed, rules-permitted composite proxies.",
            "manifest": "task1_spline_proxy_validation_manifest.json",
        },
        {
            "order": 7,
            "experiment": "nested_residual_correction",
            "role": "candidate_selection",
            "contract": "full_non_sleep",
            "table": "task1_non_sleep_nested_residual_correction_summary_metrics.csv",
            "metric": "oof_r2",
            "decision": "selected",
            "reason": "Exceeded the predeclared material-improvement threshold with leakage-safe nested residual targets.",
            "manifest": "task1_non_sleep_nested_residual_correction_manifest.json",
        },
    ]
    rows: list[dict[str, object]] = []
    for spec in specs:
        table_path = tables / spec["table"]
        result = pd.read_csv(table_path)
        if "filter_column" in spec:
            result = result.loc[result[spec["filter_column"]] == spec["filter_value"]]
        if result.empty:
            raise ValueError(f"实验结果为空: {spec['experiment']}")
        row = result.sort_values(spec["metric"], ascending=False).iloc[0]
        manifest_path = logs / spec["manifest"]
        manifest = _read_json(manifest_path)
        if manifest.get("holdout_evaluated") is not False:
            raise ValueError(f"实验使用了留出集: {spec['experiment']}")
        rows.append({
            "order": spec["order"],
            "experiment": spec["experiment"],
            "role": spec["role"],
            "feature_contract": spec["contract"],
            "best_model": row.get("model", row.get("contract", "not_applicable")),
            "oof_r2": float(row["oof_r2"]),
            "fold_r2_mean": float(row["fold_r2_mean"]),
            "fold_r2_std": float(row["fold_r2_std"]),
            "oof_mae": float(row["oof_mae"]),
            "oof_rmse": float(row["oof_rmse"]),
            "decision": spec["decision"],
            "decision_reason": spec["reason"],
            "run_id": manifest.get("run_id", "legacy_run_without_id"),
            "summary_path": f"outputs/tables/{spec['table']}",
            "manifest_path": f"outputs/logs/{spec['manifest']}",
            "holdout_evaluated": False,
        })
    registry = pd.DataFrame(rows).sort_values("order")

    selected = registry.loc[registry["decision"] == "selected"].iloc[0]
    frozen = config["candidate"]
    if selected["best_model"] != EXPECTED_MODEL:
        raise ValueError("登记表中的候选模型与冻结模型不一致")
    _assert_close(selected["oof_r2"], frozen["selected_oof_r2"], "冻结 OOF R2")
    _assert_close(selected["oof_mae"], frozen["selected_oof_mae"], "冻结 OOF MAE")
    _assert_close(selected["oof_rmse"], frozen["selected_oof_rmse"], "冻结 OOF RMSE")
    _assert_close(selection_manifest["best_oof_r2"], selected["oof_r2"], "manifest OOF R2")

    stability = pd.read_csv(tables / "task1_non_sleep_residual_stability_aggregate.csv").iloc[0]
    if str(stability["stability_accepted"]).lower() != "true" or confirmation_manifest.get("stability_accepted") is not True:
        raise ValueError("稳定性确认未通过")
    split_assignments = pd.read_csv(root / "data" / "splits" / "split_assignments.csv")
    split_counts = split_assignments["split"].value_counts().to_dict()
    if split_counts != {"development": 8000, "holdout": 2000}:
        raise ValueError(f"冻结划分数量异常: {split_counts}")
    if split_assignments["Person_ID"].duplicated().any():
        raise ValueError("冻结划分包含重复 Person_ID")

    freeze_manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "task": "task1",
        "target": "Sleep_Quality_Score",
        "status": "development_candidate_frozen",
        "candidate_config": "configs/task1_frozen_candidate.toml",
        "candidate_config_sha256": _sha256(config_path),
        "selection_run_id": selection_manifest["run_id"],
        "confirmation_run_id": confirmation_manifest["run_id"],
        "selected_model": EXPECTED_MODEL,
        "selected_oof_metrics": {
            "r2": float(selected["oof_r2"]),
            "mae": float(selected["oof_mae"]),
            "rmse": float(selected["oof_rmse"]),
        },
        "stability_metrics": {
            "mean_oof_r2": float(stability["oof_r2_mean"]),
            "std_oof_r2": float(stability["oof_r2_std"]),
            "minimum_delta_r2": float(stability["delta_r2_min"]),
            "accepted": True,
        },
        "proxy_disclosure": config["proxy_disclosure"],
        "data_sha256": selection_manifest["data_sha256"],
        "split_sha256": selection_manifest["split_sha256"],
        "development_rows": 8000,
        "sealed_holdout_rows": 2000,
        "holdout_evaluated": False,
        "final_model_trained": False,
    }
    return registry, freeze_manifest


def freeze_task1_candidate(root: Path, overwrite: bool = False) -> list[Path]:
    registry, manifest = build_registry(root)
    table_path = root / "outputs" / "tables" / "task1_experiment_registry.csv"
    manifest_path = root / "outputs" / "logs" / "task1_frozen_candidate_manifest.json"
    if not overwrite and (table_path.exists() or manifest_path.exists()):
        raise FileExistsError("任务一冻结登记已存在；确认重建时请使用 --overwrite")
    table_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    registry.to_csv(table_path, index=False, encoding="utf-8-sig")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return [table_path, manifest_path]
