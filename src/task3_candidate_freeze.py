from __future__ import annotations

import hashlib
import json
import tomllib
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_SELECTION_RUN = "20260829T235336+0800_task3_additive_c92b6dee"
EXPECTED_CONFIRMATION_RUN = "20260829T235738+0800_task3_additive_refinement_b1e98c2f"
EXPECTED_MODEL = "additive_elastic_k4_d2_a0p003"
EXPECTED_CONTRACT = "competition_proxy_inclusive"
EXPECTED_SCIENTIFIC_MODEL = "additive_ridge_k4_d2_a0p3"
EXPECTED_SCIENTIFIC_CONTRACT = "scientific_proxy_removed"


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
            "experiment": "regression_baseline_competition",
            "role": "baseline",
            "contract": EXPECTED_CONTRACT,
            "table": "task3_health_regression_baselines_summary_metrics.csv",
            "metric": "oof_r2",
            "filters": {"contract": EXPECTED_CONTRACT},
            "manifest": "task3_health_regression_baselines_manifest.json",
            "decision": "baseline_control",
            "reason": "Elastic Net established the strongest competition-contract baseline.",
        },
        {
            "order": 2,
            "experiment": "regression_baseline_scientific",
            "role": "proxy_sensitivity_baseline",
            "contract": EXPECTED_SCIENTIFIC_CONTRACT,
            "table": "task3_health_regression_baselines_summary_metrics.csv",
            "metric": "oof_r2",
            "filters": {"contract": EXPECTED_SCIENTIFIC_CONTRACT},
            "manifest": "task3_health_regression_baselines_manifest.json",
            "decision": "scientific_baseline",
            "reason": "The proxy-removed baseline quantified generalizable signal without Healthy_Aging_Score.",
        },
        {
            "order": 3,
            "experiment": "linear_tuning_competition",
            "role": "bounded_hyperparameter_tuning",
            "contract": EXPECTED_CONTRACT,
            "table": "task3_health_linear_tuning_summary_metrics.csv",
            "metric": "oof_r2",
            "filters": {"contract": EXPECTED_CONTRACT},
            "manifest": "task3_health_linear_tuning_manifest.json",
            "decision": "rejected_below_threshold",
            "reason": "The gain over fixed Elastic Net was below the predeclared material threshold.",
        },
        {
            "order": 4,
            "experiment": "linear_tuning_scientific",
            "role": "bounded_hyperparameter_tuning",
            "contract": EXPECTED_SCIENTIFIC_CONTRACT,
            "table": "task3_health_linear_tuning_summary_metrics.csv",
            "metric": "oof_r2",
            "filters": {"contract": EXPECTED_SCIENTIFIC_CONTRACT},
            "manifest": "task3_health_linear_tuning_manifest.json",
            "decision": "rejected_below_threshold",
            "reason": "The proxy-removed linear gain was positive but below the frozen threshold.",
        },
        {
            "order": 5,
            "experiment": "targeted_additive_competition",
            "role": "candidate_selection",
            "contract": EXPECTED_CONTRACT,
            "table": "task3_health_targeted_additive_models_summary_metrics.csv",
            "metric": "oof_r2",
            "filters": {"contract": EXPECTED_CONTRACT},
            "manifest": "task3_health_targeted_additive_models_manifest.json",
            "decision": "selected",
            "reason": "Targeted splines produced a material gain and retained an interpretable additive structure.",
        },
        {
            "order": 6,
            "experiment": "targeted_additive_scientific",
            "role": "scientific_proxy_comparator",
            "contract": EXPECTED_SCIENTIFIC_CONTRACT,
            "table": "task3_health_targeted_additive_models_summary_metrics.csv",
            "metric": "oof_r2",
            "filters": {"contract": EXPECTED_SCIENTIFIC_CONTRACT},
            "manifest": "task3_health_targeted_additive_models_manifest.json",
            "decision": "scientific_comparator",
            "reason": "The same controlled additive strategy quantified performance without Healthy_Aging_Score.",
        },
        {
            "order": 7,
            "experiment": "additive_boundary_competition",
            "role": "boundary_refinement",
            "contract": EXPECTED_CONTRACT,
            "table": "task3_health_additive_refinement_summary_metrics.csv",
            "metric": "oof_r2",
            "filters": {"contract": EXPECTED_CONTRACT},
            "manifest": "task3_health_additive_refinement_manifest.json",
            "decision": "confirmed_stop",
            "reason": "Six knots gave a sub-threshold gain, so the simpler four-knot model was retained.",
        },
        {
            "order": 8,
            "experiment": "additive_boundary_scientific",
            "role": "boundary_refinement",
            "contract": EXPECTED_SCIENTIFIC_CONTRACT,
            "table": "task3_health_additive_refinement_summary_metrics.csv",
            "metric": "oof_r2",
            "filters": {"contract": EXPECTED_SCIENTIFIC_CONTRACT},
            "manifest": "task3_health_additive_refinement_manifest.json",
            "decision": "confirmed_stop",
            "reason": "The four-knot scientific comparator remained optimal in the expanded boundary grid.",
        },
    ]


def _compact_oof(
    root: Path,
    source_name: str,
    prediction_column: str,
) -> pd.DataFrame:
    source = pd.read_csv(root / "outputs" / "predictions" / source_name)
    if prediction_column not in source.columns:
        raise ValueError(f"任务三 OOF 源文件缺少冻结预测列: {prediction_column}")
    compact = source[["Person_ID", "true_value", prediction_column]].rename(
        columns={prediction_column: "predicted_value"}
    )
    compact["residual"] = compact["true_value"] - compact["predicted_value"]
    if len(compact) != 8000 or compact["Person_ID"].duplicated().any() or compact.isna().any().any():
        raise ValueError("任务三冻结 OOF 预测不完整或包含重复 ID")
    return compact


def build_registry(root: Path) -> tuple[pd.DataFrame, dict, pd.DataFrame, pd.DataFrame]:
    tables = root / "outputs" / "tables"
    logs = root / "outputs" / "logs"
    config_path = root / "configs" / "task3_frozen_candidate.toml"
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    selection_manifest = _read_json(logs / "task3_health_targeted_additive_models_manifest.json")
    confirmation_manifest = _read_json(logs / "task3_health_additive_refinement_manifest.json")
    protocol_manifest = _read_json(logs / "task3_protocol_audit_manifest.json")
    if selection_manifest.get("run_id") != EXPECTED_SELECTION_RUN:
        raise ValueError("任务三候选选择运行编号发生变化")
    if confirmation_manifest.get("run_id") != EXPECTED_CONFIRMATION_RUN:
        raise ValueError("任务三边界确认运行编号发生变化")
    if any(
        manifest.get("holdout_evaluated") is not False
        for manifest in (selection_manifest, confirmation_manifest, protocol_manifest)
    ):
        raise ValueError("任务三协议、选择或确认实验错误地使用了留出集")
    selected_spec = selection_manifest["best_by_contract"][EXPECTED_CONTRACT]
    scientific_spec = selection_manifest["best_by_contract"][EXPECTED_SCIENTIFIC_CONTRACT]
    if selected_spec["model"] != EXPECTED_MODEL:
        raise ValueError("任务三选择运行没有产生预期竞赛候选")
    if scientific_spec["model"] != EXPECTED_SCIENTIFIC_MODEL:
        raise ValueError("任务三选择运行没有产生预期科学对照")
    boundary = confirmation_manifest["best_by_contract"][EXPECTED_CONTRACT]
    if boundary.get("material_improvement") is not False:
        raise ValueError("任务三边界确认尚未触发停止规则")

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
            "fold_r2_mean": float(result["fold_r2_mean"]),
            "fold_r2_std": float(result["fold_r2_std"]),
            "oof_r2": float(result["oof_r2"]),
            "oof_mae": float(result["oof_mae"]),
            "oof_rmse": float(result["oof_rmse"]),
            "decision": spec["decision"],
            "decision_reason": spec["reason"],
            "run_id": manifest["run_id"],
            "summary_path": f"outputs/tables/{spec['table']}",
            "manifest_path": f"outputs/logs/{spec['manifest']}",
            "holdout_evaluated": False,
        })
    registry = pd.DataFrame(rows).sort_values("order")
    selected = registry.loc[registry["decision"] == "selected"]
    scientific = registry.loc[registry["decision"] == "scientific_comparator"]
    if len(selected) != 1 or len(scientific) != 1:
        raise ValueError("任务三登记表必须包含一个竞赛候选和一个科学对照")
    selected = selected.iloc[0]
    scientific = scientific.iloc[0]
    candidate_config = config["competition_candidate"]
    scientific_config = config["scientific_comparator"]
    if selected["best_model"] != EXPECTED_MODEL or selected["feature_contract"] != EXPECTED_CONTRACT:
        raise ValueError("任务三登记表中的竞赛候选不一致")
    if scientific["best_model"] != EXPECTED_SCIENTIFIC_MODEL:
        raise ValueError("任务三登记表中的科学对照不一致")
    for metric in ("oof_r2", "oof_mae", "oof_rmse", "fold_r2_mean", "fold_r2_std"):
        _assert_close(selected[metric], candidate_config[metric], f"竞赛候选 {metric}")
        _assert_close(scientific[metric], scientific_config[metric], f"科学对照 {metric}")

    source_name = "task3_health_targeted_additive_models_oof_predictions.csv"
    selected_oof = _compact_oof(
        root,
        source_name,
        f"pred_{EXPECTED_CONTRACT}_{EXPECTED_MODEL}",
    )
    scientific_oof = _compact_oof(
        root,
        source_name,
        f"pred_{EXPECTED_SCIENTIFIC_CONTRACT}_{EXPECTED_SCIENTIFIC_MODEL}",
    )
    split_assignments = pd.read_csv(root / "data" / "splits" / "split_assignments.csv")
    if split_assignments["split"].value_counts().to_dict() != {"development": 8000, "holdout": 2000}:
        raise ValueError("任务三冻结划分数量异常")
    if split_assignments["Person_ID"].duplicated().any():
        raise ValueError("任务三冻结划分包含重复 Person_ID")

    freeze_manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "task": "task3",
        "target": "Health_Score",
        "status": "development_candidate_frozen",
        "candidate_config": "configs/task3_frozen_candidate.toml",
        "candidate_config_sha256": _sha256(config_path),
        "selection_run_id": selection_manifest["run_id"],
        "confirmation_run_id": confirmation_manifest["run_id"],
        "selected_model": EXPECTED_MODEL,
        "selected_feature_contract": EXPECTED_CONTRACT,
        "selected_oof_metrics": {
            "r2": float(selected["oof_r2"]),
            "mae": float(selected["oof_mae"]),
            "rmse": float(selected["oof_rmse"]),
            "fold_r2_mean": float(selected["fold_r2_mean"]),
            "fold_r2_std": float(selected["fold_r2_std"]),
        },
        "scientific_comparator": {
            "model": EXPECTED_SCIENTIFIC_MODEL,
            "feature_contract": EXPECTED_SCIENTIFIC_CONTRACT,
            "oof_r2": float(scientific["oof_r2"]),
            "oof_mae": float(scientific["oof_mae"]),
            "oof_rmse": float(scientific["oof_rmse"]),
        },
        "proxy_disclosure": {
            **config["proxy_disclosure"],
            "deterministic_proxy_checks": protocol_manifest["deterministic_proxy_checks"],
        },
        "stop_confirmation": {
            **config["stop_confirmation"],
            "run_id": confirmation_manifest["run_id"],
        },
        "data_sha256": common_data_sha,
        "split_sha256": common_split_sha,
        "development_rows": 8000,
        "sealed_holdout_rows": 2000,
        "holdout_evaluated": False,
        "final_model_trained": False,
        "selected_oof_path": "outputs/predictions/task3_frozen_candidate_oof_predictions.csv",
        "scientific_oof_path": "outputs/predictions/task3_frozen_scientific_comparator_oof_predictions.csv",
    }
    return registry, freeze_manifest, selected_oof, scientific_oof


def freeze_task3_candidate(root: Path, overwrite: bool = False) -> list[Path]:
    registry, manifest, selected_oof, scientific_oof = build_registry(root)
    table_path = root / "outputs" / "tables" / "task3_experiment_registry.csv"
    manifest_path = root / "outputs" / "logs" / "task3_frozen_candidate_manifest.json"
    selected_path = root / "outputs" / "predictions" / "task3_frozen_candidate_oof_predictions.csv"
    scientific_path = root / "outputs" / "predictions" / "task3_frozen_scientific_comparator_oof_predictions.csv"
    paths = [table_path, manifest_path, selected_path, scientific_path]
    if not overwrite and any(path.exists() for path in paths):
        raise FileExistsError("任务三冻结登记已存在；确认重建时请使用 --overwrite")
    table_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    selected_path.parent.mkdir(parents=True, exist_ok=True)
    registry.to_csv(table_path, index=False, encoding="utf-8-sig")
    selected_oof.to_csv(selected_path, index=False, encoding="utf-8-sig")
    scientific_oof.to_csv(scientific_path, index=False, encoding="utf-8-sig")
    manifest["selected_oof_sha256"] = _sha256(selected_path)
    manifest["scientific_oof_sha256"] = _sha256(scientific_path)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return paths
