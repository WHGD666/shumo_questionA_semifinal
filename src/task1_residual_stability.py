from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.task1_regression_baseline import FORBIDDEN, TARGET
from src.task1_residual_correction import (
    CONTROL_OOF_R2,
    _base_pipeline,
    _metrics,
    _residual_pipeline,
    _sha256,
    blend_prediction,
    nested_inner_splits,
)


INNER_SEEDS = (2026, 2037, 2048, 2059, 2070)
RESIDUAL_WEIGHT = 0.75
SELECTION_RUN_ID = "20260829T001923+0800_task1_nested_residual_1daa2d59"


def stability_acceptance(seed_deltas: list[float]) -> bool:
    return bool(seed_deltas and min(seed_deltas) > 0.0 and float(np.mean(seed_deltas)) >= 0.001)


def run_stability(root: Path, overwrite: bool = False) -> list[Path]:
    data_candidates = [root / "data" / "raw" / "A题数据集.csv", root / "A题数据集.csv"]
    data_path = next((path for path in data_candidates if path.exists()), None)
    if data_path is None:
        raise FileNotFoundError("未找到复赛数据集")
    split_path = root / "data" / "splits" / "split_assignments.csv"
    config_path = root / "configs" / "task1_residual_stability.toml"
    frame = pd.read_csv(data_path)
    assignments = pd.read_csv(split_path)
    merged = frame.merge(assignments, on="Person_ID", validate="one_to_one")
    development = merged.loc[merged["split"] == "development"].copy().reset_index(drop=True)
    if len(development) != 8000 or set(development["cv_fold"].unique()) != {0, 1, 2, 3, 4}:
        raise ValueError("开发集划分不符合冻结协议")
    features = [column for column in frame.columns if column not in FORBIDDEN]
    X = development[features]
    y = pd.to_numeric(development[TARGET]).reset_index(drop=True)
    folds = development["cv_fold"].astype(int).to_numpy()
    fold_rows: list[dict[str, object]] = []
    seed_rows: list[dict[str, object]] = []
    oof = pd.DataFrame({"Person_ID": development["Person_ID"], "true_value": y})
    started_at = datetime.now().astimezone()
    total_started = time.perf_counter()

    for inner_seed in INNER_SEEDS:
        corrected_oof = np.full(len(development), np.nan)
        base_oof = np.full(len(development), np.nan)
        print(f"[inner seed {inner_seed}] confirming frozen candidate...")
        for outer_fold in range(5):
            train_positions = np.flatnonzero(folds != outer_fold)
            valid_positions = np.flatnonzero(folds == outer_fold)
            X_train, X_valid = X.iloc[train_positions].copy(), X.iloc[valid_positions].copy()
            y_train, y_valid = y.iloc[train_positions], y.iloc[valid_positions]
            inner_oof = np.full(len(train_positions), np.nan)
            for inner_train, inner_valid in nested_inner_splits(
                len(train_positions), outer_fold, seed=inner_seed
            ):
                inner_model = _base_pipeline(X_train.iloc[inner_train])
                inner_model.fit(X_train.iloc[inner_train], y_train.iloc[inner_train])
                inner_oof[inner_valid] = inner_model.predict(X_train.iloc[inner_valid])
            if np.isnan(inner_oof).any():
                raise ValueError("内层 OOF 主模型预测不完整")

            base_model = _base_pipeline(X_train)
            base_model.fit(X_train, y_train)
            base_valid = base_model.predict(X_valid)
            residual_train = X_train.copy()
            residual_valid = X_valid.copy()
            residual_train["Base_Prediction"] = inner_oof
            residual_valid["Base_Prediction"] = base_valid
            residual_model = _residual_pipeline(residual_train, "lightgbm", inner_seed + outer_fold)
            residual_model.fit(residual_train, y_train.to_numpy() - inner_oof)
            corrected = blend_prediction(base_valid, residual_model.predict(residual_valid), RESIDUAL_WEIGHT)
            base_oof[valid_positions] = base_valid
            corrected_oof[valid_positions] = corrected
            base_metric = _metrics(y_valid, base_valid)
            corrected_metric = _metrics(y_valid, corrected)
            fold_rows.append({
                "inner_seed": inner_seed,
                "outer_fold": outer_fold,
                "base_r2": base_metric["r2"],
                "corrected_r2": corrected_metric["r2"],
                "delta_r2": corrected_metric["r2"] - base_metric["r2"],
                "corrected_mae": corrected_metric["mae"],
                "corrected_rmse": corrected_metric["rmse"],
            })
        pooled = _metrics(y, corrected_oof)
        base_pooled = _metrics(y, base_oof)
        seed_rows.append({
            "inner_seed": inner_seed,
            "oof_r2": pooled["r2"],
            "oof_mae": pooled["mae"],
            "oof_rmse": pooled["rmse"],
            "base_oof_r2": base_pooled["r2"],
            "delta_vs_base_r2": pooled["r2"] - base_pooled["r2"],
        })
        oof[f"pred_seed_{inner_seed}"] = corrected_oof
        print(f"  OOF R2={pooled['r2']:.6f}, delta={pooled['r2'] - base_pooled['r2']:+.6f}")

    fold_metrics = pd.DataFrame(fold_rows)
    seed_summary = pd.DataFrame(seed_rows)
    deltas = seed_summary["delta_vs_base_r2"].tolist()
    aggregate = pd.DataFrame([{
        "candidate": "spline_ridge_plus_lightgbm_residual_w0p75",
        "seed_count": len(INNER_SEEDS),
        "oof_r2_mean": float(seed_summary["oof_r2"].mean()),
        "oof_r2_std": float(seed_summary["oof_r2"].std()),
        "oof_r2_min": float(seed_summary["oof_r2"].min()),
        "oof_r2_max": float(seed_summary["oof_r2"].max()),
        "delta_r2_mean": float(np.mean(deltas)),
        "delta_r2_min": float(np.min(deltas)),
        "all_seed_deltas_positive": bool(min(deltas) > 0),
        "stability_accepted": stability_acceptance(deltas),
    }])

    output_root = root / "outputs"
    for directory in ("tables", "predictions", "logs", "logs/history"):
        (output_root / directory).mkdir(parents=True, exist_ok=True)
    prefix = "task1_non_sleep_residual_stability"
    paths = [
        output_root / "tables" / f"{prefix}_fold_metrics.csv",
        output_root / "tables" / f"{prefix}_seed_summary.csv",
        output_root / "tables" / f"{prefix}_aggregate.csv",
        output_root / "predictions" / f"{prefix}_oof_predictions.csv",
        output_root / "logs" / f"{prefix}_manifest.json",
    ]
    if not overwrite and any(path.exists() for path in paths):
        raise FileExistsError("残差稳定性输出已存在；确认重跑时请使用 --overwrite")
    identity = hashlib.sha256(
        f"task1|residual_stability|{_sha256(data_path)}|{_sha256(split_path)}|{_sha256(config_path)}".encode()
    ).hexdigest()[:8]
    run_id = f"{started_at.strftime('%Y%m%dT%H%M%S%z')}_task1_residual_stability_{identity}"
    manifest = {
        "run_id": run_id,
        "status": "completed",
        "task": "task1",
        "target": TARGET,
        "experiment": "non_sleep_residual_stability",
        "experiment_role": "candidate_confirmation",
        "selection_run_id": SELECTION_RUN_ID,
        "frozen_candidate": "spline_ridge_plus_lightgbm_residual_w0p75",
        "inner_seeds": list(INNER_SEEDS),
        "outer_folds": 5,
        "inner_folds": 4,
        "feature_count": len(features),
        "control_oof_r2": CONTROL_OOF_R2,
        "mean_oof_r2": float(seed_summary["oof_r2"].mean()),
        "mean_delta_vs_base_r2": float(np.mean(deltas)),
        "minimum_delta_vs_base_r2": float(np.min(deltas)),
        "stability_accepted": stability_acceptance(deltas),
        "duration_seconds": time.perf_counter() - total_started,
        "data_sha256": _sha256(data_path),
        "split_sha256": _sha256(split_path),
        "config_sha256": _sha256(config_path),
        "holdout_evaluated": False,
    }
    fold_metrics.to_csv(paths[0], index=False, encoding="utf-8-sig")
    seed_summary.to_csv(paths[1], index=False, encoding="utf-8-sig")
    aggregate.to_csv(paths[2], index=False, encoding="utf-8-sig")
    oof.to_csv(paths[3], index=False, encoding="utf-8-sig")
    manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2)
    paths[4].write_text(manifest_text, encoding="utf-8")
    history = output_root / "logs" / "history" / f"{run_id}_manifest.json"
    history.write_text(manifest_text, encoding="utf-8")
    return [*paths, history]
