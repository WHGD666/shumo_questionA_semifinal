from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import lightgbm
import numpy as np
import pandas as pd
import sklearn
from lightgbm import LGBMRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

from src.task2_baseline import SEED, _metrics, _sha256, make_preprocessor
from src.task2_elastic_tuning import _git_state
from src.task2_protocol import TARGET, feature_contracts
from src.task2_time_encoding import TIME_FIELDS, engineer_cyclic_times


CONTRACT = "scientific_proxy_removed"
INNER_FOLDS = 4
WEIGHTS = (0.25, 0.5, 0.75, 1.0)
CONTROL_ADJUSTED_R2 = 0.6201624354997604
MINIMUM_MATERIAL_IMPROVEMENT = 0.001


def candidate_name(residual_model: str, weight: float) -> str:
    return f"{residual_model}_residual_w{str(weight).replace('.', 'p')}"


def blend_prediction(base_prediction, residual_prediction, weight: float) -> np.ndarray:
    return np.asarray(base_prediction, dtype=float) + weight * np.asarray(
        residual_prediction, dtype=float
    )


def nested_inner_splits(
    row_count: int,
    outer_fold: int,
    inner_folds: int = INNER_FOLDS,
    seed: int = SEED,
):
    splitter = KFold(n_splits=inner_folds, shuffle=True, random_state=seed + outer_fold)
    return list(splitter.split(np.arange(row_count)))


def _base_pipeline(frame: pd.DataFrame, seed: int) -> Pipeline:
    return Pipeline([
        ("preprocess", make_preprocessor(frame)),
        ("model", ElasticNet(
            alpha=0.01,
            l1_ratio=0.9,
            max_iter=20000,
            random_state=seed,
        )),
    ])


def _residual_pipeline(frame: pd.DataFrame, name: str, seed: int) -> Pipeline:
    if name == "ridge":
        estimator = Ridge(alpha=10.0)
    elif name == "lightgbm":
        estimator = LGBMRegressor(
            n_estimators=300,
            learning_rate=0.03,
            num_leaves=15,
            min_child_samples=30,
            reg_alpha=0.5,
            reg_lambda=2.0,
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
        )
    else:
        raise ValueError(f"未知残差模型: {name}")
    return Pipeline([("preprocess", make_preprocessor(frame)), ("model", estimator)])


def run_residual_correction(root: Path, overwrite: bool = False) -> list[Path]:
    data_candidates = [root / "data" / "raw" / "A题数据集.csv", root / "A题数据集.csv"]
    data_path = next((path for path in data_candidates if path.exists()), None)
    if data_path is None:
        raise FileNotFoundError("未找到复赛数据集")
    split_path = root / "data" / "splits" / "split_assignments.csv"
    config_path = root / "configs" / "task2_residual_correction.toml"
    frame = pd.read_csv(data_path)
    assignments = pd.read_csv(split_path)
    merged = frame.merge(assignments, on="Person_ID", validate="one_to_one")
    development = merged.loc[merged["split"] == "development"].copy().reset_index(drop=True)
    if len(development) != 8000 or set(development["cv_fold"].unique()) != {0, 1, 2, 3, 4}:
        raise ValueError("任务二开发集划分不符合冻结协议")

    contracts = feature_contracts(frame.columns.tolist())
    source_features = contracts[CONTRACT]
    X = engineer_cyclic_times(development[source_features].reset_index(drop=True))
    engineered_width = int(X.shape[1])
    expected_width = len(source_features) + sum(field in source_features for field in TIME_FIELDS)
    if engineered_width != expected_width:
        raise ValueError("周期时间特征数量与预期不一致")
    y = pd.to_numeric(development[TARGET]).reset_index(drop=True)
    person_ids = development["Person_ID"].reset_index(drop=True)
    folds = development["cv_fold"].astype(int).to_numpy()

    candidate_names = ["base_elastic"] + [
        candidate_name(model, weight)
        for model in ("ridge", "lightgbm")
        for weight in WEIGHTS
    ]
    predictions = {name: np.full(len(development), np.nan) for name in candidate_names}
    fold_rows: list[dict[str, object]] = []
    started_at = datetime.now().astimezone()
    started = time.perf_counter()

    for outer_fold in range(5):
        train_positions = np.flatnonzero(folds != outer_fold)
        valid_positions = np.flatnonzero(folds == outer_fold)
        X_train, X_valid = X.iloc[train_positions].copy(), X.iloc[valid_positions].copy()
        y_train, y_valid = y.iloc[train_positions], y.iloc[valid_positions]
        print(f"[outer fold {outer_fold + 1}/5] generating leakage-safe residual targets...")

        inner_oof = np.full(len(train_positions), np.nan)
        for inner_train, inner_valid in nested_inner_splits(len(train_positions), outer_fold):
            inner_model = _base_pipeline(X_train.iloc[inner_train], SEED + outer_fold)
            inner_model.fit(X_train.iloc[inner_train], y_train.iloc[inner_train])
            inner_oof[inner_valid] = inner_model.predict(X_train.iloc[inner_valid])
        if np.isnan(inner_oof).any():
            raise ValueError("内层 OOF 主模型预测不完整")

        base_started = time.perf_counter()
        base_model = _base_pipeline(X_train, SEED + outer_fold)
        base_model.fit(X_train, y_train)
        base_valid = np.asarray(base_model.predict(X_valid)).reshape(-1)
        base_seconds = time.perf_counter() - base_started
        predictions["base_elastic"][valid_positions] = base_valid
        base_metric = _metrics(
            y_valid,
            base_valid,
            raw_p=engineered_width,
            transformed_p=engineered_width,
        )
        fold_rows.append({
            "fold": outer_fold,
            "model": "base_elastic",
            "residual_model": "none",
            "weight": 0.0,
            **base_metric,
            "fit_seconds": base_seconds,
        })

        residual_target = y_train.to_numpy() - inner_oof
        residual_train = X_train.copy()
        residual_valid = X_valid.copy()
        residual_train["Base_Prediction"] = inner_oof
        residual_valid["Base_Prediction"] = base_valid
        for residual_name in ("ridge", "lightgbm"):
            fit_started = time.perf_counter()
            residual_model = _residual_pipeline(
                residual_train,
                residual_name,
                SEED + outer_fold,
            )
            residual_model.fit(residual_train, residual_target)
            residual_valid_prediction = np.asarray(
                residual_model.predict(residual_valid)
            ).reshape(-1)
            residual_seconds = time.perf_counter() - fit_started
            for weight in WEIGHTS:
                name = candidate_name(residual_name, weight)
                corrected = blend_prediction(base_valid, residual_valid_prediction, weight)
                predictions[name][valid_positions] = corrected
                metric = _metrics(
                    y_valid,
                    corrected,
                    raw_p=engineered_width,
                    transformed_p=engineered_width,
                )
                fold_rows.append({
                    "fold": outer_fold,
                    "model": name,
                    "residual_model": residual_name,
                    "weight": weight,
                    **metric,
                    "fit_seconds": residual_seconds,
                })
        fold_best = max(
            (row for row in fold_rows if row["fold"] == outer_fold),
            key=lambda row: row["adjusted_r2_raw"],
        )
        print(f"  best={fold_best['model']}: adjR2={fold_best['adjusted_r2_raw']:.4f}")

    fold_metrics = pd.DataFrame(fold_rows)
    summary_rows: list[dict[str, object]] = []
    for name in candidate_names:
        prediction = predictions[name]
        if np.isnan(prediction).any():
            raise ValueError(f"{name} 的外层 OOF 预测不完整")
        subset = fold_metrics.loc[fold_metrics["model"] == name]
        pooled = _metrics(
            y,
            prediction,
            raw_p=engineered_width,
            transformed_p=engineered_width,
        )
        summary_rows.append({
            "contract": CONTRACT,
            "model": name,
            "fold_adjusted_r2_raw_mean": float(subset["adjusted_r2_raw"].mean()),
            "fold_adjusted_r2_raw_std": float(subset["adjusted_r2_raw"].std()),
            "oof_adjusted_r2_raw": pooled["adjusted_r2_raw"],
            "oof_r2": pooled["r2"],
            "oof_mae": pooled["mae"],
            "oof_rmse": pooled["rmse"],
            "delta_vs_control_adjusted_r2": pooled["adjusted_r2_raw"] - CONTROL_ADJUSTED_R2,
            "total_fit_seconds": float(subset["fit_seconds"].sum()),
        })
    summary = pd.DataFrame(summary_rows).sort_values(
        ["oof_adjusted_r2_raw", "fold_adjusted_r2_raw_std"],
        ascending=[False, True],
    )
    oof = pd.DataFrame({"Person_ID": person_ids, "true_value": y})
    for name in candidate_names:
        oof[f"pred_{name}"] = predictions[name]

    output_root = root / "outputs"
    for directory in ("tables", "predictions", "logs", "logs/history"):
        (output_root / directory).mkdir(parents=True, exist_ok=True)
    prefix = "task2_cyclic_nested_residual_correction"
    paths = [
        output_root / "tables" / f"{prefix}_fold_metrics.csv",
        output_root / "tables" / f"{prefix}_summary_metrics.csv",
        output_root / "predictions" / f"{prefix}_oof_predictions.csv",
        output_root / "logs" / f"{prefix}_manifest.json",
    ]
    if not overwrite and any(path.exists() for path in paths):
        raise FileExistsError("任务二残差修正输出已存在；确认重跑时请使用 --overwrite")

    identity = hashlib.sha256(
        f"task2|cyclic_nested_residual|{_sha256(data_path)}|{_sha256(split_path)}|{_sha256(config_path)}".encode()
    ).hexdigest()[:8]
    run_id = f"{started_at.strftime('%Y%m%dT%H%M%S%z')}_task2_nested_residual_{identity}"
    best = summary.iloc[0]
    manifest = {
        "run_id": run_id,
        "status": "completed",
        "task": "task2",
        "target": TARGET,
        "experiment": prefix,
        "experiment_role": "controlled_residual_improvement",
        "method": "outer 5-fold evaluation with inner 4-fold cross-fitted base predictions for residual targets",
        "command": "python scripts/run_task2_residual_correction.py",
        "working_directory": str(root),
        "git": _git_state(root),
        "feature_contract": CONTRACT,
        "source_feature_count": len(source_features),
        "engineered_feature_count": engineered_width,
        "time_encoding": "sin_cos_1440_minutes",
        "outer_folds": 5,
        "inner_folds": INNER_FOLDS,
        "base_model": {
            "family": "elastic_net",
            "alpha": 0.01,
            "l1_ratio": 0.9,
            "max_iter": 20000,
        },
        "residual_models": ["ridge", "lightgbm"],
        "blend_weights": list(WEIGHTS),
        "adjusted_r2_p_definition": "engineered raw feature count; Base_Prediction is derived from the same predictors and is not counted as a new external predictor",
        "control_oof_adjusted_r2": CONTROL_ADJUSTED_R2,
        "best_model": str(best["model"]),
        "best_oof_adjusted_r2": float(best["oof_adjusted_r2_raw"]),
        "best_delta_vs_control_adjusted_r2": float(best["delta_vs_control_adjusted_r2"]),
        "minimum_material_improvement": MINIMUM_MATERIAL_IMPROVEMENT,
        "material_improvement": bool(
            best["delta_vs_control_adjusted_r2"] >= MINIMUM_MATERIAL_IMPROVEMENT
        ),
        "data_sha256": _sha256(data_path),
        "split_sha256": _sha256(split_path),
        "config_sha256": _sha256(config_path),
        "development_rows": 8000,
        "seed": SEED,
        "holdout_evaluated": False,
        "duration_seconds": time.perf_counter() - started,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "dependencies": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "lightgbm": lightgbm.__version__,
        },
        "artifact_paths": [str(path.relative_to(root)).replace("\\", "/") for path in paths[:-1]],
    }
    fold_metrics.to_csv(paths[0], index=False, encoding="utf-8-sig")
    summary.to_csv(paths[1], index=False, encoding="utf-8-sig")
    oof.to_csv(paths[2], index=False, encoding="utf-8-sig")
    manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2)
    paths[3].write_text(manifest_text, encoding="utf-8")
    history = output_root / "logs" / "history" / f"{run_id}_manifest.json"
    history.write_text(manifest_text, encoding="utf-8")
    return [*paths, history]
