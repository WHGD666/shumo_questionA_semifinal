from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import catboost
import lightgbm
import numpy as np
import pandas as pd
import sklearn
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

from src.task2_baseline import SEED, _metrics, _sha256, make_preprocessor
from src.task2_elastic_tuning import _git_state
from src.task2_protocol import TARGET, feature_contracts
from src.task2_time_encoding import TIME_FIELDS, engineer_cyclic_times


CONTROL_ADJUSTED_R2 = {
    "competition": 0.6199117677866997,
    "scientific_proxy_removed": 0.6201624354997604,
}
MINIMUM_MATERIAL_IMPROVEMENT = 0.001


def model_factories(seed: int = SEED) -> dict[str, object]:
    return {
        "lightgbm_fixed": LGBMRegressor(
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=30,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
        ),
        "catboost_fixed": CatBoostRegressor(
            iterations=500,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=5.0,
            loss_function="RMSE",
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
        ),
    }


def run_strong_models(root: Path, overwrite: bool = False) -> list[Path]:
    data_candidates = [root / "data" / "raw" / "A题数据集.csv", root / "A题数据集.csv"]
    data_path = next((path for path in data_candidates if path.exists()), None)
    if data_path is None:
        raise FileNotFoundError("未找到复赛数据集")
    split_path = root / "data" / "splits" / "split_assignments.csv"
    config_path = root / "configs" / "task2_strong_models.toml"
    frame = pd.read_csv(data_path)
    assignments = pd.read_csv(split_path)
    merged = frame.merge(assignments, on="Person_ID", validate="one_to_one")
    development = merged.loc[merged["split"] == "development"].copy().reset_index(drop=True)
    if len(development) != 8000 or set(development["cv_fold"].unique()) != {0, 1, 2, 3, 4}:
        raise ValueError("任务二开发集划分不符合冻结协议")

    contracts = feature_contracts(frame.columns.tolist())
    y = pd.to_numeric(development[TARGET]).reset_index(drop=True)
    fold_assignments = development["cv_fold"].astype(int).to_numpy()
    fold_rows: list[dict[str, object]] = []
    oof = pd.DataFrame({"Person_ID": development["Person_ID"], "true_value": y})
    started_at = datetime.now().astimezone()
    started = time.perf_counter()

    for contract_name, source_features in contracts.items():
        X = engineer_cyclic_times(development[source_features].reset_index(drop=True))
        engineered_width = int(X.shape[1])
        expected_width = len(source_features) + sum(field in source_features for field in TIME_FIELDS)
        if engineered_width != expected_width:
            raise ValueError("周期时间特征数量与预期不一致")
        for fold in range(5):
            train_positions = np.flatnonzero(fold_assignments != fold)
            valid_positions = np.flatnonzero(fold_assignments == fold)
            X_train, X_valid = X.iloc[train_positions], X.iloc[valid_positions]
            y_train, y_valid = y.iloc[train_positions], y.iloc[valid_positions]
            print(f"[{contract_name} strong fold {fold + 1}/5] preparing development split...")
            preprocessor = make_preprocessor(X_train)
            train_matrix = preprocessor.fit_transform(X_train)
            valid_matrix = preprocessor.transform(X_valid)
            transformed_width = int(train_matrix.shape[1])
            for model_name, estimator in model_factories(SEED + fold).items():
                fit_started = time.perf_counter()
                estimator.fit(train_matrix, y_train)
                prediction = np.asarray(estimator.predict(valid_matrix)).reshape(-1)
                metric = _metrics(
                    y_valid,
                    prediction,
                    raw_p=engineered_width,
                    transformed_p=transformed_width,
                )
                fold_rows.append({
                    "contract": contract_name,
                    "fold": fold,
                    "model": model_name,
                    "source_feature_count": len(source_features),
                    "engineered_feature_count": engineered_width,
                    "transformed_width": transformed_width,
                    **metric,
                    "fit_seconds": time.perf_counter() - fit_started,
                })
                oof.loc[valid_positions, f"pred_{contract_name}_{model_name}"] = prediction
                print(
                    f"  {model_name}: adjR2={metric['adjusted_r2_raw']:.4f}, "
                    f"R2={metric['r2']:.4f}, MAE={metric['mae']:.4f}"
                )

    fold_metrics = pd.DataFrame(fold_rows)
    summary_rows: list[dict[str, object]] = []
    for contract_name, source_features in contracts.items():
        engineered_width = len(source_features) + sum(field in source_features for field in TIME_FIELDS)
        for model_name in model_factories():
            subset = fold_metrics.loc[
                (fold_metrics["contract"] == contract_name) & (fold_metrics["model"] == model_name)
            ]
            prediction = oof[f"pred_{contract_name}_{model_name}"].to_numpy()
            if np.isnan(prediction).any():
                raise ValueError(f"{contract_name}/{model_name} 的 OOF 预测不完整")
            transformed_width = int(subset["transformed_width"].max())
            pooled = _metrics(
                y,
                prediction,
                raw_p=engineered_width,
                transformed_p=transformed_width,
            )
            summary_rows.append({
                "contract": contract_name,
                "model": model_name,
                "source_feature_count": len(source_features),
                "engineered_feature_count": engineered_width,
                "transformed_width_max": transformed_width,
                "fold_adjusted_r2_raw_mean": float(subset["adjusted_r2_raw"].mean()),
                "fold_adjusted_r2_raw_std": float(subset["adjusted_r2_raw"].std()),
                "oof_adjusted_r2_raw": pooled["adjusted_r2_raw"],
                "oof_adjusted_r2_transformed": pooled["adjusted_r2_transformed"],
                "oof_r2": pooled["r2"],
                "oof_mae": pooled["mae"],
                "oof_rmse": pooled["rmse"],
                "control_adjusted_r2_raw": CONTROL_ADJUSTED_R2[contract_name],
                "delta_vs_tuned_elastic": pooled["adjusted_r2_raw"] - CONTROL_ADJUSTED_R2[contract_name],
                "total_fit_seconds": float(subset["fit_seconds"].sum()),
            })
    summary = pd.DataFrame(summary_rows).sort_values(
        ["contract", "oof_adjusted_r2_raw", "fold_adjusted_r2_raw_std"],
        ascending=[True, False, True],
    )

    output_root = root / "outputs"
    for directory in ("tables", "predictions", "logs", "logs/history"):
        (output_root / directory).mkdir(parents=True, exist_ok=True)
    prefix = "task2_cyclic_strong_models_fixed"
    paths = [
        output_root / "tables" / f"{prefix}_fold_metrics.csv",
        output_root / "tables" / f"{prefix}_summary_metrics.csv",
        output_root / "predictions" / f"{prefix}_oof_predictions.csv",
        output_root / "logs" / f"{prefix}_manifest.json",
    ]
    if not overwrite and any(path.exists() for path in paths):
        raise FileExistsError("任务二固定强模型输出已存在；确认重跑时请使用 --overwrite")

    identity = hashlib.sha256(
        f"task2|cyclic_strong_fixed|{_sha256(data_path)}|{_sha256(split_path)}|{_sha256(config_path)}".encode()
    ).hexdigest()[:8]
    run_id = f"{started_at.strftime('%Y%m%dT%H%M%S%z')}_task2_strong_fixed_{identity}"
    best = summary.iloc[summary["oof_adjusted_r2_raw"].astype(float).argmax()]
    manifest = {
        "run_id": run_id,
        "status": "completed",
        "task": "task2",
        "target": TARGET,
        "experiment": prefix,
        "experiment_role": "controlled_model_family_comparison",
        "command": "python scripts/run_task2_strong_models.py",
        "working_directory": str(root),
        "git": _git_state(root),
        "feature_contracts": {name: len(features) for name, features in contracts.items()},
        "time_encoding": "sin_cos_1440_minutes",
        "models": {name: model.get_params() for name, model in model_factories().items()},
        "control_adjusted_r2": CONTROL_ADJUSTED_R2,
        "minimum_material_improvement": MINIMUM_MATERIAL_IMPROVEMENT,
        "best_contract": str(best["contract"]),
        "best_model": str(best["model"]),
        "best_oof_adjusted_r2_raw": float(best["oof_adjusted_r2_raw"]),
        "best_delta_vs_tuned_elastic": float(best["delta_vs_tuned_elastic"]),
        "material_improvement": bool(best["delta_vs_tuned_elastic"] >= MINIMUM_MATERIAL_IMPROVEMENT),
        "data_sha256": _sha256(data_path),
        "split_sha256": _sha256(split_path),
        "config_sha256": _sha256(config_path),
        "development_rows": 8000,
        "cv_folds": 5,
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
            "catboost": catboost.__version__,
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
