from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.cross_decomposition import PLSRegression
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.task2_baseline import make_preprocessor
from src.task2_time_encoding import TIME_FIELDS, engineer_cyclic_times
from src.task3_protocol import TARGET, feature_contracts


SEED = 2026


def model_factories(seed: int = SEED) -> dict[str, object]:
    return {
        "dummy_mean": DummyRegressor(strategy="mean"),
        "ordinary_least_squares": LinearRegression(),
        "ridge_fixed": Ridge(alpha=10.0),
        "elastic_net_fixed": ElasticNet(
            alpha=0.001,
            l1_ratio=0.5,
            max_iter=20000,
            random_state=seed,
        ),
        "pls_16": PLSRegression(n_components=16, scale=False, max_iter=2000),
        "extra_trees_fixed": ExtraTreesRegressor(
            n_estimators=400,
            min_samples_leaf=2,
            max_features=0.8,
            n_jobs=-1,
            random_state=seed,
        ),
    }


def _metrics(y_true, prediction) -> dict[str, float]:
    prediction = np.asarray(prediction).reshape(-1)
    return {
        "r2": float(r2_score(y_true, prediction)),
        "mae": float(mean_absolute_error(y_true, prediction)),
        "rmse": float(mean_squared_error(y_true, prediction) ** 0.5),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def run_task3_baselines(root: Path, overwrite: bool = False) -> list[Path]:
    data_candidates = [root / "data" / "raw" / "A题数据集.csv", root / "A题数据集.csv"]
    data_path = next((path for path in data_candidates if path.exists()), None)
    if data_path is None:
        raise FileNotFoundError("未找到复赛数据集")
    split_path = root / "data" / "splits" / "split_assignments.csv"
    config_path = root / "configs" / "task3_baseline_models.toml"
    frame = pd.read_csv(data_path)
    assignments = pd.read_csv(split_path)
    merged = frame.merge(assignments, on="Person_ID", validate="one_to_one")
    development = merged.loc[merged["split"] == "development"].copy().reset_index(drop=True)
    if len(development) != 8000 or set(development["cv_fold"].unique()) != {0, 1, 2, 3, 4}:
        raise ValueError("任务三开发集划分不符合冻结协议")

    contracts = feature_contracts(frame.columns.tolist())
    y = pd.to_numeric(development[TARGET]).reset_index(drop=True)
    folds = development["cv_fold"].astype(int).to_numpy()
    fold_rows: list[dict[str, object]] = []
    oof = pd.DataFrame({"Person_ID": development["Person_ID"], "true_value": y})
    started_at = datetime.now().astimezone()
    started = time.perf_counter()

    for contract, source_features in contracts.items():
        X = engineer_cyclic_times(development[source_features].reset_index(drop=True))
        engineered_width = int(X.shape[1])
        expected_width = len(source_features) + sum(field in source_features for field in TIME_FIELDS)
        if engineered_width != expected_width:
            raise ValueError("任务三周期时间特征数量与预期不一致")
        for model_name in model_factories():
            oof[f"pred_{contract}_{model_name}"] = np.nan
        for fold in range(5):
            train_positions = np.flatnonzero(folds != fold)
            valid_positions = np.flatnonzero(folds == fold)
            X_train, X_valid = X.iloc[train_positions], X.iloc[valid_positions]
            y_train, y_valid = y.iloc[train_positions], y.iloc[valid_positions]
            print(f"[{contract} fold {fold + 1}/5] preparing development split...")
            preprocessor = make_preprocessor(X_train)
            transform_started = time.perf_counter()
            train_matrix = preprocessor.fit_transform(X_train)
            valid_matrix = preprocessor.transform(X_valid)
            transform_seconds = time.perf_counter() - transform_started
            transformed_width = int(train_matrix.shape[1])
            if transformed_width < 16:
                raise ValueError("任务三折内变换宽度不足以拟合 PLS-16")
            for model_name, estimator in model_factories(SEED + fold).items():
                fit_started = time.perf_counter()
                estimator.fit(train_matrix, y_train)
                prediction = np.asarray(estimator.predict(valid_matrix)).reshape(-1)
                metric = _metrics(y_valid, prediction)
                fold_rows.append({
                    "contract": contract,
                    "fold": fold,
                    "model": model_name,
                    "source_feature_count": len(source_features),
                    "engineered_feature_count": engineered_width,
                    "transformed_width": transformed_width,
                    **metric,
                    "transform_seconds": transform_seconds,
                    "fit_seconds": time.perf_counter() - fit_started,
                })
                oof.loc[valid_positions, f"pred_{contract}_{model_name}"] = prediction
                print(
                    f"  {model_name}: R2={metric['r2']:.4f}, "
                    f"MAE={metric['mae']:.4f}, RMSE={metric['rmse']:.4f}"
                )

    fold_metrics = pd.DataFrame(fold_rows)
    summary_rows: list[dict[str, object]] = []
    for contract, source_features in contracts.items():
        engineered_width = len(source_features) + sum(field in source_features for field in TIME_FIELDS)
        for model_name in model_factories():
            subset = fold_metrics.loc[
                (fold_metrics["contract"] == contract) & (fold_metrics["model"] == model_name)
            ]
            prediction = oof[f"pred_{contract}_{model_name}"].to_numpy()
            if np.isnan(prediction).any():
                raise ValueError(f"{contract}/{model_name} 的 OOF 预测不完整")
            pooled = _metrics(y, prediction)
            summary_rows.append({
                "contract": contract,
                "model": model_name,
                "source_feature_count": len(source_features),
                "engineered_feature_count": engineered_width,
                "transformed_width_max": int(subset["transformed_width"].max()),
                "fold_r2_mean": float(subset["r2"].mean()),
                "fold_r2_std": float(subset["r2"].std()),
                "oof_r2": pooled["r2"],
                "oof_mae": pooled["mae"],
                "oof_rmse": pooled["rmse"],
                "total_fit_seconds": float(subset["fit_seconds"].sum()),
            })
    summary = pd.DataFrame(summary_rows).sort_values(
        ["contract", "oof_r2", "fold_r2_std"], ascending=[True, False, True]
    )

    output_root = root / "outputs"
    for directory in ("tables", "predictions", "logs", "logs/history"):
        (output_root / directory).mkdir(parents=True, exist_ok=True)
    prefix = "task3_health_regression_baselines"
    paths = [
        output_root / "tables" / f"{prefix}_fold_metrics.csv",
        output_root / "tables" / f"{prefix}_summary_metrics.csv",
        output_root / "predictions" / f"{prefix}_oof_predictions.csv",
        output_root / "logs" / f"{prefix}_manifest.json",
    ]
    if not overwrite and any(path.exists() for path in paths):
        raise FileExistsError("任务三基线输出已存在；确认重跑时请使用 --overwrite")

    identity = hashlib.sha256(
        f"task3|baseline|{_sha256(data_path)}|{_sha256(split_path)}|{_sha256(config_path)}".encode()
    ).hexdigest()[:8]
    run_id = f"{started_at.strftime('%Y%m%dT%H%M%S%z')}_task3_baseline_{identity}"
    winners = summary.groupby("contract", sort=False).head(1)
    manifest = {
        "run_id": run_id,
        "status": "completed",
        "task": "task3",
        "target": TARGET,
        "experiment": prefix,
        "experiment_role": "baseline",
        "command": "python scripts/run_task3_baselines.py",
        "working_directory": str(root),
        "git": _git_state(root),
        "feature_contracts": {name: len(features) for name, features in contracts.items()},
        "time_encoding": "sin_cos_1440_minutes",
        "models": list(model_factories()),
        "primary_metric": "oof_r2",
        "best_by_contract": {
            row["contract"]: {
                "model": row["model"],
                "oof_r2": float(row["oof_r2"]),
                "oof_mae": float(row["oof_mae"]),
                "oof_rmse": float(row["oof_rmse"]),
            }
            for row in winners.to_dict(orient="records")
        },
        "data_sha256": _sha256(data_path),
        "split_sha256": _sha256(split_path),
        "config_sha256": _sha256(config_path),
        "development_rows": 8000,
        "sealed_holdout_rows": 2000,
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
