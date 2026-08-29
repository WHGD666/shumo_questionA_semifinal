from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.cross_decomposition import PLSRegression
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.task2_protocol import TARGET, adjusted_r2, feature_contracts


SEED = 2026


def make_preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    numeric = [column for column in frame.columns if pd.api.types.is_numeric_dtype(frame[column])]
    categorical = [column for column in frame.columns if column not in numeric]
    return ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)),
        ]), categorical),
    ], sparse_threshold=0.0)


def model_factories(seed: int = SEED) -> dict[str, object]:
    return {
        "dummy_mean": DummyRegressor(strategy="mean"),
        "ordinary_least_squares": LinearRegression(),
        "ridge_fixed": Ridge(alpha=10.0),
        "elastic_net_fixed": ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=10000, random_state=seed),
        "pls_8": PLSRegression(n_components=8, scale=False, max_iter=1000),
    }


def predictor_count(model_name: str, raw_width: int) -> int:
    return 0 if model_name == "dummy_mean" else raw_width


def _metrics(y_true, prediction, raw_p: int, transformed_p: int) -> dict[str, float]:
    prediction = np.asarray(prediction).reshape(-1)
    r2 = float(r2_score(y_true, prediction))
    n = len(y_true)
    return {
        "r2": r2,
        "adjusted_r2_raw": adjusted_r2(r2, n=n, p=raw_p),
        "adjusted_r2_transformed": adjusted_r2(r2, n=n, p=transformed_p),
        "mae": float(mean_absolute_error(y_true, prediction)),
        "rmse": float(mean_squared_error(y_true, prediction) ** 0.5),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_task2_baselines(root: Path, overwrite: bool = False) -> list[Path]:
    data_candidates = [root / "data" / "raw" / "A题数据集.csv", root / "A题数据集.csv"]
    data_path = next((path for path in data_candidates if path.exists()), None)
    if data_path is None:
        raise FileNotFoundError("未找到复赛数据集")
    split_path = root / "data" / "splits" / "split_assignments.csv"
    config_path = root / "configs" / "task2_baseline_models.toml"
    frame = pd.read_csv(data_path)
    assignments = pd.read_csv(split_path)
    merged = frame.merge(assignments, on="Person_ID", validate="one_to_one")
    development = merged.loc[merged["split"] == "development"].copy().reset_index(drop=True)
    if len(development) != 8000 or set(development["cv_fold"].unique()) != {0, 1, 2, 3, 4}:
        raise ValueError("任务二开发集划分不符合冻结协议")
    contracts = feature_contracts(frame.columns.tolist())
    y = pd.to_numeric(development[TARGET]).reset_index(drop=True)
    fold_assignments = development["cv_fold"].astype(int).reset_index(drop=True)
    fold_rows: list[dict[str, object]] = []
    oof = pd.DataFrame({"Person_ID": development["Person_ID"], "true_value": y})
    widths: dict[str, set[int]] = {contract: set() for contract in contracts}
    started_at = datetime.now().astimezone()
    total_started = time.perf_counter()

    for contract_name, features in contracts.items():
        X = development[features].reset_index(drop=True)
        raw_width = len(features)
        for fold in range(5):
            train_positions = np.flatnonzero(fold_assignments.to_numpy() != fold)
            valid_positions = np.flatnonzero(fold_assignments.to_numpy() == fold)
            X_train, X_valid = X.iloc[train_positions], X.iloc[valid_positions]
            y_train, y_valid = y.iloc[train_positions], y.iloc[valid_positions]
            print(f"[{contract_name} fold {fold + 1}/5] preparing development split...")
            preprocessor = make_preprocessor(X_train)
            transform_started = time.perf_counter()
            train_matrix = preprocessor.fit_transform(X_train)
            valid_matrix = preprocessor.transform(X_valid)
            transform_seconds = time.perf_counter() - transform_started
            transformed_width = int(train_matrix.shape[1])
            widths[contract_name].add(transformed_width)
            for model_name, estimator in model_factories(SEED + fold).items():
                fit_started = time.perf_counter()
                estimator.fit(train_matrix, y_train)
                prediction = np.asarray(estimator.predict(valid_matrix)).reshape(-1)
                elapsed = time.perf_counter() - fit_started
                raw_p = predictor_count(model_name, raw_width)
                transformed_p = predictor_count(model_name, transformed_width)
                metric = _metrics(y_valid, prediction, raw_p=raw_p, transformed_p=transformed_p)
                fold_rows.append({
                    "contract": contract_name,
                    "fold": fold,
                    "model": model_name,
                    "raw_feature_count": raw_width,
                    "transformed_width": transformed_width,
                    **metric,
                    "transform_seconds": transform_seconds,
                    "fit_seconds": elapsed,
                })
                column = f"pred_{contract_name}_{model_name}"
                oof.loc[valid_positions, column] = prediction
                print(
                    f"  {model_name}: adjR2={metric['adjusted_r2_raw']:.4f}, "
                    f"R2={metric['r2']:.4f}, MAE={metric['mae']:.4f}"
                )

    fold_metrics = pd.DataFrame(fold_rows)
    summary_rows = []
    for contract_name, features in contracts.items():
        raw_width = len(features)
        for model_name in model_factories():
            subset = fold_metrics.loc[
                (fold_metrics["contract"] == contract_name) & (fold_metrics["model"] == model_name)
            ]
            prediction = oof[f"pred_{contract_name}_{model_name}"].to_numpy()
            if np.isnan(prediction).any():
                raise ValueError(f"{contract_name}/{model_name} 的 OOF 预测不完整")
            transformed_width = int(subset["transformed_width"].max())
            raw_p = predictor_count(model_name, raw_width)
            transformed_p = predictor_count(model_name, transformed_width)
            pooled = _metrics(y, prediction, raw_p=raw_p, transformed_p=transformed_p)
            summary_rows.append({
                "contract": contract_name,
                "model": model_name,
                "raw_feature_count": raw_width,
                "transformed_width_max": transformed_width,
                "fold_adjusted_r2_raw_mean": float(subset["adjusted_r2_raw"].mean()),
                "fold_adjusted_r2_raw_std": float(subset["adjusted_r2_raw"].std()),
                "fold_r2_mean": float(subset["r2"].mean()),
                "fold_r2_std": float(subset["r2"].std()),
                "oof_adjusted_r2_raw": pooled["adjusted_r2_raw"],
                "oof_adjusted_r2_transformed": pooled["adjusted_r2_transformed"],
                "oof_r2": pooled["r2"],
                "oof_mae": pooled["mae"],
                "oof_rmse": pooled["rmse"],
                "total_fit_seconds": float(subset["fit_seconds"].sum()),
            })
    summary = pd.DataFrame(summary_rows).sort_values(
        ["contract", "oof_adjusted_r2_raw", "fold_adjusted_r2_raw_std"],
        ascending=[True, False, True],
    )

    output_root = root / "outputs"
    for directory in ("tables", "predictions", "logs", "logs/history"):
        (output_root / directory).mkdir(parents=True, exist_ok=True)
    prefix = "task2_multicollinearity_baselines"
    paths = [
        output_root / "tables" / f"{prefix}_fold_metrics.csv",
        output_root / "tables" / f"{prefix}_summary_metrics.csv",
        output_root / "predictions" / f"{prefix}_oof_predictions.csv",
        output_root / "logs" / f"{prefix}_manifest.json",
    ]
    if not overwrite and any(path.exists() for path in paths):
        raise FileExistsError("任务二基线输出已存在；确认重跑时请使用 --overwrite")
    identity = hashlib.sha256(
        f"task2|baseline|{_sha256(data_path)}|{_sha256(split_path)}|{_sha256(config_path)}".encode()
    ).hexdigest()[:8]
    run_id = f"{started_at.strftime('%Y%m%dT%H%M%S%z')}_task2_baseline_{identity}"
    manifest = {
        "run_id": run_id,
        "status": "completed",
        "task": "task2",
        "target": TARGET,
        "experiment": "task2_multicollinearity_baselines",
        "experiment_role": "baseline",
        "feature_contracts": {name: len(features) for name, features in contracts.items()},
        "models": list(model_factories()),
        "primary_metric": "oof_adjusted_r2_raw",
        "adjusted_r2_p_definition": "raw feature count in each declared contract; dummy uses p=0",
        "transformed_width_by_contract": {name: sorted(values) for name, values in widths.items()},
        "data_sha256": _sha256(data_path),
        "split_sha256": _sha256(split_path),
        "config_sha256": _sha256(config_path),
        "development_rows": 8000,
        "cv_folds": 5,
        "seed": SEED,
        "holdout_evaluated": False,
        "duration_seconds": time.perf_counter() - total_started,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "dependencies": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    fold_metrics.to_csv(paths[0], index=False, encoding="utf-8-sig")
    summary.to_csv(paths[1], index=False, encoding="utf-8-sig")
    oof.to_csv(paths[2], index=False, encoding="utf-8-sig")
    manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2)
    paths[3].write_text(manifest_text, encoding="utf-8")
    history = output_root / "logs" / "history" / f"{run_id}_manifest.json"
    history.write_text(manifest_text, encoding="utf-8")
    return [*paths, history]
