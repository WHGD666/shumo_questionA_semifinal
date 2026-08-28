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
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from src.task1_regression_baseline import FORBIDDEN, TARGET, make_preprocessor


SEED = 2026
RIDGE_ALPHAS = (0.1, 1.0, 3.0, 10.0, 30.0, 100.0)
ELASTIC_ALPHAS = (0.001, 0.005, 0.01)
ELASTIC_L1_RATIOS = (0.1, 0.5, 0.9)


def _number_token(value: float) -> str:
    return str(value).replace(".", "p")


def model_factories(seed: int = SEED) -> dict[str, object]:
    models: dict[str, object] = {
        f"ridge_alpha_{_number_token(alpha)}": Ridge(alpha=alpha)
        for alpha in RIDGE_ALPHAS
    }
    for alpha in ELASTIC_ALPHAS:
        for ratio in ELASTIC_L1_RATIOS:
            name = f"elastic_alpha_{_number_token(alpha)}_l1_{_number_token(ratio)}"
            models[name] = ElasticNet(
                alpha=alpha,
                l1_ratio=ratio,
                max_iter=10000,
                random_state=seed,
            )
    return models


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metrics(y_true: pd.Series | np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, prediction)),
        "mae": float(mean_absolute_error(y_true, prediction)),
        "rmse": float(mean_squared_error(y_true, prediction) ** 0.5),
    }


def run_linear_tuning(root: Path, overwrite: bool = False) -> list[Path]:
    data_candidates = [root / "data" / "raw" / "A题数据集.csv", root / "A题数据集.csv"]
    data_path = next((path for path in data_candidates if path.exists()), None)
    if data_path is None:
        raise FileNotFoundError("未找到复赛数据集")
    split_path = root / "data" / "splits" / "split_assignments.csv"
    config_path = root / "configs" / "task1_linear_tuning.toml"
    assignments = pd.read_csv(split_path)
    frame = pd.read_csv(data_path)
    merged = frame.merge(assignments, on="Person_ID", validate="one_to_one")
    development = merged.loc[merged["split"] == "development"].copy()
    if len(development) != 8000 or (development["cv_fold"] < 0).any():
        raise ValueError("开发集划分不符合冻结协议")
    if (merged["split"] == "holdout").sum() != 2000:
        raise ValueError("留出集划分不符合冻结协议")

    feature_names = [column for column in frame.columns if column not in FORBIDDEN]
    X = development[feature_names]
    y = pd.to_numeric(development[TARGET])
    factories = model_factories()
    fold_rows: list[dict[str, object]] = []
    oof = pd.DataFrame({"Person_ID": development["Person_ID"].to_numpy(), "true_value": y.to_numpy()})
    started_at = datetime.now().astimezone()
    total_started = time.perf_counter()

    for fold in range(5):
        train_mask = development["cv_fold"] != fold
        valid_mask = development["cv_fold"] == fold
        X_train, X_valid = X.loc[train_mask], X.loc[valid_mask]
        y_train, y_valid = y.loc[train_mask], y.loc[valid_mask]
        valid_ids = development.loc[valid_mask, "Person_ID"]
        for name, estimator in factories.items():
            fit_started = time.perf_counter()
            pipeline = Pipeline([
                ("preprocess", make_preprocessor(X_train)),
                ("model", estimator),
            ])
            pipeline.fit(X_train, y_train)
            prediction = pipeline.predict(X_valid)
            metric = _metrics(y_valid, prediction)
            fold_rows.append({
                "fold": fold,
                "model": name,
                **metric,
                "fit_seconds": time.perf_counter() - fit_started,
            })
            id_to_prediction = dict(zip(valid_ids, prediction))
            oof.loc[oof["Person_ID"].isin(id_to_prediction), f"pred_{name}"] = (
                oof.loc[oof["Person_ID"].isin(id_to_prediction), "Person_ID"].map(id_to_prediction)
            )

    fold_metrics = pd.DataFrame(fold_rows)
    summary_rows = []
    for name in factories:
        subset = fold_metrics.loc[fold_metrics["model"] == name]
        prediction = oof[f"pred_{name}"].to_numpy()
        if np.isnan(prediction).any():
            raise ValueError(f"{name} 的 OOF 预测不完整")
        pooled = _metrics(oof["true_value"], prediction)
        summary_rows.append({
            "model": name,
            "fold_r2_mean": float(subset["r2"].mean()),
            "fold_r2_std": float(subset["r2"].std()),
            "oof_r2": pooled["r2"],
            "oof_mae": pooled["mae"],
            "oof_rmse": pooled["rmse"],
            "total_fit_seconds": float(subset["fit_seconds"].sum()),
        })
    summary = pd.DataFrame(summary_rows).sort_values(
        ["oof_r2", "fold_r2_std"], ascending=[False, True]
    )

    output_root = root / "outputs"
    for directory in ("tables", "predictions", "logs", "logs/history"):
        (output_root / directory).mkdir(parents=True, exist_ok=True)
    prefix = "task1_non_sleep_linear_tuning"
    current_paths = [
        output_root / "tables" / f"{prefix}_fold_metrics.csv",
        output_root / "tables" / f"{prefix}_summary_metrics.csv",
        output_root / "predictions" / f"{prefix}_oof_predictions.csv",
        output_root / "logs" / f"{prefix}_manifest.json",
    ]
    if not overwrite and any(path.exists() for path in current_paths):
        raise FileExistsError("线性调参输出已存在；确认重跑时请使用 --overwrite")

    config_hash = _sha256(config_path)
    split_hash = _sha256(split_path)
    data_hash = _sha256(data_path)
    identity = hashlib.sha256(
        f"task1|linear_tuning|{config_hash}|{split_hash}|{data_hash}".encode("utf-8")
    ).hexdigest()[:8]
    run_id = f"{started_at.strftime('%Y%m%dT%H%M%S%z')}_task1_linear_tuning_{identity}"
    finished_at = datetime.now().astimezone()
    manifest = {
        "run_id": run_id,
        "status": "completed",
        "task": "task1",
        "target": TARGET,
        "experiment": "non_sleep_linear_tuning",
        "experiment_role": "controlled_improvement",
        "start_time": started_at.isoformat(),
        "end_time": finished_at.isoformat(),
        "duration_seconds": time.perf_counter() - total_started,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "dependencies": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "data_sha256": data_hash,
        "split_sha256": split_hash,
        "config_sha256": config_hash,
        "development_rows": len(development),
        "feature_count": len(feature_names),
        "allowed_features": feature_names,
        "forbidden_features": sorted(FORBIDDEN),
        "models": {name: model.get_params() for name, model in factories.items()},
        "selection_metric": "oof_r2",
        "best_model": summary.iloc[0]["model"],
        "best_oof_r2": float(summary.iloc[0]["oof_r2"]),
        "holdout_evaluated": False,
    }
    history_manifest = output_root / "logs" / "history" / f"{run_id}_manifest.json"
    fold_metrics.to_csv(current_paths[0], index=False, encoding="utf-8-sig")
    summary.to_csv(current_paths[1], index=False, encoding="utf-8-sig")
    oof.to_csv(current_paths[2], index=False, encoding="utf-8-sig")
    manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2)
    current_paths[3].write_text(manifest_text, encoding="utf-8")
    history_manifest.write_text(manifest_text, encoding="utf-8")
    return [*current_paths, history_manifest]
