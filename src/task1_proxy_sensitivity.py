from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from src.task1_regression_baseline import FORBIDDEN, TARGET, make_preprocessor


SEED = 2026
PROXY_FIELDS = {"Health_Score", "Fitness_Level", "Healthy_Aging_Score", "Wellness_Category"}


def feature_contracts(all_columns: list[str]) -> dict[str, list[str]]:
    return {
        "full_non_sleep": [column for column in all_columns if column not in FORBIDDEN],
        "proxy_removed": [column for column in all_columns if column not in FORBIDDEN | PROXY_FIELDS],
    }


def model_factories() -> dict[str, object]:
    return {
        "ridge_alpha_3": Ridge(alpha=3.0),
        "elastic_alpha_0p001_l1_0p9": ElasticNet(
            alpha=0.001, l1_ratio=0.9, max_iter=10000, random_state=SEED
        ),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metrics(y_true, prediction) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, prediction)),
        "mae": float(mean_absolute_error(y_true, prediction)),
        "rmse": float(mean_squared_error(y_true, prediction) ** 0.5),
    }


def run_proxy_sensitivity(root: Path, overwrite: bool = False) -> list[Path]:
    candidates = [root / "data" / "raw" / "A题数据集.csv", root / "A题数据集.csv"]
    data_path = next((path for path in candidates if path.exists()), None)
    if data_path is None:
        raise FileNotFoundError("未找到复赛数据集")
    split_path = root / "data" / "splits" / "split_assignments.csv"
    config_path = root / "configs" / "task1_proxy_sensitivity.toml"
    frame = pd.read_csv(data_path)
    assignments = pd.read_csv(split_path)
    merged = frame.merge(assignments, on="Person_ID", validate="one_to_one")
    development = merged.loc[merged["split"] == "development"].copy()
    contracts = feature_contracts(list(frame.columns))
    models = model_factories()
    y = pd.to_numeric(development[TARGET])
    fold_rows, coefficient_rows = [], []
    oof = pd.DataFrame({"Person_ID": development["Person_ID"].to_numpy(), "true_value": y.to_numpy()})
    started_at = datetime.now().astimezone()
    total_started = time.perf_counter()

    for contract_name, features in contracts.items():
        X = development[features]
        for fold in range(5):
            train_mask = development["cv_fold"] != fold
            valid_mask = development["cv_fold"] == fold
            print(f"[{contract_name} fold {fold + 1}/5] preparing split...")
            for model_name, estimator in models.items():
                started = time.perf_counter()
                pipeline = Pipeline([
                    ("preprocess", make_preprocessor(X.loc[train_mask])),
                    ("model", estimator),
                ])
                pipeline.fit(X.loc[train_mask], y.loc[train_mask])
                prediction = pipeline.predict(X.loc[valid_mask])
                metric = _metrics(y.loc[valid_mask], prediction)
                print(f"  {model_name}: R2={metric['r2']:.4f}, MAE={metric['mae']:.4f}")
                fold_rows.append({
                    "contract": contract_name,
                    "fold": fold,
                    "model": model_name,
                    "feature_count": len(features),
                    **metric,
                    "fit_seconds": time.perf_counter() - started,
                })
                valid_ids = development.loc[valid_mask, "Person_ID"]
                mapping = dict(zip(valid_ids, prediction))
                column = f"pred_{contract_name}_{model_name}"
                mask = oof["Person_ID"].isin(mapping)
                oof.loc[mask, column] = oof.loc[mask, "Person_ID"].map(mapping)
                names = pipeline.named_steps["preprocess"].get_feature_names_out()
                coefficients = np.asarray(pipeline.named_steps["model"].coef_).ravel()
                for feature, coefficient in zip(names, coefficients):
                    coefficient_rows.append({
                        "contract": contract_name,
                        "fold": fold,
                        "model": model_name,
                        "transformed_feature": feature,
                        "coefficient": float(coefficient),
                        "abs_coefficient": abs(float(coefficient)),
                    })

    fold_metrics = pd.DataFrame(fold_rows)
    summary_rows = []
    for contract_name in contracts:
        for model_name in models:
            subset = fold_metrics.loc[
                (fold_metrics["contract"] == contract_name) & (fold_metrics["model"] == model_name)
            ]
            prediction = oof[f"pred_{contract_name}_{model_name}"].to_numpy()
            if np.isnan(prediction).any():
                raise ValueError("OOF 预测不完整")
            pooled = _metrics(oof["true_value"], prediction)
            summary_rows.append({
                "contract": contract_name,
                "model": model_name,
                "feature_count": len(contracts[contract_name]),
                "fold_r2_mean": float(subset["r2"].mean()),
                "fold_r2_std": float(subset["r2"].std()),
                "oof_r2": pooled["r2"],
                "oof_mae": pooled["mae"],
                "oof_rmse": pooled["rmse"],
            })
    summary = pd.DataFrame(summary_rows).sort_values("oof_r2", ascending=False)
    coefficient_fold = pd.DataFrame(coefficient_rows)
    coefficient_summary = coefficient_fold.groupby(
        ["contract", "model", "transformed_feature"], as_index=False
    ).agg(
        coefficient_mean=("coefficient", "mean"),
        coefficient_std=("coefficient", "std"),
        abs_coefficient_mean=("abs_coefficient", "mean"),
    ).sort_values(["contract", "model", "abs_coefficient_mean"], ascending=[True, True, False])

    output_root = root / "outputs"
    for directory in ("tables", "predictions", "logs", "logs/history"):
        (output_root / directory).mkdir(parents=True, exist_ok=True)
    prefix = "task1_non_sleep_proxy_sensitivity"
    paths = [
        output_root / "tables" / f"{prefix}_fold_metrics.csv",
        output_root / "tables" / f"{prefix}_summary_metrics.csv",
        output_root / "tables" / f"{prefix}_coefficient_fold.csv",
        output_root / "tables" / f"{prefix}_coefficient_summary.csv",
        output_root / "predictions" / f"{prefix}_oof_predictions.csv",
        output_root / "logs" / f"{prefix}_manifest.json",
    ]
    if not overwrite and any(path.exists() for path in paths):
        raise FileExistsError("代理敏感性输出已存在；确认重跑时请使用 --overwrite")
    identity = hashlib.sha256(
        f"task1|proxy|{_sha256(data_path)}|{_sha256(split_path)}|{_sha256(config_path)}".encode()
    ).hexdigest()[:8]
    run_id = f"{started_at.strftime('%Y%m%dT%H%M%S%z')}_task1_proxy_sensitivity_{identity}"
    manifest = {
        "run_id": run_id,
        "status": "completed",
        "task": "task1",
        "target": TARGET,
        "experiment": "non_sleep_proxy_sensitivity",
        "experiment_role": "scientific_comparison",
        "contracts": contracts,
        "proxy_fields": sorted(PROXY_FIELDS),
        "models": {name: model.get_params() for name, model in models.items()},
        "duration_seconds": time.perf_counter() - total_started,
        "best_contract": summary.iloc[0]["contract"],
        "best_model": summary.iloc[0]["model"],
        "best_oof_r2": float(summary.iloc[0]["oof_r2"]),
        "holdout_evaluated": False,
    }
    fold_metrics.to_csv(paths[0], index=False, encoding="utf-8-sig")
    summary.to_csv(paths[1], index=False, encoding="utf-8-sig")
    coefficient_fold.to_csv(paths[2], index=False, encoding="utf-8-sig")
    coefficient_summary.to_csv(paths[3], index=False, encoding="utf-8-sig")
    oof.to_csv(paths[4], index=False, encoding="utf-8-sig")
    manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2)
    paths[5].write_text(manifest_text, encoding="utf-8")
    history = output_root / "logs" / "history" / f"{run_id}_manifest.json"
    history.write_text(manifest_text, encoding="utf-8")
    return [*paths, history]
