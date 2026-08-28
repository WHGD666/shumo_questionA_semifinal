from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.dummy import DummyRegressor


TARGET = "Sleep_Quality_Score"
FORBIDDEN = {"Person_ID", TARGET, "Sleep_Duration_Hours", "Sleep_Time", "Wake_Up_Time", "Number_of_Night_Awakenings", "Weekend_Sleep_Difference_Hours", "Nap_Frequency_Per_Week", "Screen_Time_Before_Bed_Hours", "Sleep_Disorder_Risk"}


def make_preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    numeric = [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]
    categorical = [c for c in frame.columns if c not in numeric]
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])


def model_factories(seed: int = 2026) -> dict[str, object]:
    return {
        "dummy_mean": DummyRegressor(strategy="mean"),
        "ridge": Ridge(alpha=10.0),
        "elastic_net": ElasticNet(alpha=0.01, l1_ratio=0.2, max_iter=5000, random_state=seed),
        "extra_trees": ExtraTreesRegressor(n_estimators=300, random_state=seed, n_jobs=-1, min_samples_leaf=2),
    }


def run_task1(root: Path, overwrite: bool = False) -> list[Path]:
    data_path = root / "data" / "raw" / "A题数据集.csv"
    if not data_path.exists():
        data_path = root / "A题数据集.csv"
    assignments = pd.read_csv(root / "data" / "splits" / "split_assignments.csv")
    frame = pd.read_csv(data_path)
    merged = frame.merge(assignments, on="Person_ID", validate="one_to_one")
    features = [c for c in frame.columns if c not in FORBIDDEN]
    X = merged[features]
    y = pd.to_numeric(merged[TARGET])
    dev = merged[merged.split == "development"]
    rows, importance = [], []
    oof = pd.DataFrame({"Person_ID": dev.Person_ID, "true_value": y[dev.index].to_numpy()})
    for fold in range(5):
        valid_mask = (dev.cv_fold == fold).to_numpy()
        train_mask = ~valid_mask
        train_idx, valid_idx = dev.index.to_numpy()[train_mask], dev.index.to_numpy()[valid_mask]
        for name, estimator in model_factories().items():
            started = time.perf_counter()
            pipe = Pipeline([("preprocess", make_preprocessor(X.loc[train_idx])), ("model", estimator)])
            pipe.fit(X.loc[train_idx], y.loc[train_idx])
            pred = pipe.predict(X.loc[valid_idx])
            elapsed = time.perf_counter() - started
            rows.append({"fold": fold, "model": name, "r2": r2_score(y.loc[valid_idx], pred), "mae": mean_absolute_error(y.loc[valid_idx], pred), "rmse": mean_squared_error(y.loc[valid_idx], pred) ** 0.5, "fit_seconds": elapsed})
            oof.loc[oof.Person_ID.isin(dev.loc[valid_idx, "Person_ID"]), f"pred_{name}"] = pred
    metrics = pd.DataFrame(rows)
    summary = metrics.groupby("model", as_index=False).agg(r2_mean=("r2", "mean"), r2_std=("r2", "std"), mae_mean=("mae", "mean"), rmse_mean=("rmse", "mean"), total_fit_seconds=("fit_seconds", "sum"))
    out = root / "outputs"
    for sub in ("tables", "predictions", "logs"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    prefix = "task1_non_sleep_baseline"
    paths = [out / "tables" / f"{prefix}_fold_metrics.csv", out / "tables" / f"{prefix}_summary_metrics.csv", out / "predictions" / f"{prefix}_oof_predictions.csv", out / "logs" / f"{prefix}_manifest.json"]
    if not overwrite and any(p.exists() for p in paths):
        raise FileExistsError("任务一基线输出已存在，请加 --overwrite")
    metrics.to_csv(paths[0], index=False, encoding="utf-8-sig")
    summary.to_csv(paths[1], index=False, encoding="utf-8-sig")
    oof.to_csv(paths[2], index=False, encoding="utf-8-sig")
    manifest = {"task": "task1", "target": TARGET, "forbidden": sorted(FORBIDDEN), "feature_count": len(features), "development_rows": len(dev), "cv_folds": 5, "seed": 2026, "holdout_evaluated": False, "models": list(model_factories())}
    paths[3].write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return paths
