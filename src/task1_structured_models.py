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
from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.cross_decomposition import PLSRegression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

from src.task1_regression_baseline import FORBIDDEN, TARGET, make_preprocessor


SEED = 2026


def candidate_names() -> tuple[str, ...]:
    return ("pls_8", "pls_16", "spline_ridge", "catboost_native")


def make_spline_preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    numeric = [column for column in frame.columns if pd.api.types.is_numeric_dtype(frame[column])]
    categorical = [column for column in frame.columns if column not in numeric]
    return ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("spline", SplineTransformer(n_knots=5, degree=3, include_bias=False)),
            ("scale", StandardScaler()),
        ]), numeric),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)),
        ]), categorical),
    ], sparse_threshold=0.0)


def make_pls_pipeline(frame: pd.DataFrame, components: int) -> Pipeline:
    preprocessor = make_preprocessor(frame)
    preprocessor.set_params(cat__onehot__sparse_output=False)
    return Pipeline([
        ("preprocess", preprocessor),
        ("model", PLSRegression(n_components=components, max_iter=1000, scale=False)),
    ])


def prepare_native_catboost(
    train: pd.DataFrame, valid: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, list[int]]:
    train_out, valid_out = train.copy(), valid.copy()
    categorical = [column for column in train.columns if not pd.api.types.is_numeric_dtype(train[column])]
    numeric = [column for column in train.columns if column not in categorical]
    for column in numeric:
        median = pd.to_numeric(train[column], errors="coerce").median()
        train_out[column] = pd.to_numeric(train[column], errors="coerce").fillna(median).astype(float)
        valid_out[column] = pd.to_numeric(valid[column], errors="coerce").fillna(median).astype(float)
    for column in categorical:
        train_out[column] = train[column].fillna("Missing_Unknown").astype(str)
        valid_out[column] = valid[column].fillna("Missing_Unknown").astype(str)
    indices = [train_out.columns.get_loc(column) for column in categorical]
    return train_out, valid_out, indices


def _metrics(y_true, prediction) -> dict[str, float]:
    prediction = np.asarray(prediction).ravel()
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


def run_structured_models(root: Path, overwrite: bool = False) -> list[Path]:
    data_candidates = [root / "data" / "raw" / "A题数据集.csv", root / "A题数据集.csv"]
    data_path = next((path for path in data_candidates if path.exists()), None)
    if data_path is None:
        raise FileNotFoundError("未找到复赛数据集")
    split_path = root / "data" / "splits" / "split_assignments.csv"
    config_path = root / "configs" / "task1_structured_models.toml"
    frame = pd.read_csv(data_path)
    assignments = pd.read_csv(split_path)
    merged = frame.merge(assignments, on="Person_ID", validate="one_to_one")
    development = merged.loc[merged["split"] == "development"].copy()
    if len(development) != 8000 or set(development["cv_fold"].unique()) != {0, 1, 2, 3, 4}:
        raise ValueError("开发集划分不符合冻结协议")
    features = [column for column in frame.columns if column not in FORBIDDEN]
    X, y = development[features], pd.to_numeric(development[TARGET])
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
        print(f"[fold {fold + 1}/5] preparing development split...")

        candidates: list[tuple[str, object, object, object]] = [
            ("pls_8", make_pls_pipeline(X_train, 8), X_train, X_valid),
            ("pls_16", make_pls_pipeline(X_train, 16), X_train, X_valid),
            ("spline_ridge", Pipeline([
                ("preprocess", make_spline_preprocessor(X_train)),
                ("model", Ridge(alpha=10.0)),
            ]), X_train, X_valid),
        ]
        native_train, native_valid, cat_indices = prepare_native_catboost(X_train, X_valid)
        candidates.append((
            "catboost_native",
            CatBoostRegressor(
                iterations=800, learning_rate=0.03, depth=6, l2_leaf_reg=5.0,
                loss_function="RMSE", random_seed=SEED, verbose=False,
                allow_writing_files=False, thread_count=-1, cat_features=cat_indices,
            ),
            native_train,
            native_valid,
        ))

        for name, model, train_matrix, valid_matrix in candidates:
            fit_started = time.perf_counter()
            model.fit(train_matrix, y_train)
            prediction = np.asarray(model.predict(valid_matrix)).ravel()
            metric = _metrics(y_valid, prediction)
            elapsed = time.perf_counter() - fit_started
            print(f"  {name}: R2={metric['r2']:.4f}, MAE={metric['mae']:.4f}, RMSE={metric['rmse']:.4f}")
            fold_rows.append({"fold": fold, "model": name, **metric, "fit_seconds": elapsed})
            mapping = dict(zip(valid_ids, prediction))
            mask = oof["Person_ID"].isin(mapping)
            oof.loc[mask, f"pred_{name}"] = oof.loc[mask, "Person_ID"].map(mapping)

    fold_metrics = pd.DataFrame(fold_rows)
    summary_rows = []
    for name in candidate_names():
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
    summary = pd.DataFrame(summary_rows).sort_values(["oof_r2", "fold_r2_std"], ascending=[False, True])

    output_root = root / "outputs"
    for directory in ("tables", "predictions", "logs", "logs/history"):
        (output_root / directory).mkdir(parents=True, exist_ok=True)
    prefix = "task1_non_sleep_structured_models"
    paths = [
        output_root / "tables" / f"{prefix}_fold_metrics.csv",
        output_root / "tables" / f"{prefix}_summary_metrics.csv",
        output_root / "predictions" / f"{prefix}_oof_predictions.csv",
        output_root / "logs" / f"{prefix}_manifest.json",
    ]
    if not overwrite and any(path.exists() for path in paths):
        raise FileExistsError("结构化模型输出已存在；确认重跑时请使用 --overwrite")
    identity = hashlib.sha256(
        f"task1|structured|{_sha256(data_path)}|{_sha256(split_path)}|{_sha256(config_path)}".encode()
    ).hexdigest()[:8]
    run_id = f"{started_at.strftime('%Y%m%dT%H%M%S%z')}_task1_structured_{identity}"
    manifest = {
        "run_id": run_id,
        "status": "completed",
        "task": "task1",
        "target": TARGET,
        "experiment": "non_sleep_structured_models",
        "experiment_role": "controlled_improvement",
        "duration_seconds": time.perf_counter() - total_started,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "dependencies": {"numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
        "data_sha256": _sha256(data_path),
        "split_sha256": _sha256(split_path),
        "config_sha256": _sha256(config_path),
        "feature_count": len(features),
        "forbidden_features": sorted(FORBIDDEN),
        "models": list(candidate_names()),
        "control_model": "elastic_alpha_0p001_l1_0p9",
        "control_oof_r2": 0.7783725670589594,
        "best_model": summary.iloc[0]["model"],
        "best_oof_r2": float(summary.iloc[0]["oof_r2"]),
        "holdout_evaluated": False,
    }
    fold_metrics.to_csv(paths[0], index=False, encoding="utf-8-sig")
    summary.to_csv(paths[1], index=False, encoding="utf-8-sig")
    oof.to_csv(paths[2], index=False, encoding="utf-8-sig")
    manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2)
    paths[3].write_text(manifest_text, encoding="utf-8")
    history = output_root / "logs" / "history" / f"{run_id}_manifest.json"
    history.write_text(manifest_text, encoding="utf-8")
    return [*paths, history]
