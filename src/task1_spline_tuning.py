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
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

from src.task1_regression_baseline import FORBIDDEN, TARGET


SEED = 2026
KNOTS = (4, 5, 6, 7)
DEGREES = (2, 3)
ALPHAS = (3.0, 10.0, 30.0)


def model_name(n_knots: int, degree: int, alpha: float) -> str:
    return f"spline_k{n_knots}_d{degree}_a{str(alpha).replace('.', 'p')}"


def parameter_grid() -> list[dict[str, float | int]]:
    return [
        {"n_knots": knots, "degree": degree, "alpha": alpha}
        for knots in KNOTS
        for degree in DEGREES
        for alpha in ALPHAS
    ]


def make_preprocessor(frame: pd.DataFrame, n_knots: int, degree: int) -> ColumnTransformer:
    numeric = [column for column in frame.columns if pd.api.types.is_numeric_dtype(frame[column])]
    categorical = [column for column in frame.columns if column not in numeric]
    return ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("spline", SplineTransformer(n_knots=n_knots, degree=degree, include_bias=False)),
            ("scale", StandardScaler()),
        ]), numeric),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)),
        ]), categorical),
    ], sparse_threshold=0.0)


def _metrics(y_true, prediction) -> dict[str, float]:
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


def run_spline_tuning(
    root: Path,
    overwrite: bool = False,
    *,
    knots_values: tuple[int, ...] = KNOTS,
    degree_values: tuple[int, ...] = DEGREES,
    alpha_values: tuple[float, ...] = ALPHAS,
    config_filename: str = "task1_spline_tuning.toml",
    output_prefix: str = "task1_non_sleep_spline_tuning",
    experiment_name: str = "non_sleep_spline_tuning",
    control_model: str = "spline_k5_d3_a10p0",
    control_oof_r2: float = 0.7869007019669725,
) -> list[Path]:
    candidates = [root / "data" / "raw" / "A题数据集.csv", root / "A题数据集.csv"]
    data_path = next((path for path in candidates if path.exists()), None)
    if data_path is None:
        raise FileNotFoundError("未找到复赛数据集")
    split_path = root / "data" / "splits" / "split_assignments.csv"
    config_path = root / "configs" / config_filename
    frame = pd.read_csv(data_path)
    assignments = pd.read_csv(split_path)
    merged = frame.merge(assignments, on="Person_ID", validate="one_to_one")
    development = merged.loc[merged["split"] == "development"].copy()
    if len(development) != 8000 or set(development["cv_fold"].unique()) != {0, 1, 2, 3, 4}:
        raise ValueError("开发集划分不符合冻结协议")
    features = [column for column in frame.columns if column not in FORBIDDEN]
    X, y = development[features], pd.to_numeric(development[TARGET])
    grid = [
        {"n_knots": knots, "degree": degree, "alpha": alpha}
        for knots in knots_values
        for degree in degree_values
        for alpha in alpha_values
    ]
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
        for knots in knots_values:
            for degree in degree_values:
                preprocessor = make_preprocessor(X_train, knots, degree)
                transform_started = time.perf_counter()
                train_matrix = preprocessor.fit_transform(X_train)
                valid_matrix = preprocessor.transform(X_valid)
                transform_seconds = time.perf_counter() - transform_started
                width = int(train_matrix.shape[1])
                for alpha in alpha_values:
                    name = model_name(knots, degree, alpha)
                    fit_started = time.perf_counter()
                    model = Ridge(alpha=alpha)
                    model.fit(train_matrix, y_train)
                    prediction = model.predict(valid_matrix)
                    metric = _metrics(y_valid, prediction)
                    fold_rows.append({
                        "fold": fold,
                        "model": name,
                        "n_knots": knots,
                        "degree": degree,
                        "alpha": alpha,
                        "transformed_width": width,
                        **metric,
                        "transform_seconds": transform_seconds,
                        "fit_seconds": time.perf_counter() - fit_started,
                    })
                    mapping = dict(zip(valid_ids, prediction))
                    mask = oof["Person_ID"].isin(mapping)
                    oof.loc[mask, f"pred_{name}"] = oof.loc[mask, "Person_ID"].map(mapping)
        fold_best = max((row for row in fold_rows if row["fold"] == fold), key=lambda row: row["r2"])
        print(f"  best={fold_best['model']}: R2={fold_best['r2']:.4f}")

    fold_metrics = pd.DataFrame(fold_rows)
    summary_rows = []
    for parameters in grid:
        name = model_name(parameters["n_knots"], parameters["degree"], parameters["alpha"])
        subset = fold_metrics.loc[fold_metrics["model"] == name]
        prediction = oof[f"pred_{name}"].to_numpy()
        if np.isnan(prediction).any():
            raise ValueError(f"{name} 的 OOF 预测不完整")
        pooled = _metrics(oof["true_value"], prediction)
        summary_rows.append({
            "model": name,
            **parameters,
            "transformed_width": int(subset["transformed_width"].iloc[0]),
            "fold_r2_mean": float(subset["r2"].mean()),
            "fold_r2_std": float(subset["r2"].std()),
            "oof_r2": pooled["r2"],
            "oof_mae": pooled["mae"],
            "oof_rmse": pooled["rmse"],
            "total_fit_seconds": float(subset["fit_seconds"].sum()),
        })
    summary = pd.DataFrame(summary_rows).sort_values(
        ["oof_r2", "fold_r2_std", "transformed_width"], ascending=[False, True, True]
    )

    output_root = root / "outputs"
    for directory in ("tables", "predictions", "logs", "logs/history"):
        (output_root / directory).mkdir(parents=True, exist_ok=True)
    prefix = output_prefix
    paths = [
        output_root / "tables" / f"{prefix}_fold_metrics.csv",
        output_root / "tables" / f"{prefix}_summary_metrics.csv",
        output_root / "predictions" / f"{prefix}_oof_predictions.csv",
        output_root / "logs" / f"{prefix}_manifest.json",
    ]
    if not overwrite and any(path.exists() for path in paths):
        raise FileExistsError("样条调参输出已存在；确认重跑时请使用 --overwrite")
    identity = hashlib.sha256(
        f"task1|{experiment_name}|{_sha256(data_path)}|{_sha256(split_path)}|{_sha256(config_path)}".encode()
    ).hexdigest()[:8]
    run_id = f"{started_at.strftime('%Y%m%dT%H%M%S%z')}_task1_{experiment_name}_{identity}"
    best = summary.iloc[0]
    manifest = {
        "run_id": run_id,
        "status": "completed",
        "task": "task1",
        "target": TARGET,
        "experiment": experiment_name,
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
        "parameter_grid": grid,
        "control_model": control_model,
        "control_oof_r2": control_oof_r2,
        "best_model": best["model"],
        "best_parameters": {
            "n_knots": int(best["n_knots"]),
            "degree": int(best["degree"]),
            "alpha": float(best["alpha"]),
        },
        "best_oof_r2": float(best["oof_r2"]),
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
