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
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

from src.task1_regression_baseline import FORBIDDEN, TARGET, make_preprocessor as make_linear_preprocessor
from src.task1_spline_tuning import make_preprocessor as make_spline_preprocessor


SEED = 2026
INNER_FOLDS = 4
WEIGHTS = (0.25, 0.5, 0.75, 1.0)
CONTROL_OOF_R2 = 0.7906023958255962


def candidate_name(residual_model: str, weight: float) -> str:
    return f"{residual_model}_residual_w{str(weight).replace('.', 'p')}"


def blend_prediction(base_prediction, residual_prediction, weight: float) -> np.ndarray:
    return np.asarray(base_prediction, dtype=float) + weight * np.asarray(residual_prediction, dtype=float)


def nested_inner_splits(
    row_count: int,
    outer_fold: int,
    inner_folds: int = INNER_FOLDS,
    seed: int = SEED,
):
    splitter = KFold(n_splits=inner_folds, shuffle=True, random_state=seed + outer_fold)
    return list(splitter.split(np.arange(row_count)))


def _base_pipeline(frame: pd.DataFrame) -> Pipeline:
    return Pipeline([
        ("preprocess", make_spline_preprocessor(frame, n_knots=4, degree=2)),
        ("model", Ridge(alpha=1.0)),
    ])


def _residual_pipeline(frame: pd.DataFrame, name: str, seed: int) -> Pipeline:
    if name == "ridge":
        estimator = Ridge(alpha=10.0)
    elif name == "lightgbm":
        from lightgbm import LGBMRegressor

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
    return Pipeline([("preprocess", make_linear_preprocessor(frame)), ("model", estimator)])


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


def run_residual_correction(root: Path, overwrite: bool = False) -> list[Path]:
    data_candidates = [root / "data" / "raw" / "A题数据集.csv", root / "A题数据集.csv"]
    data_path = next((path for path in data_candidates if path.exists()), None)
    if data_path is None:
        raise FileNotFoundError("未找到复赛数据集")
    split_path = root / "data" / "splits" / "split_assignments.csv"
    config_path = root / "configs" / "task1_residual_correction.toml"
    frame = pd.read_csv(data_path)
    assignments = pd.read_csv(split_path)
    merged = frame.merge(assignments, on="Person_ID", validate="one_to_one")
    development = merged.loc[merged["split"] == "development"].copy()
    if len(development) != 8000 or set(development["cv_fold"].unique()) != {0, 1, 2, 3, 4}:
        raise ValueError("开发集划分不符合冻结协议")

    features = [column for column in frame.columns if column not in FORBIDDEN]
    X = development[features].reset_index(drop=True)
    y = pd.to_numeric(development[TARGET]).reset_index(drop=True)
    person_ids = development["Person_ID"].reset_index(drop=True)
    folds = development["cv_fold"].astype(int).reset_index(drop=True)
    candidate_names = ["base_spline"] + [
        candidate_name(model, weight) for model in ("ridge", "lightgbm") for weight in WEIGHTS
    ]
    predictions = {name: np.full(len(development), np.nan) for name in candidate_names}
    fold_rows: list[dict[str, object]] = []
    started_at = datetime.now().astimezone()
    total_started = time.perf_counter()

    for outer_fold in range(5):
        train_positions = np.flatnonzero(folds.to_numpy() != outer_fold)
        valid_positions = np.flatnonzero(folds.to_numpy() == outer_fold)
        X_train, X_valid = X.iloc[train_positions].copy(), X.iloc[valid_positions].copy()
        y_train, y_valid = y.iloc[train_positions], y.iloc[valid_positions]
        print(f"[outer fold {outer_fold + 1}/5] generating leakage-safe residual targets...")

        inner_oof = np.full(len(train_positions), np.nan)
        for inner_train, inner_valid in nested_inner_splits(len(train_positions), outer_fold):
            inner_model = _base_pipeline(X_train.iloc[inner_train])
            inner_model.fit(X_train.iloc[inner_train], y_train.iloc[inner_train])
            inner_oof[inner_valid] = inner_model.predict(X_train.iloc[inner_valid])
        if np.isnan(inner_oof).any():
            raise ValueError("内层 OOF 主模型预测不完整")

        base_started = time.perf_counter()
        base_model = _base_pipeline(X_train)
        base_model.fit(X_train, y_train)
        base_valid = base_model.predict(X_valid)
        base_seconds = time.perf_counter() - base_started
        predictions["base_spline"][valid_positions] = base_valid
        fold_rows.append({"fold": outer_fold, "model": "base_spline", **_metrics(y_valid, base_valid), "fit_seconds": base_seconds})

        residual_target = y_train.to_numpy() - inner_oof
        residual_train = X_train.copy()
        residual_valid = X_valid.copy()
        residual_train["Base_Prediction"] = inner_oof
        residual_valid["Base_Prediction"] = base_valid
        for residual_name in ("ridge", "lightgbm"):
            fit_started = time.perf_counter()
            residual_model = _residual_pipeline(residual_train, residual_name, SEED + outer_fold)
            residual_model.fit(residual_train, residual_target)
            residual_valid_prediction = residual_model.predict(residual_valid)
            residual_seconds = time.perf_counter() - fit_started
            for weight in WEIGHTS:
                name = candidate_name(residual_name, weight)
                corrected = blend_prediction(base_valid, residual_valid_prediction, weight)
                predictions[name][valid_positions] = corrected
                fold_rows.append({
                    "fold": outer_fold,
                    "model": name,
                    "residual_model": residual_name,
                    "weight": weight,
                    **_metrics(y_valid, corrected),
                    "fit_seconds": residual_seconds,
                })
        fold_best = max((row for row in fold_rows if row["fold"] == outer_fold), key=lambda row: row["r2"])
        print(f"  best={fold_best['model']}: R2={fold_best['r2']:.4f}")

    fold_metrics = pd.DataFrame(fold_rows)
    summary_rows = []
    for name in candidate_names:
        prediction = predictions[name]
        if np.isnan(prediction).any():
            raise ValueError(f"{name} 的外层 OOF 预测不完整")
        subset = fold_metrics.loc[fold_metrics["model"] == name]
        pooled = _metrics(y, prediction)
        summary_rows.append({
            "model": name,
            "fold_r2_mean": float(subset["r2"].mean()),
            "fold_r2_std": float(subset["r2"].std()),
            "oof_r2": pooled["r2"],
            "oof_mae": pooled["mae"],
            "oof_rmse": pooled["rmse"],
            "delta_vs_control_r2": pooled["r2"] - CONTROL_OOF_R2,
            "total_fit_seconds": float(subset["fit_seconds"].sum()),
        })
    summary = pd.DataFrame(summary_rows).sort_values(["oof_r2", "fold_r2_std"], ascending=[False, True])
    oof = pd.DataFrame({"Person_ID": person_ids, "true_value": y})
    for name in candidate_names:
        oof[f"pred_{name}"] = predictions[name]

    output_root = root / "outputs"
    for directory in ("tables", "predictions", "logs", "logs/history"):
        (output_root / directory).mkdir(parents=True, exist_ok=True)
    prefix = "task1_non_sleep_nested_residual_correction"
    paths = [
        output_root / "tables" / f"{prefix}_fold_metrics.csv",
        output_root / "tables" / f"{prefix}_summary_metrics.csv",
        output_root / "predictions" / f"{prefix}_oof_predictions.csv",
        output_root / "logs" / f"{prefix}_manifest.json",
    ]
    if not overwrite and any(path.exists() for path in paths):
        raise FileExistsError("残差修正输出已存在；确认重跑时请使用 --overwrite")
    identity = hashlib.sha256(
        f"task1|nested_residual|{_sha256(data_path)}|{_sha256(split_path)}|{_sha256(config_path)}".encode()
    ).hexdigest()[:8]
    run_id = f"{started_at.strftime('%Y%m%dT%H%M%S%z')}_task1_nested_residual_{identity}"
    best = summary.iloc[0]
    manifest = {
        "run_id": run_id,
        "status": "completed",
        "task": "task1",
        "target": TARGET,
        "experiment": "non_sleep_nested_residual_correction",
        "experiment_role": "controlled_improvement",
        "method": "outer 5-fold evaluation with inner 4-fold cross-fitted base predictions for residual targets",
        "duration_seconds": time.perf_counter() - total_started,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "dependencies": {"numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
        "data_sha256": _sha256(data_path),
        "split_sha256": _sha256(split_path),
        "config_sha256": _sha256(config_path),
        "feature_count": len(features),
        "forbidden_features": sorted(FORBIDDEN),
        "outer_folds": 5,
        "inner_folds": INNER_FOLDS,
        "base_model": {"family": "spline_ridge", "n_knots": 4, "degree": 2, "alpha": 1.0},
        "residual_models": ["ridge", "lightgbm"],
        "blend_weights": list(WEIGHTS),
        "control_oof_r2": CONTROL_OOF_R2,
        "best_model": str(best["model"]),
        "best_oof_r2": float(best["oof_r2"]),
        "best_delta_vs_control_r2": float(best["delta_vs_control_r2"]),
        "minimum_material_improvement": 0.001,
        "material_improvement_achieved": bool(best["delta_vs_control_r2"] >= 0.001),
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
