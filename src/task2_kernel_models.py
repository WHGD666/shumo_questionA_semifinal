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
from sklearn.kernel_approximation import Nystroem, RBFSampler
from sklearn.linear_model import Ridge

from src.task2_baseline import SEED, _metrics, _sha256, make_preprocessor
from src.task2_elastic_tuning import _git_state
from src.task2_protocol import TARGET, feature_contracts
from src.task2_time_encoding import TIME_FIELDS, engineer_cyclic_times


CONTRACT = "scientific_proxy_removed"
GAMMAS = (0.001, 0.003, 0.01)
RBF_COMPONENTS = (256, 512)
NYSTROEM_COMPONENTS = (128, 256)
RIDGE_ALPHAS = (1.0, 10.0)
CONTROL_ADJUSTED_R2 = 0.6201624354997604
MINIMUM_MATERIAL_IMPROVEMENT = 0.001


def candidate_name(family: str, gamma: float, components: int, alpha: float) -> str:
    gamma_text = f"{gamma:g}".replace(".", "p")
    alpha_text = f"{alpha:g}".replace(".", "p")
    return f"{family}_g{gamma_text}_c{components}_a{alpha_text}"


def candidate_grid() -> list[dict[str, object]]:
    grid: list[dict[str, object]] = []
    for family, component_values in (
        ("rbf_sampler", RBF_COMPONENTS),
        ("nystroem", NYSTROEM_COMPONENTS),
    ):
        for gamma in GAMMAS:
            for components in component_values:
                for alpha in RIDGE_ALPHAS:
                    grid.append({
                        "model": candidate_name(family, gamma, components, alpha),
                        "family": family,
                        "gamma": gamma,
                        "kernel_components": components,
                        "ridge_alpha": alpha,
                    })
    return grid


def make_feature_map(family: str, gamma: float, components: int, seed: int):
    if family == "rbf_sampler":
        return RBFSampler(
            gamma=gamma,
            n_components=components,
            random_state=seed,
        )
    if family == "nystroem":
        return Nystroem(
            kernel="rbf",
            gamma=gamma,
            n_components=components,
            random_state=seed,
        )
    raise ValueError(f"未知核近似方法: {family}")


def run_kernel_models(root: Path, overwrite: bool = False) -> list[Path]:
    data_candidates = [root / "data" / "raw" / "A题数据集.csv", root / "A题数据集.csv"]
    data_path = next((path for path in data_candidates if path.exists()), None)
    if data_path is None:
        raise FileNotFoundError("未找到复赛数据集")
    split_path = root / "data" / "splits" / "split_assignments.csv"
    config_path = root / "configs" / "task2_kernel_models.toml"
    frame = pd.read_csv(data_path)
    assignments = pd.read_csv(split_path)
    merged = frame.merge(assignments, on="Person_ID", validate="one_to_one")
    development = merged.loc[merged["split"] == "development"].copy().reset_index(drop=True)
    if len(development) != 8000 or set(development["cv_fold"].unique()) != {0, 1, 2, 3, 4}:
        raise ValueError("任务二开发集划分不符合冻结协议")

    source_features = feature_contracts(frame.columns.tolist())[CONTRACT]
    X = engineer_cyclic_times(development[source_features].reset_index(drop=True))
    engineered_width = int(X.shape[1])
    expected_width = len(source_features) + sum(field in source_features for field in TIME_FIELDS)
    if engineered_width != expected_width:
        raise ValueError("周期时间特征数量与预期不一致")
    y = pd.to_numeric(development[TARGET]).reset_index(drop=True)
    folds = development["cv_fold"].astype(int).to_numpy()
    grid = candidate_grid()
    fold_rows: list[dict[str, object]] = []
    oof = pd.DataFrame({"Person_ID": development["Person_ID"], "true_value": y})
    for item in grid:
        oof[f"pred_{item['model']}"] = np.nan
    started_at = datetime.now().astimezone()
    started = time.perf_counter()

    for fold in range(5):
        train_positions = np.flatnonzero(folds != fold)
        valid_positions = np.flatnonzero(folds == fold)
        X_train, X_valid = X.iloc[train_positions], X.iloc[valid_positions]
        y_train, y_valid = y.iloc[train_positions], y.iloc[valid_positions]
        print(f"[kernel fold {fold + 1}/5] preparing development split...")
        preprocessor = make_preprocessor(X_train)
        transform_started = time.perf_counter()
        train_matrix = preprocessor.fit_transform(X_train)
        valid_matrix = preprocessor.transform(X_valid)
        preprocess_seconds = time.perf_counter() - transform_started
        transformed_width = int(train_matrix.shape[1])
        for family, component_values in (
            ("rbf_sampler", RBF_COMPONENTS),
            ("nystroem", NYSTROEM_COMPONENTS),
        ):
            for gamma in GAMMAS:
                for components in component_values:
                    map_started = time.perf_counter()
                    feature_map = make_feature_map(
                        family,
                        gamma,
                        components,
                        SEED + fold,
                    )
                    train_kernel = feature_map.fit_transform(train_matrix)
                    valid_kernel = feature_map.transform(valid_matrix)
                    map_seconds = time.perf_counter() - map_started
                    for alpha in RIDGE_ALPHAS:
                        name = candidate_name(family, gamma, components, alpha)
                        fit_started = time.perf_counter()
                        model = Ridge(alpha=alpha)
                        model.fit(train_kernel, y_train)
                        prediction = np.asarray(model.predict(valid_kernel)).reshape(-1)
                        metric = _metrics(
                            y_valid,
                            prediction,
                            raw_p=engineered_width,
                            transformed_p=components,
                        )
                        fold_rows.append({
                            "fold": fold,
                            "model": name,
                            "family": family,
                            "gamma": gamma,
                            "kernel_components": components,
                            "ridge_alpha": alpha,
                            "engineered_feature_count": engineered_width,
                            "preprocessed_width": transformed_width,
                            **metric,
                            "preprocess_seconds": preprocess_seconds,
                            "kernel_map_seconds": map_seconds,
                            "fit_seconds": time.perf_counter() - fit_started,
                        })
                        oof.loc[valid_positions, f"pred_{name}"] = prediction
        fold_best = max(
            (row for row in fold_rows if row["fold"] == fold),
            key=lambda row: row["adjusted_r2_raw"],
        )
        print(f"  best={fold_best['model']}: adjR2={fold_best['adjusted_r2_raw']:.4f}")

    fold_metrics = pd.DataFrame(fold_rows)
    summary_rows: list[dict[str, object]] = []
    for item in grid:
        model = str(item["model"])
        components = int(item["kernel_components"])
        subset = fold_metrics.loc[fold_metrics["model"] == model]
        prediction = oof[f"pred_{model}"].to_numpy()
        if np.isnan(prediction).any():
            raise ValueError(f"{model} 的 OOF 预测不完整")
        pooled = _metrics(
            y,
            prediction,
            raw_p=engineered_width,
            transformed_p=components,
        )
        summary_rows.append({
            "contract": CONTRACT,
            **item,
            "engineered_feature_count": engineered_width,
            "preprocessed_width_max": int(subset["preprocessed_width"].max()),
            "fold_adjusted_r2_raw_mean": float(subset["adjusted_r2_raw"].mean()),
            "fold_adjusted_r2_raw_std": float(subset["adjusted_r2_raw"].std()),
            "oof_adjusted_r2_raw": pooled["adjusted_r2_raw"],
            "oof_adjusted_r2_kernel_components": pooled["adjusted_r2_transformed"],
            "oof_r2": pooled["r2"],
            "oof_mae": pooled["mae"],
            "oof_rmse": pooled["rmse"],
            "control_adjusted_r2_raw": CONTROL_ADJUSTED_R2,
            "delta_vs_control_adjusted_r2": pooled["adjusted_r2_raw"] - CONTROL_ADJUSTED_R2,
            "total_kernel_map_seconds": float(subset["kernel_map_seconds"].sum()),
            "total_fit_seconds": float(subset["fit_seconds"].sum()),
        })
    summary = pd.DataFrame(summary_rows).sort_values(
        ["oof_adjusted_r2_raw", "fold_adjusted_r2_raw_std", "kernel_components"],
        ascending=[False, True, True],
    )

    output_root = root / "outputs"
    for directory in ("tables", "predictions", "logs", "logs/history"):
        (output_root / directory).mkdir(parents=True, exist_ok=True)
    prefix = "task2_cyclic_kernel_approximation_models"
    paths = [
        output_root / "tables" / f"{prefix}_fold_metrics.csv",
        output_root / "tables" / f"{prefix}_summary_metrics.csv",
        output_root / "predictions" / f"{prefix}_oof_predictions.csv",
        output_root / "logs" / f"{prefix}_manifest.json",
    ]
    if not overwrite and any(path.exists() for path in paths):
        raise FileExistsError("任务二核近似模型输出已存在；确认重跑时请使用 --overwrite")

    identity = hashlib.sha256(
        f"task2|kernel_approximation|{_sha256(data_path)}|{_sha256(split_path)}|{_sha256(config_path)}".encode()
    ).hexdigest()[:8]
    run_id = f"{started_at.strftime('%Y%m%dT%H%M%S%z')}_task2_kernel_{identity}"
    best = summary.iloc[0]
    manifest = {
        "run_id": run_id,
        "status": "completed",
        "task": "task2",
        "target": TARGET,
        "experiment": prefix,
        "experiment_role": "controlled_smooth_interactions",
        "command": "python scripts/run_task2_kernel_models.py",
        "working_directory": str(root),
        "git": _git_state(root),
        "feature_contract": CONTRACT,
        "source_feature_count": len(source_features),
        "engineered_feature_count": engineered_width,
        "time_encoding": "sin_cos_1440_minutes",
        "candidate_grid": grid,
        "candidate_count": len(grid),
        "adjusted_r2_p_definition": "primary uses engineered raw feature count; kernel-component adjustment is diagnostic",
        "control_oof_adjusted_r2": CONTROL_ADJUSTED_R2,
        "best_model": str(best["model"]),
        "best_family": str(best["family"]),
        "best_gamma": float(best["gamma"]),
        "best_kernel_components": int(best["kernel_components"]),
        "best_ridge_alpha": float(best["ridge_alpha"]),
        "best_oof_adjusted_r2": float(best["oof_adjusted_r2_raw"]),
        "best_delta_vs_control_adjusted_r2": float(best["delta_vs_control_adjusted_r2"]),
        "minimum_material_improvement": MINIMUM_MATERIAL_IMPROVEMENT,
        "material_improvement": bool(
            best["delta_vs_control_adjusted_r2"] >= MINIMUM_MATERIAL_IMPROVEMENT
        ),
        "requires_seed_confirmation_if_promoted": True,
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
