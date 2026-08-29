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

from src.task2_time_encoding import TIME_FIELDS, engineer_cyclic_times
from src.task3_additive_models import (
    make_estimator,
    make_preprocessor,
    model_name,
    spline_features_for_contract,
)
from src.task3_baseline import SEED, _metrics, _sha256
from src.task3_linear_tuning import _git_state
from src.task3_protocol import TARGET, feature_contracts


KNOTS = (4, 5, 6)
DEGREE = 2
COMPETITION_ALPHAS = (0.001, 0.003, 0.005, 0.01)
SCIENTIFIC_ALPHAS = (0.1, 0.3, 1.0, 3.0)
CONTROL_OOF_R2 = {
    "competition_proxy_inclusive": 0.9807002146476884,
    "scientific_proxy_removed": 0.9383925875801471,
}
MINIMUM_MATERIAL_IMPROVEMENT = 0.0002


def candidate_grid(contract: str) -> list[dict[str, object]]:
    if contract == "competition_proxy_inclusive":
        penalty = "elastic"
        alphas = COMPETITION_ALPHAS
        l1_ratio = 0.9
    elif contract == "scientific_proxy_removed":
        penalty = "ridge"
        alphas = SCIENTIFIC_ALPHAS
        l1_ratio = 0.0
    else:
        raise ValueError(f"未知任务三特征契约：{contract}")
    return [
        {
            "model": model_name(penalty, n_knots, DEGREE, alpha),
            "penalty": penalty,
            "n_knots": n_knots,
            "degree": DEGREE,
            "alpha": alpha,
            "l1_ratio": l1_ratio,
        }
        for n_knots in KNOTS
        for alpha in alphas
    ]


def run_task3_additive_refinement(root: Path, overwrite: bool = False) -> list[Path]:
    data_candidates = [root / "data" / "raw" / "A题数据集.csv", root / "A题数据集.csv"]
    data_path = next((path for path in data_candidates if path.exists()), None)
    if data_path is None:
        raise FileNotFoundError("未找到复赛数据集")
    split_path = root / "data" / "splits" / "split_assignments.csv"
    config_path = root / "configs" / "task3_additive_refinement.toml"
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
        contract_grid = candidate_grid(contract)
        contract_splines = spline_features_for_contract(contract)
        X = engineer_cyclic_times(development[source_features].reset_index(drop=True))
        engineered_width = int(X.shape[1])
        expected_width = len(source_features) + sum(field in source_features for field in TIME_FIELDS)
        if engineered_width != expected_width:
            raise ValueError("任务三周期时间特征数量与预期不一致")
        for item in contract_grid:
            oof[f"pred_{contract}_{item['model']}"] = np.nan
        for fold in range(5):
            train_positions = np.flatnonzero(folds != fold)
            valid_positions = np.flatnonzero(folds == fold)
            X_train, X_valid = X.iloc[train_positions], X.iloc[valid_positions]
            y_train, y_valid = y.iloc[train_positions], y.iloc[valid_positions]
            print(f"[{contract} refinement fold {fold + 1}/5] preparing development split...")
            for n_knots in KNOTS:
                preprocessor = make_preprocessor(
                    X_train, contract_splines, n_knots=n_knots, degree=DEGREE
                )
                transform_started = time.perf_counter()
                train_matrix = preprocessor.fit_transform(X_train)
                valid_matrix = preprocessor.transform(X_valid)
                transform_seconds = time.perf_counter() - transform_started
                transformed_width = int(train_matrix.shape[1])
                knot_items = [item for item in contract_grid if item["n_knots"] == n_knots]
                for item in knot_items:
                    estimator = make_estimator(
                        str(item["penalty"]), float(item["alpha"]), SEED + fold
                    )
                    fit_started = time.perf_counter()
                    estimator.fit(train_matrix, y_train)
                    prediction = np.asarray(estimator.predict(valid_matrix)).reshape(-1)
                    metric = _metrics(y_valid, prediction)
                    fold_rows.append({
                        "contract": contract,
                        "fold": fold,
                        **item,
                        "source_feature_count": len(source_features),
                        "spline_feature_count": len(contract_splines),
                        "engineered_feature_count": engineered_width,
                        "transformed_width": transformed_width,
                        **metric,
                        "transform_seconds": transform_seconds,
                        "fit_seconds": time.perf_counter() - fit_started,
                    })
                    oof.loc[
                        valid_positions, f"pred_{contract}_{item['model']}"
                    ] = prediction
            fold_best = max(
                (row for row in fold_rows if row["contract"] == contract and row["fold"] == fold),
                key=lambda row: row["r2"],
            )
            print(f"  best={fold_best['model']}: R2={fold_best['r2']:.6f}")

    fold_metrics = pd.DataFrame(fold_rows)
    summary_rows: list[dict[str, object]] = []
    for contract, source_features in contracts.items():
        contract_splines = spline_features_for_contract(contract)
        engineered_width = len(source_features) + sum(field in source_features for field in TIME_FIELDS)
        for item in candidate_grid(contract):
            model = str(item["model"])
            subset = fold_metrics.loc[
                (fold_metrics["contract"] == contract) & (fold_metrics["model"] == model)
            ]
            prediction = oof[f"pred_{contract}_{model}"].to_numpy()
            if np.isnan(prediction).any():
                raise ValueError(f"{contract}/{model} 的 OOF 预测不完整")
            pooled = _metrics(y, prediction)
            summary_rows.append({
                "contract": contract,
                **item,
                "source_feature_count": len(source_features),
                "spline_feature_count": len(contract_splines),
                "engineered_feature_count": engineered_width,
                "transformed_width_max": int(subset["transformed_width"].max()),
                "fold_r2_mean": float(subset["r2"].mean()),
                "fold_r2_std": float(subset["r2"].std()),
                "oof_r2": pooled["r2"],
                "oof_mae": pooled["mae"],
                "oof_rmse": pooled["rmse"],
                "control_oof_r2": CONTROL_OOF_R2[contract],
                "delta_vs_additive_control_r2": pooled["r2"] - CONTROL_OOF_R2[contract],
                "total_fit_seconds": float(subset["fit_seconds"].sum()),
            })
    summary = pd.DataFrame(summary_rows).sort_values(
        ["contract", "oof_r2", "fold_r2_std", "transformed_width_max"],
        ascending=[True, False, True, True],
    )
    winners = summary.groupby("contract", sort=False).head(1).copy()
    winners["material_improvement"] = (
        winners["delta_vs_additive_control_r2"] >= MINIMUM_MATERIAL_IMPROVEMENT
    )

    output_root = root / "outputs"
    for directory in ("tables", "predictions", "logs", "logs/history"):
        (output_root / directory).mkdir(parents=True, exist_ok=True)
    prefix = "task3_health_additive_refinement"
    paths = [
        output_root / "tables" / f"{prefix}_fold_metrics.csv",
        output_root / "tables" / f"{prefix}_summary_metrics.csv",
        output_root / "tables" / f"{prefix}_winners.csv",
        output_root / "predictions" / f"{prefix}_oof_predictions.csv",
        output_root / "logs" / f"{prefix}_manifest.json",
    ]
    if not overwrite and any(path.exists() for path in paths):
        raise FileExistsError("任务三加性边界确认输出已存在；确认重跑时请使用 --overwrite")

    identity = hashlib.sha256(
        f"task3|additive_refinement|{_sha256(data_path)}|{_sha256(split_path)}|{_sha256(config_path)}".encode()
    ).hexdigest()[:8]
    run_id = f"{started_at.strftime('%Y%m%dT%H%M%S%z')}_task3_additive_refinement_{identity}"
    best_by_contract = {
        row["contract"]: {
            "model": row["model"],
            "penalty": row["penalty"],
            "n_knots": int(row["n_knots"]),
            "degree": int(row["degree"]),
            "alpha": float(row["alpha"]),
            "l1_ratio": float(row["l1_ratio"]),
            "oof_r2": float(row["oof_r2"]),
            "delta_vs_additive_control_r2": float(row["delta_vs_additive_control_r2"]),
            "material_improvement": bool(row["material_improvement"]),
        }
        for row in winners.to_dict(orient="records")
    }
    manifest = {
        "run_id": run_id,
        "status": "completed",
        "task": "task3",
        "target": TARGET,
        "experiment": prefix,
        "experiment_role": "boundary_refinement",
        "command": "python scripts/run_task3_additive_refinement.py",
        "working_directory": str(root),
        "git": _git_state(root),
        "feature_contracts": {name: len(features) for name, features in contracts.items()},
        "time_encoding": "sin_cos_1440_minutes",
        "spline_features": {
            name: list(spline_features_for_contract(name)) for name in contracts
        },
        "grid": {
            name: candidate_grid(name) for name in contracts
        },
        "candidate_count_per_contract": {
            name: len(candidate_grid(name)) for name in contracts
        },
        "control_oof_r2": CONTROL_OOF_R2,
        "minimum_material_improvement": MINIMUM_MATERIAL_IMPROVEMENT,
        "best_by_contract": best_by_contract,
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
    winners.to_csv(paths[2], index=False, encoding="utf-8-sig")
    oof.to_csv(paths[3], index=False, encoding="utf-8-sig")
    manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2)
    paths[4].write_text(manifest_text, encoding="utf-8")
    history = output_root / "logs" / "history" / f"{run_id}_manifest.json"
    history.write_text(manifest_text, encoding="utf-8")
    return [*paths, history]
