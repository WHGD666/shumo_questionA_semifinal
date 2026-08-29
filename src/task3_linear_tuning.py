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
from sklearn.linear_model import ElasticNet, Ridge

from src.task2_baseline import make_preprocessor
from src.task2_time_encoding import TIME_FIELDS, engineer_cyclic_times
from src.task3_baseline import SEED, _metrics, _sha256
from src.task3_protocol import TARGET, feature_contracts


RIDGE_ALPHAS = (0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
ELASTIC_ALPHAS = (0.0001, 0.0003, 0.001, 0.003, 0.01)
L1_RATIOS = (0.1, 0.3, 0.5, 0.7, 0.9)
CONTROL_OOF_R2 = {
    "competition_proxy_inclusive": 0.9796213695740751,
    "scientific_proxy_removed": 0.9275015979762417,
}
MINIMUM_MATERIAL_IMPROVEMENT = 0.0002


def _number_label(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def ridge_name(alpha: float) -> str:
    return f"ridge_a{_number_label(alpha)}"


def elastic_name(alpha: float, l1_ratio: float) -> str:
    return f"elastic_a{_number_label(alpha)}_l1_{_number_label(l1_ratio)}"


def candidate_grid() -> list[dict[str, float | str]]:
    ridge = [
        {"family": "ridge", "model": ridge_name(alpha), "alpha": alpha, "l1_ratio": np.nan}
        for alpha in RIDGE_ALPHAS
    ]
    elastic = [
        {
            "family": "elastic_net",
            "model": elastic_name(alpha, l1_ratio),
            "alpha": alpha,
            "l1_ratio": l1_ratio,
        }
        for alpha in ELASTIC_ALPHAS
        for l1_ratio in L1_RATIOS
    ]
    return [*ridge, *elastic]


def make_candidate(item: dict[str, float | str], seed: int):
    if item["family"] == "ridge":
        return Ridge(alpha=float(item["alpha"]))
    if item["family"] == "elastic_net":
        return ElasticNet(
            alpha=float(item["alpha"]),
            l1_ratio=float(item["l1_ratio"]),
            max_iter=20000,
            random_state=seed,
        )
    raise ValueError(f"未知线性候选族：{item['family']}")


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


def run_task3_linear_tuning(root: Path, overwrite: bool = False) -> list[Path]:
    data_candidates = [root / "data" / "raw" / "A题数据集.csv", root / "A题数据集.csv"]
    data_path = next((path for path in data_candidates if path.exists()), None)
    if data_path is None:
        raise FileNotFoundError("未找到复赛数据集")
    split_path = root / "data" / "splits" / "split_assignments.csv"
    config_path = root / "configs" / "task3_linear_tuning.toml"
    frame = pd.read_csv(data_path)
    assignments = pd.read_csv(split_path)
    merged = frame.merge(assignments, on="Person_ID", validate="one_to_one")
    development = merged.loc[merged["split"] == "development"].copy().reset_index(drop=True)
    if len(development) != 8000 or set(development["cv_fold"].unique()) != {0, 1, 2, 3, 4}:
        raise ValueError("任务三开发集划分不符合冻结协议")

    contracts = feature_contracts(frame.columns.tolist())
    grid = candidate_grid()
    y = pd.to_numeric(development[TARGET]).reset_index(drop=True)
    fold_assignments = development["cv_fold"].astype(int).to_numpy()
    fold_rows: list[dict[str, object]] = []
    oof = pd.DataFrame({"Person_ID": development["Person_ID"], "true_value": y})
    started_at = datetime.now().astimezone()
    started = time.perf_counter()

    for contract_name, source_features in contracts.items():
        X = engineer_cyclic_times(development[source_features].reset_index(drop=True))
        engineered_width = int(X.shape[1])
        expected_width = len(source_features) + sum(field in source_features for field in TIME_FIELDS)
        if engineered_width != expected_width:
            raise ValueError("任务三周期时间特征数量与预期不一致")
        for item in grid:
            oof[f"pred_{contract_name}_{item['model']}"] = np.nan
        for fold in range(5):
            train_positions = np.flatnonzero(fold_assignments != fold)
            valid_positions = np.flatnonzero(fold_assignments == fold)
            X_train, X_valid = X.iloc[train_positions], X.iloc[valid_positions]
            y_train, y_valid = y.iloc[train_positions], y.iloc[valid_positions]
            print(f"[{contract_name} linear fold {fold + 1}/5] preparing development split...")
            preprocessor = make_preprocessor(X_train)
            train_matrix = preprocessor.fit_transform(X_train)
            valid_matrix = preprocessor.transform(X_valid)
            transformed_width = int(train_matrix.shape[1])
            fold_best_name = ""
            fold_best_r2 = -np.inf
            for item in grid:
                model_name = str(item["model"])
                estimator = make_candidate(item, SEED + fold)
                fit_started = time.perf_counter()
                estimator.fit(train_matrix, y_train)
                prediction = np.asarray(estimator.predict(valid_matrix)).reshape(-1)
                metric = _metrics(y_valid, prediction)
                fold_rows.append({
                    "contract": contract_name,
                    "fold": fold,
                    "family": item["family"],
                    "model": model_name,
                    "alpha": item["alpha"],
                    "l1_ratio": item["l1_ratio"],
                    "source_feature_count": len(source_features),
                    "engineered_feature_count": engineered_width,
                    "transformed_width": transformed_width,
                    **metric,
                    "fit_seconds": time.perf_counter() - fit_started,
                })
                oof.loc[valid_positions, f"pred_{contract_name}_{model_name}"] = prediction
                if metric["r2"] > fold_best_r2:
                    fold_best_r2 = metric["r2"]
                    fold_best_name = model_name
            print(f"  best={fold_best_name}: R2={fold_best_r2:.6f}")

    fold_metrics = pd.DataFrame(fold_rows)
    summary_rows: list[dict[str, object]] = []
    for contract_name, source_features in contracts.items():
        engineered_width = len(source_features) + sum(field in source_features for field in TIME_FIELDS)
        for item in grid:
            model_name = str(item["model"])
            subset = fold_metrics.loc[
                (fold_metrics["contract"] == contract_name)
                & (fold_metrics["model"] == model_name)
            ]
            prediction = oof[f"pred_{contract_name}_{model_name}"].to_numpy()
            if np.isnan(prediction).any():
                raise ValueError(f"{contract_name}/{model_name} 的 OOF 预测不完整")
            pooled = _metrics(y, prediction)
            summary_rows.append({
                "contract": contract_name,
                "family": item["family"],
                "model": model_name,
                "alpha": item["alpha"],
                "l1_ratio": item["l1_ratio"],
                "source_feature_count": len(source_features),
                "engineered_feature_count": engineered_width,
                "transformed_width_max": int(subset["transformed_width"].max()),
                "fold_r2_mean": float(subset["r2"].mean()),
                "fold_r2_std": float(subset["r2"].std()),
                "oof_r2": pooled["r2"],
                "oof_mae": pooled["mae"],
                "oof_rmse": pooled["rmse"],
                "control_oof_r2": CONTROL_OOF_R2[contract_name],
                "delta_vs_fixed_elastic_r2": pooled["r2"] - CONTROL_OOF_R2[contract_name],
                "total_fit_seconds": float(subset["fit_seconds"].sum()),
            })
    summary = pd.DataFrame(summary_rows).sort_values(
        ["contract", "oof_r2", "fold_r2_std"], ascending=[True, False, True]
    )
    winners = summary.groupby("contract", sort=False).head(1).copy()
    winners["material_improvement"] = (
        winners["delta_vs_fixed_elastic_r2"] >= MINIMUM_MATERIAL_IMPROVEMENT
    )

    output_root = root / "outputs"
    for directory in ("tables", "predictions", "logs", "logs/history"):
        (output_root / directory).mkdir(parents=True, exist_ok=True)
    prefix = "task3_health_linear_tuning"
    paths = [
        output_root / "tables" / f"{prefix}_fold_metrics.csv",
        output_root / "tables" / f"{prefix}_summary_metrics.csv",
        output_root / "tables" / f"{prefix}_winners.csv",
        output_root / "predictions" / f"{prefix}_oof_predictions.csv",
        output_root / "logs" / f"{prefix}_manifest.json",
    ]
    if not overwrite and any(path.exists() for path in paths):
        raise FileExistsError("任务三线性调参输出已存在；确认重跑时请使用 --overwrite")

    identity = hashlib.sha256(
        f"task3|linear_tuning|{_sha256(data_path)}|{_sha256(split_path)}|{_sha256(config_path)}".encode()
    ).hexdigest()[:8]
    run_id = f"{started_at.strftime('%Y%m%dT%H%M%S%z')}_task3_linear_tuning_{identity}"
    best_by_contract = {
        row["contract"]: {
            "family": row["family"],
            "model": row["model"],
            "alpha": float(row["alpha"]),
            "l1_ratio": None if pd.isna(row["l1_ratio"]) else float(row["l1_ratio"]),
            "oof_r2": float(row["oof_r2"]),
            "delta_vs_fixed_elastic_r2": float(row["delta_vs_fixed_elastic_r2"]),
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
        "experiment_role": "bounded_hyperparameter_tuning",
        "command": "python scripts/run_task3_linear_tuning.py",
        "working_directory": str(root),
        "git": _git_state(root),
        "feature_contracts": {name: len(features) for name, features in contracts.items()},
        "time_encoding": "sin_cos_1440_minutes",
        "grid": {
            "ridge_alpha": list(RIDGE_ALPHAS),
            "elastic_alpha": list(ELASTIC_ALPHAS),
            "elastic_l1_ratio": list(L1_RATIOS),
            "candidate_count": len(grid),
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
