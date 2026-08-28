from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.task1_proxy_sensitivity import PROXY_FIELDS, feature_contracts
from src.task1_regression_baseline import TARGET
from src.task1_spline_tuning import make_preprocessor


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


def run_validation(root: Path, overwrite: bool = False) -> list[Path]:
    candidates = [root / "data" / "raw" / "A题数据集.csv", root / "A题数据集.csv"]
    data_path = next((path for path in candidates if path.exists()), None)
    if data_path is None:
        raise FileNotFoundError("未找到复赛数据集")
    split_path = root / "data" / "splits" / "split_assignments.csv"
    config_path = root / "configs" / "task1_spline_proxy_validation.toml"
    frame = pd.read_csv(data_path)
    assignments = pd.read_csv(split_path)
    merged = frame.merge(assignments, on="Person_ID", validate="one_to_one")
    development = merged.loc[merged["split"] == "development"].copy()
    contracts = feature_contracts(list(frame.columns))
    y = pd.to_numeric(development[TARGET])
    fold_rows: list[dict[str, object]] = []
    oof = pd.DataFrame({"Person_ID": development["Person_ID"].to_numpy(), "true_value": y.to_numpy()})
    started_at = datetime.now().astimezone()
    total_started = time.perf_counter()

    for contract_name, features in contracts.items():
        X = development[features]
        for fold in range(5):
            train_mask = development["cv_fold"] != fold
            valid_mask = development["cv_fold"] == fold
            started = time.perf_counter()
            preprocessor = make_preprocessor(X.loc[train_mask], n_knots=4, degree=2)
            train_matrix = preprocessor.fit_transform(X.loc[train_mask])
            valid_matrix = preprocessor.transform(X.loc[valid_mask])
            model = Ridge(alpha=1.0)
            model.fit(train_matrix, y.loc[train_mask])
            prediction = model.predict(valid_matrix)
            metric = _metrics(y.loc[valid_mask], prediction)
            print(f"[{contract_name} fold {fold + 1}/5] R2={metric['r2']:.4f}, MAE={metric['mae']:.4f}")
            fold_rows.append({
                "contract": contract_name,
                "fold": fold,
                "feature_count": len(features),
                "transformed_width": int(train_matrix.shape[1]),
                **metric,
                "fit_seconds": time.perf_counter() - started,
            })
            valid_ids = development.loc[valid_mask, "Person_ID"]
            mapping = dict(zip(valid_ids, prediction))
            mask = oof["Person_ID"].isin(mapping)
            oof.loc[mask, f"pred_{contract_name}"] = oof.loc[mask, "Person_ID"].map(mapping)

    fold_metrics = pd.DataFrame(fold_rows)
    summary_rows = []
    for contract_name, features in contracts.items():
        subset = fold_metrics.loc[fold_metrics["contract"] == contract_name]
        prediction = oof[f"pred_{contract_name}"].to_numpy()
        if np.isnan(prediction).any():
            raise ValueError(f"{contract_name} 的 OOF 预测不完整")
        pooled = _metrics(oof["true_value"], prediction)
        summary_rows.append({
            "contract": contract_name,
            "feature_count": len(features),
            "transformed_width": int(subset["transformed_width"].iloc[0]),
            "fold_r2_mean": float(subset["r2"].mean()),
            "fold_r2_std": float(subset["r2"].std()),
            "oof_r2": pooled["r2"],
            "oof_mae": pooled["mae"],
            "oof_rmse": pooled["rmse"],
        })
    summary = pd.DataFrame(summary_rows).sort_values("oof_r2", ascending=False)
    full_r2 = float(summary.loc[summary["contract"] == "full_non_sleep", "oof_r2"].iloc[0])
    strict_r2 = float(summary.loc[summary["contract"] == "proxy_removed", "oof_r2"].iloc[0])
    summary["proxy_r2_gap"] = full_r2 - strict_r2

    output_root = root / "outputs"
    for directory in ("tables", "predictions", "logs", "logs/history"):
        (output_root / directory).mkdir(parents=True, exist_ok=True)
    prefix = "task1_spline_proxy_validation"
    paths = [
        output_root / "tables" / f"{prefix}_fold_metrics.csv",
        output_root / "tables" / f"{prefix}_summary_metrics.csv",
        output_root / "predictions" / f"{prefix}_oof_predictions.csv",
        output_root / "logs" / f"{prefix}_manifest.json",
    ]
    if not overwrite and any(path.exists() for path in paths):
        raise FileExistsError("样条代理验证输出已存在；确认重跑时请使用 --overwrite")
    identity = hashlib.sha256(
        f"task1|spline_proxy|{_sha256(data_path)}|{_sha256(split_path)}|{_sha256(config_path)}".encode()
    ).hexdigest()[:8]
    run_id = f"{started_at.strftime('%Y%m%dT%H%M%S%z')}_task1_spline_proxy_{identity}"
    manifest = {
        "run_id": run_id,
        "status": "completed",
        "task": "task1",
        "target": TARGET,
        "experiment": "spline_proxy_validation",
        "experiment_role": "scientific_comparison",
        "model": {"n_knots": 4, "degree": 2, "alpha": 1.0},
        "contracts": contracts,
        "proxy_fields": sorted(PROXY_FIELDS),
        "duration_seconds": time.perf_counter() - total_started,
        "full_oof_r2": full_r2,
        "proxy_removed_oof_r2": strict_r2,
        "proxy_r2_gap": full_r2 - strict_r2,
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
