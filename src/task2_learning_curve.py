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

from src.task2_baseline import SEED, _metrics, _sha256, make_preprocessor
from src.task2_elastic_tuning import _git_state, make_candidate
from src.task2_protocol import TARGET, feature_contracts
from src.task2_time_encoding import TIME_FIELDS, engineer_cyclic_times


CONTRACT = "scientific_proxy_removed"
TRAINING_FRACTIONS = (0.2, 0.4, 0.6, 0.8, 1.0)
SUBSAMPLE_SEEDS = (2026, 2037, 2048, 2059, 2070)
FROZEN_ALPHA = 0.01
FROZEN_L1_RATIO = 0.9
FROZEN_MODEL = "elastic_a0p01_l1_0p9"
CONTROL_ADJUSTED_R2 = 0.6201624354997604
CONTROL_R2 = 0.6230115733757466
MAXIMUM_PLATEAU_GAIN = 0.005
MAXIMUM_FULL_GENERALIZATION_GAP = 0.03


def fraction_label(fraction: float) -> str:
    return f"{int(round(100 * fraction)):03d}pct"


def nested_training_positions(
    train_positions: np.ndarray,
    fractions: tuple[float, ...],
    seed: int,
) -> dict[float, np.ndarray]:
    """Return deterministic, nested, no-replacement training subsets."""
    positions = np.asarray(train_positions, dtype=int)
    if positions.ndim != 1 or len(positions) == 0:
        raise ValueError("训练位置必须是一维非空数组")
    if tuple(sorted(set(fractions))) != fractions:
        raise ValueError("训练比例必须严格递增且不重复")
    if fractions[0] <= 0 or fractions[-1] != 1.0:
        raise ValueError("训练比例必须位于 (0, 1] 且以 1.0 结束")
    permutation = np.random.default_rng(seed).permutation(positions)
    subsets: dict[float, np.ndarray] = {}
    for fraction in fractions:
        size = len(positions) if fraction == 1.0 else max(1, int(round(len(positions) * fraction)))
        subsets[fraction] = permutation[:size].copy()
    return subsets


def diagnose_plateau(
    gain_80_to_100: float,
    full_train_validation_r2_gap: float,
) -> bool:
    return bool(
        gain_80_to_100 <= MAXIMUM_PLATEAU_GAIN
        and full_train_validation_r2_gap <= MAXIMUM_FULL_GENERALIZATION_GAP
    )


def run_learning_curve(root: Path, overwrite: bool = False) -> list[Path]:
    data_candidates = [root / "data" / "raw" / "A题数据集.csv", root / "A题数据集.csv"]
    data_path = next((path for path in data_candidates if path.exists()), None)
    if data_path is None:
        raise FileNotFoundError("未找到复赛数据集")
    split_path = root / "data" / "splits" / "split_assignments.csv"
    config_path = root / "configs" / "task2_learning_curve.toml"

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

    fold_rows: list[dict[str, object]] = []
    oof = pd.DataFrame({"Person_ID": development["Person_ID"], "true_value": y})
    for fraction in TRAINING_FRACTIONS:
        for repeat_index, seed in enumerate(SUBSAMPLE_SEEDS):
            oof[f"pred_{fraction_label(fraction)}_seed_{seed}"] = np.nan

    started_at = datetime.now().astimezone()
    started = time.perf_counter()
    for fold in range(5):
        outer_train = np.flatnonzero(folds != fold)
        valid_positions = np.flatnonzero(folds == fold)
        X_valid = X.iloc[valid_positions]
        y_valid = y.iloc[valid_positions]
        for repeat_index, seed in enumerate(SUBSAMPLE_SEEDS):
            subsets = nested_training_positions(
                outer_train,
                TRAINING_FRACTIONS,
                seed=seed + 1009 * fold,
            )
            for fraction, train_positions in subsets.items():
                X_train = X.iloc[train_positions]
                y_train = y.iloc[train_positions]
                preprocessor = make_preprocessor(X_train)
                transform_started = time.perf_counter()
                train_matrix = preprocessor.fit_transform(X_train)
                valid_matrix = preprocessor.transform(X_valid)
                transform_seconds = time.perf_counter() - transform_started
                transformed_width = int(train_matrix.shape[1])

                estimator = make_candidate(FROZEN_ALPHA, FROZEN_L1_RATIO, SEED)
                fit_started = time.perf_counter()
                estimator.fit(train_matrix, y_train)
                train_prediction = np.asarray(estimator.predict(train_matrix)).reshape(-1)
                valid_prediction = np.asarray(estimator.predict(valid_matrix)).reshape(-1)
                fit_seconds = time.perf_counter() - fit_started
                train_metric = _metrics(
                    y_train,
                    train_prediction,
                    raw_p=engineered_width,
                    transformed_p=transformed_width,
                )
                valid_metric = _metrics(
                    y_valid,
                    valid_prediction,
                    raw_p=engineered_width,
                    transformed_p=transformed_width,
                )
                fold_rows.append({
                    "fold": fold,
                    "repeat": repeat_index,
                    "subsample_seed": seed,
                    "training_fraction": fraction,
                    "training_rows": len(train_positions),
                    "validation_rows": len(valid_positions),
                    "engineered_feature_count": engineered_width,
                    "transformed_width": transformed_width,
                    "train_r2": train_metric["r2"],
                    "train_adjusted_r2_raw": train_metric["adjusted_r2_raw"],
                    "validation_r2": valid_metric["r2"],
                    "validation_adjusted_r2_raw": valid_metric["adjusted_r2_raw"],
                    "validation_mae": valid_metric["mae"],
                    "validation_rmse": valid_metric["rmse"],
                    "generalization_gap_r2": train_metric["r2"] - valid_metric["r2"],
                    "transform_seconds": transform_seconds,
                    "fit_seconds": fit_seconds,
                })
                column = f"pred_{fraction_label(fraction)}_seed_{seed}"
                oof.loc[valid_positions, column] = valid_prediction
        print(f"[learning curve fold {fold + 1}/5] completed all fractions and seeds")

    fold_metrics = pd.DataFrame(fold_rows)
    repeat_rows: list[dict[str, object]] = []
    for fraction in TRAINING_FRACTIONS:
        for repeat_index, seed in enumerate(SUBSAMPLE_SEEDS):
            column = f"pred_{fraction_label(fraction)}_seed_{seed}"
            prediction = oof[column].to_numpy()
            if np.isnan(prediction).any():
                raise ValueError(f"{column} 的 OOF 预测不完整")
            subset = fold_metrics.loc[
                (fold_metrics["training_fraction"] == fraction)
                & (fold_metrics["repeat"] == repeat_index)
            ]
            pooled = _metrics(
                y,
                prediction,
                raw_p=engineered_width,
                transformed_p=int(subset["transformed_width"].max()),
            )
            repeat_rows.append({
                "repeat": repeat_index,
                "subsample_seed": seed,
                "training_fraction": fraction,
                "training_rows_per_fold": int(subset["training_rows"].min()),
                "mean_train_r2": float(subset["train_r2"].mean()),
                "mean_fold_validation_r2": float(subset["validation_r2"].mean()),
                "mean_generalization_gap_r2": float(subset["generalization_gap_r2"].mean()),
                "oof_adjusted_r2_raw": pooled["adjusted_r2_raw"],
                "oof_r2": pooled["r2"],
                "oof_mae": pooled["mae"],
                "oof_rmse": pooled["rmse"],
            })
    repeat_metrics = pd.DataFrame(repeat_rows)

    aggregate_rows: list[dict[str, object]] = []
    previous_r2: float | None = None
    for fraction in TRAINING_FRACTIONS:
        subset = repeat_metrics.loc[repeat_metrics["training_fraction"] == fraction]
        mean_r2 = float(subset["oof_r2"].mean())
        aggregate_rows.append({
            "training_fraction": fraction,
            "training_rows_per_fold": int(subset["training_rows_per_fold"].min()),
            "repeat_count": len(subset),
            "mean_train_r2": float(subset["mean_train_r2"].mean()),
            "mean_oof_adjusted_r2_raw": float(subset["oof_adjusted_r2_raw"].mean()),
            "std_oof_adjusted_r2_raw": float(subset["oof_adjusted_r2_raw"].std(ddof=0)),
            "mean_oof_r2": mean_r2,
            "std_oof_r2": float(subset["oof_r2"].std(ddof=0)),
            "mean_oof_mae": float(subset["oof_mae"].mean()),
            "mean_oof_rmse": float(subset["oof_rmse"].mean()),
            "mean_generalization_gap_r2": float(subset["mean_generalization_gap_r2"].mean()),
            "delta_oof_r2_vs_previous_fraction": np.nan if previous_r2 is None else mean_r2 - previous_r2,
        })
        previous_r2 = mean_r2
    aggregate = pd.DataFrame(aggregate_rows)

    row_80 = aggregate.loc[aggregate["training_fraction"] == 0.8].iloc[0]
    row_100 = aggregate.loc[aggregate["training_fraction"] == 1.0].iloc[0]
    gain_80_to_100 = float(row_100["mean_oof_r2"] - row_80["mean_oof_r2"])
    full_gap = float(row_100["mean_generalization_gap_r2"])
    plateau_supported = diagnose_plateau(gain_80_to_100, full_gap)

    output_root = root / "outputs"
    for directory in ("tables", "predictions", "logs", "logs/history"):
        (output_root / directory).mkdir(parents=True, exist_ok=True)
    prefix = "task2_cyclic_elastic_learning_curve"
    paths = [
        output_root / "tables" / f"{prefix}_fold_metrics.csv",
        output_root / "tables" / f"{prefix}_repeat_metrics.csv",
        output_root / "tables" / f"{prefix}_aggregate.csv",
        output_root / "predictions" / f"{prefix}_oof_predictions.csv",
        output_root / "logs" / f"{prefix}_manifest.json",
    ]
    if not overwrite and any(path.exists() for path in paths):
        raise FileExistsError("任务二学习曲线输出已存在；确认重跑时请使用 --overwrite")

    identity = hashlib.sha256(
        f"task2|learning_curve|{_sha256(data_path)}|{_sha256(split_path)}|{_sha256(config_path)}".encode()
    ).hexdigest()[:8]
    run_id = f"{started_at.strftime('%Y%m%dT%H%M%S%z')}_task2_learning_curve_{identity}"
    manifest = {
        "run_id": run_id,
        "status": "completed",
        "task": "task2",
        "target": TARGET,
        "experiment": prefix,
        "experiment_role": "diagnostic_learning_curve",
        "command": "python scripts/run_task2_learning_curve.py",
        "working_directory": str(root),
        "git": _git_state(root),
        "feature_contract": CONTRACT,
        "source_feature_count": len(source_features),
        "engineered_feature_count": engineered_width,
        "frozen_candidate": {
            "model": FROZEN_MODEL,
            "alpha": FROZEN_ALPHA,
            "l1_ratio": FROZEN_L1_RATIO,
            "control_oof_adjusted_r2_raw": CONTROL_ADJUSTED_R2,
            "control_oof_r2": CONTROL_R2,
        },
        "training_fractions": list(TRAINING_FRACTIONS),
        "subsample_seeds": list(SUBSAMPLE_SEEDS),
        "nested_subsamples": True,
        "selection_or_tuning_allowed": False,
        "diagnosis": {
            "gain_oof_r2_80_to_100": gain_80_to_100,
            "full_train_validation_r2_gap": full_gap,
            "maximum_plateau_gain_80_to_100": MAXIMUM_PLATEAU_GAIN,
            "maximum_full_train_validation_r2_gap": MAXIMUM_FULL_GENERALIZATION_GAP,
            "practical_plateau_supported": plateau_supported,
        },
        "data_sha256": _sha256(data_path),
        "split_sha256": _sha256(split_path),
        "config_sha256": _sha256(config_path),
        "development_rows": 8000,
        "cv_folds": 5,
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
    repeat_metrics.to_csv(paths[1], index=False, encoding="utf-8-sig")
    aggregate.to_csv(paths[2], index=False, encoding="utf-8-sig")
    oof.to_csv(paths[3], index=False, encoding="utf-8-sig")
    manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2)
    paths[4].write_text(manifest_text, encoding="utf-8")
    history = output_root / "logs" / "history" / f"{run_id}_manifest.json"
    history.write_text(manifest_text, encoding="utf-8")
    return [*paths, history]
