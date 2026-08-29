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


TARGET = "Health_Score"
ID_COLUMN = "Person_ID"
DETERMINISTIC_TARGET_PROXIES = {"Fitness_Level", "Wellness_Category"}
STRONG_COMPOSITE_PROXIES = {"Healthy_Aging_Score"}
FORBIDDEN = {ID_COLUMN, TARGET} | DETERMINISTIC_TARGET_PROXIES
LABEL_ORDER = ("Poor", "Average", "Good", "Excellent")


def feature_contracts(columns: list[str]) -> dict[str, list[str]]:
    proxy_inclusive = [column for column in columns if column not in FORBIDDEN]
    proxy_removed = [column for column in proxy_inclusive if column not in STRONG_COMPOSITE_PROXIES]
    return {
        "competition_proxy_inclusive": proxy_inclusive,
        "scientific_proxy_removed": proxy_removed,
    }


def health_score_to_category(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values)
    mapped = pd.cut(
        numeric,
        bins=[-np.inf, 45.0, 65.0, 80.0, np.inf],
        labels=list(LABEL_ORDER),
        right=False,
    )
    return mapped.astype("string")


def validate_deterministic_target_proxies(frame: pd.DataFrame) -> dict[str, bool]:
    required = {TARGET, *DETERMINISTIC_TARGET_PROXIES}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"任务三代理审计缺少字段: {sorted(missing)}")
    expected = health_score_to_category(frame[TARGET])
    return {
        "fitness_matches_health_score_bins": bool(
            frame["Fitness_Level"].astype("string").reset_index(drop=True).equals(expected.reset_index(drop=True))
        ),
        "wellness_matches_health_score_bins": bool(
            frame["Wellness_Category"].astype("string").reset_index(drop=True).equals(expected.reset_index(drop=True))
        ),
        "fitness_equals_wellness": bool(
            frame["Fitness_Level"].astype("string").reset_index(drop=True).equals(
                frame["Wellness_Category"].astype("string").reset_index(drop=True)
            )
        ),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _correlation_ratio(categories: pd.Series, values: pd.Series) -> float:
    valid = categories.notna() & values.notna()
    categories = categories.loc[valid].astype(str)
    values = values.loc[valid].astype(float)
    if values.empty:
        return float("nan")
    grand_mean = float(values.mean())
    numerator = sum(
        len(group) * (float(group.mean()) - grand_mean) ** 2
        for _, group in values.groupby(categories)
    )
    denominator = float(((values - grand_mean) ** 2).sum())
    return float(numerator / denominator) if denominator > 0 else 0.0


def _numeric_linear_association(feature: pd.Series, target: pd.Series) -> tuple[float, float]:
    valid = feature.notna() & target.notna()
    x = feature.loc[valid].astype(float).to_numpy()
    y = target.loc[valid].astype(float).to_numpy()
    if len(x) < 2 or np.std(x) == 0:
        return float("nan"), 0.0
    correlation = float(np.corrcoef(x, y)[0, 1])
    design = np.column_stack([np.ones(len(x)), x])
    prediction = design @ np.linalg.lstsq(design, y, rcond=None)[0]
    denominator = float(np.sum((y - y.mean()) ** 2))
    linear_r2 = float(1.0 - np.sum((y - prediction) ** 2) / denominator)
    return correlation, linear_r2


def _univariate_associations(development: pd.DataFrame) -> pd.DataFrame:
    y = pd.to_numeric(development[TARGET])
    rows: list[dict[str, object]] = []
    for feature in development.columns:
        if feature in {ID_COLUMN, TARGET, "split", "cv_fold"}:
            continue
        series = development[feature]
        if pd.api.types.is_numeric_dtype(series):
            correlation, strength = _numeric_linear_association(series, y)
            association_type = "univariate_linear_r2"
        else:
            correlation = float("nan")
            strength = _correlation_ratio(series, y)
            association_type = "eta_squared"
        rows.append({
            "feature": feature,
            "dtype": str(series.dtype),
            "role": (
                "deterministic_target_binning"
                if feature in DETERMINISTIC_TARGET_PROXIES
                else "strong_composite_proxy"
                if feature in STRONG_COMPOSITE_PROXIES
                else "candidate_predictor"
            ),
            "pearson_correlation": correlation,
            "absolute_correlation": abs(correlation) if np.isfinite(correlation) else float("nan"),
            "association_type": association_type,
            "association_strength": strength,
            "allowed_in_proxy_inclusive": feature not in FORBIDDEN,
            "allowed_in_proxy_removed": feature not in FORBIDDEN | STRONG_COMPOSITE_PROXIES,
        })
    return pd.DataFrame(rows).sort_values(
        ["association_strength", "feature"], ascending=[False, True]
    ).reset_index(drop=True)


def _proxy_category_ranges(development: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for feature in sorted(DETERMINISTIC_TARGET_PROXIES):
        grouped = development.groupby(feature, dropna=False)[TARGET].agg(
            sample_count="size", target_min="min", target_max="max", target_mean="mean"
        )
        for label, row in grouped.reset_index().iterrows():
            rows.append({
                "feature": feature,
                "category": row[feature],
                "sample_count": int(row["sample_count"]),
                "target_min": float(row["target_min"]),
                "target_max": float(row["target_max"]),
                "target_mean": float(row["target_mean"]),
            })
    return pd.DataFrame(rows).sort_values(["feature", "target_min"]).reset_index(drop=True)


def audit_task3_protocol(root: Path, overwrite: bool = False) -> list[Path]:
    data_candidates = [root / "data" / "raw" / "A题数据集.csv", root / "A题数据集.csv"]
    data_path = next((path for path in data_candidates if path.exists()), None)
    if data_path is None:
        raise FileNotFoundError("未找到复赛数据集")
    split_path = root / "data" / "splits" / "split_assignments.csv"
    config_path = root / "configs" / "task3_protocol.toml"
    frame = pd.read_csv(data_path)
    assignments = pd.read_csv(split_path)
    merged = frame.merge(assignments, on=ID_COLUMN, validate="one_to_one")
    development = merged.loc[merged["split"] == "development"].copy().reset_index(drop=True)
    if len(development) != 8000 or set(development["cv_fold"].unique()) != {0, 1, 2, 3, 4}:
        raise ValueError("任务三开发集划分不符合冻结协议")
    relationships = validate_deterministic_target_proxies(development)
    if not all(relationships.values()):
        raise ValueError(f"任务三确定性代理关系发生变化: {relationships}")
    contracts = feature_contracts(frame.columns.tolist())
    if set(contracts["competition_proxy_inclusive"]) & FORBIDDEN:
        raise ValueError("任务三竞赛契约包含目标、ID或确定性目标代理")

    contract_rows: list[dict[str, object]] = []
    for contract, features in contracts.items():
        for feature in features:
            contract_rows.append({
                "contract": contract,
                "feature": feature,
                "dtype": str(frame[feature].dtype),
                "is_strong_composite_proxy": feature in STRONG_COMPOSITE_PROXIES,
                "prediction_time_available": True,
            })
    contract_table = pd.DataFrame(contract_rows)
    associations = _univariate_associations(development)
    ranges = _proxy_category_ranges(development)
    healthy_row = associations.loc[associations["feature"] == "Healthy_Aging_Score"].iloc[0]

    output_root = root / "outputs"
    for directory in ("tables", "logs", "logs/history"):
        (output_root / directory).mkdir(parents=True, exist_ok=True)
    paths = [
        output_root / "tables" / "task3_feature_contract.csv",
        output_root / "tables" / "task3_univariate_associations.csv",
        output_root / "tables" / "task3_proxy_category_ranges.csv",
        output_root / "logs" / "task3_protocol_audit_manifest.json",
    ]
    if not overwrite and any(path.exists() for path in paths):
        raise FileExistsError("任务三协议审计输出已存在；确认重建时请使用 --overwrite")

    started_at = datetime.now().astimezone()
    started = time.perf_counter()
    identity = hashlib.sha256(
        f"task3|protocol|{_sha256(data_path)}|{_sha256(split_path)}|{_sha256(config_path)}".encode()
    ).hexdigest()[:8]
    run_id = f"{started_at.strftime('%Y%m%dT%H%M%S%z')}_task3_protocol_{identity}"
    manifest = {
        "run_id": run_id,
        "status": "completed",
        "task": "task3",
        "target": TARGET,
        "experiment": "task3_protocol_audit",
        "experiment_role": "diagnostic",
        "command": "python scripts/audit_task3_protocol.py",
        "working_directory": str(root),
        "git": _git_state(root),
        "feature_contracts": {name: len(features) for name, features in contracts.items()},
        "deterministic_target_proxies": sorted(DETERMINISTIC_TARGET_PROXIES),
        "strong_composite_proxies": sorted(STRONG_COMPOSITE_PROXIES),
        "deterministic_proxy_checks": relationships,
        "healthy_aging_pearson_correlation": float(healthy_row["pearson_correlation"]),
        "healthy_aging_univariate_linear_r2": float(healthy_row["association_strength"]),
        "data_sha256": _sha256(data_path),
        "split_sha256": _sha256(split_path),
        "config_sha256": _sha256(config_path),
        "development_rows": 8000,
        "sealed_holdout_rows": 2000,
        "cv_folds": 5,
        "seed": 2026,
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
    contract_table.to_csv(paths[0], index=False, encoding="utf-8-sig")
    associations.to_csv(paths[1], index=False, encoding="utf-8-sig")
    ranges.to_csv(paths[2], index=False, encoding="utf-8-sig")
    manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2)
    paths[3].write_text(manifest_text, encoding="utf-8")
    history = output_root / "logs" / "history" / f"{run_id}_manifest.json"
    history.write_text(manifest_text, encoding="utf-8")
    return [*paths, history]
