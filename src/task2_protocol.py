from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


TARGET = "Productivity_Score"
ID_COLUMN = "Person_ID"
FORBIDDEN = {ID_COLUMN, TARGET}
COMPOSITE_PROXIES = {
    "Health_Score",
    "Fitness_Level",
    "Healthy_Aging_Score",
    "Wellness_Category",
}
HIGH_SIGNAL_DRIVERS = {
    "Energy_Level_Score",
    "Fatigue_Level_Score",
    "Mood_Score",
    "Sleep_Quality_Score",
    "Anxiety_Score",
    "Depression_Risk_Score",
    "Life_Satisfaction_Score",
    "Stress_Level",
    "Focus_Concentration_Score",
}


def adjusted_r2(r2: float, n: int, p: int) -> float:
    if n <= p + 1:
        raise ValueError(f"调整 R2 要求 n > p + 1，收到 n={n}, p={p}")
    return float(1.0 - (1.0 - float(r2)) * (n - 1) / (n - p - 1))


def feature_contracts(columns: list[str]) -> dict[str, list[str]]:
    competition = [column for column in columns if column not in FORBIDDEN]
    proxy_removed = [column for column in competition if column not in COMPOSITE_PROXIES]
    return {"competition": competition, "scientific_proxy_removed": proxy_removed}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _numeric_vif(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = frame.select_dtypes(include="number").copy()
    numeric = numeric.loc[:, numeric.nunique(dropna=True) > 1]
    numeric = numeric.fillna(numeric.median())
    standardized = (numeric - numeric.mean()) / numeric.std(ddof=0)
    correlation = standardized.corr().to_numpy(dtype=float)
    inverse = np.linalg.pinv(correlation, hermitian=True)
    values = np.clip(np.diag(inverse), 1.0, None)
    result = pd.DataFrame({"feature": numeric.columns, "vif": values})
    result["severity"] = np.select(
        [result["vif"] >= 10.0, result["vif"] >= 5.0],
        ["high", "attention"],
        default="low",
    )
    return result.sort_values(["vif", "feature"], ascending=[False, True]).reset_index(drop=True)


def _pairwise_numeric_correlations(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = frame.select_dtypes(include="number")
    matrix = numeric.corr()
    rows = []
    for left_index, left in enumerate(matrix.columns):
        for right in matrix.columns[left_index + 1:]:
            correlation = matrix.loc[left, right]
            rows.append({
                "feature_1": left,
                "feature_2": right,
                "correlation": float(correlation),
                "absolute_correlation": float(abs(correlation)),
                "above_0p8": bool(abs(correlation) >= 0.8),
            })
    return pd.DataFrame(rows).sort_values("absolute_correlation", ascending=False).reset_index(drop=True)


def _correlation_ratio(categories: pd.Series, values: pd.Series) -> float:
    valid = categories.notna() & values.notna()
    categories = categories.loc[valid].astype(str)
    values = values.loc[valid].astype(float)
    if values.empty:
        return float("nan")
    grand_mean = values.mean()
    numerator = sum(len(group) * (group.mean() - grand_mean) ** 2 for _, group in values.groupby(categories))
    denominator = float(((values - grand_mean) ** 2).sum())
    return float(numerator / denominator) if denominator > 0 else 0.0


def _proxy_audit(development: pd.DataFrame) -> pd.DataFrame:
    y = pd.to_numeric(development[TARGET])
    rows = []
    fields = sorted(COMPOSITE_PROXIES | HIGH_SIGNAL_DRIVERS)
    for field in fields:
        series = development[field]
        if pd.api.types.is_numeric_dtype(series):
            valid = series.notna() & y.notna()
            x = series.loc[valid].astype(float).to_numpy()
            target = y.loc[valid].to_numpy()
            correlation = float(np.corrcoef(x, target)[0, 1])
            design = np.column_stack([np.ones(len(x)), x])
            coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
            prediction = design @ coefficients
            denominator = np.sum((target - target.mean()) ** 2)
            association = float(1.0 - np.sum((target - prediction) ** 2) / denominator)
            association_type = "univariate_linear_r2"
        else:
            correlation = float("nan")
            association = _correlation_ratio(series, y)
            association_type = "eta_squared"
        rows.append({
            "feature": field,
            "role": "disclosed_composite_proxy" if field in COMPOSITE_PROXIES else "substantive_driver",
            "dtype": str(series.dtype),
            "pearson_correlation": correlation,
            "absolute_correlation": abs(correlation) if np.isfinite(correlation) else float("nan"),
            "association_type": association_type,
            "association_strength": association,
            "exact_target_match": bool(series.equals(development[TARGET])),
            "competition_contract": True,
            "proxy_removed_contract": field not in COMPOSITE_PROXIES,
        })
    return pd.DataFrame(rows).sort_values("association_strength", ascending=False).reset_index(drop=True)


def audit_task2_protocol(root: Path, overwrite: bool = False) -> list[Path]:
    data_candidates = [root / "data" / "raw" / "A题数据集.csv", root / "A题数据集.csv"]
    data_path = next((path for path in data_candidates if path.exists()), None)
    if data_path is None:
        raise FileNotFoundError("未找到复赛数据集")
    split_path = root / "data" / "splits" / "split_assignments.csv"
    config_path = root / "configs" / "task2_protocol.toml"
    frame = pd.read_csv(data_path)
    assignments = pd.read_csv(split_path)
    merged = frame.merge(assignments, on=ID_COLUMN, validate="one_to_one")
    development = merged.loc[merged["split"] == "development"].copy()
    if len(development) != 8000 or set(development["cv_fold"].unique()) != {0, 1, 2, 3, 4}:
        raise ValueError("任务二开发集划分不符合冻结协议")
    contracts = feature_contracts(frame.columns.tolist())
    if TARGET in contracts["competition"] or ID_COLUMN in contracts["competition"]:
        raise ValueError("任务二竞赛特征契约包含目标或ID")

    contract_rows = []
    for contract_name, features in contracts.items():
        for feature in features:
            contract_rows.append({
                "contract": contract_name,
                "feature": feature,
                "dtype": str(frame[feature].dtype),
                "is_composite_proxy": feature in COMPOSITE_PROXIES,
                "is_high_signal_driver": feature in HIGH_SIGNAL_DRIVERS,
                "prediction_time_available": True,
            })
    contract_table = pd.DataFrame(contract_rows)
    competition_numeric = development[contracts["competition"]].select_dtypes(include="number")
    vif = _numeric_vif(competition_numeric)
    pairs = _pairwise_numeric_correlations(competition_numeric)
    proxy_audit = _proxy_audit(development)

    output_root = root / "outputs"
    for directory in ("tables", "logs", "logs/history"):
        (output_root / directory).mkdir(parents=True, exist_ok=True)
    paths = [
        output_root / "tables" / "task2_feature_contract.csv",
        output_root / "tables" / "task2_numeric_collinearity.csv",
        output_root / "tables" / "task2_vif.csv",
        output_root / "tables" / "task2_proxy_audit.csv",
        output_root / "logs" / "task2_protocol_audit_manifest.json",
    ]
    if not overwrite and any(path.exists() for path in paths):
        raise FileExistsError("任务二协议审查输出已存在；确认重建时请使用 --overwrite")
    started_at = datetime.now().astimezone()
    started = time.perf_counter()
    identity = hashlib.sha256(
        f"task2|protocol|{_sha256(data_path)}|{_sha256(split_path)}|{_sha256(config_path)}".encode()
    ).hexdigest()[:8]
    run_id = f"{started_at.strftime('%Y%m%dT%H%M%S%z')}_task2_protocol_{identity}"
    manifest = {
        "run_id": run_id,
        "status": "completed",
        "task": "task2",
        "target": TARGET,
        "experiment": "task2_protocol_audit",
        "experiment_role": "diagnostic",
        "primary_metric": "adjusted_r2_raw_feature_count",
        "adjusted_r2_formula": "1 - (1 - r2) * (n - 1) / (n - p - 1)",
        "competition_feature_count": len(contracts["competition"]),
        "proxy_removed_feature_count": len(contracts["scientific_proxy_removed"]),
        "numeric_feature_count_for_vif": int(len(vif)),
        "high_vif_feature_count": int((vif["vif"] >= 10.0).sum()),
        "attention_vif_feature_count": int((vif["vif"] >= 5.0).sum()),
        "pairwise_correlation_above_0p8_count": int(pairs["above_0p8"].sum()),
        "automatic_feature_removal": False,
        "data_sha256": _sha256(data_path),
        "split_sha256": _sha256(split_path),
        "config_sha256": _sha256(config_path),
        "development_rows": 8000,
        "sealed_holdout_rows": 2000,
        "holdout_evaluated": False,
        "duration_seconds": time.perf_counter() - started,
    }
    contract_table.to_csv(paths[0], index=False, encoding="utf-8-sig")
    pairs.to_csv(paths[1], index=False, encoding="utf-8-sig")
    vif.to_csv(paths[2], index=False, encoding="utf-8-sig")
    proxy_audit.to_csv(paths[3], index=False, encoding="utf-8-sig")
    manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2)
    paths[4].write_text(manifest_text, encoding="utf-8")
    history = output_root / "logs" / "history" / f"{run_id}_manifest.json"
    history.write_text(manifest_text, encoding="utf-8")
    return [*paths, history]
