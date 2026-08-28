from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


TARGETS = ["Sleep_Quality_Score", "Productivity_Score", "Health_Score"]
ID_COLUMN = "Person_ID"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_audit(root: Path) -> list[Path]:
    candidates = [root / "data" / "raw" / "A题数据集.csv", root / "A题数据集.csv"]
    data_path = next((path for path in candidates if path.exists()), None)
    if data_path is None:
        raise FileNotFoundError(
            "未找到数据集，请将 A题数据集.csv 放入 fusai\\data\\raw 或 fusai 根目录"
        )
    frame = pd.read_csv(data_path)
    tables = root / "outputs" / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    overview = pd.DataFrame([{
        "file": data_path.name,
        "rows": len(frame),
        "columns": len(frame.columns),
        "duplicate_rows": int(frame.duplicated().sum()),
        "duplicate_person_id": int(frame[ID_COLUMN].duplicated().sum()),
        "sha256": sha256(data_path),
    }])
    profile = pd.DataFrame({
        "column": frame.columns,
        "dtype": [str(frame[c].dtype) for c in frame.columns],
        "missing_count": [int(frame[c].isna().sum()) for c in frame.columns],
        "missing_rate": [float(frame[c].isna().mean()) for c in frame.columns],
        "unique_count": [int(frame[c].nunique(dropna=False)) for c in frame.columns],
    })
    target_rows = []
    for target in TARGETS:
        series = pd.to_numeric(frame[target], errors="coerce")
        target_rows.append({
            "target": target,
            "dtype": str(frame[target].dtype),
            "missing_count": int(series.isna().sum()),
            "unique_count": int(series.nunique(dropna=False)),
            "min": float(series.min()),
            "max": float(series.max()),
            "mean": float(series.mean()),
            "std": float(series.std()),
        })
    targets = pd.DataFrame(target_rows)
    numeric = frame.select_dtypes(include="number")
    corr = numeric.corr(numeric_only=True)
    pairs = []
    for i, left in enumerate(corr.columns):
        for right in corr.columns[i + 1:]:
            value = corr.loc[left, right]
            if pd.notna(value):
                pairs.append({"left": left, "right": right, "correlation": float(value), "abs_correlation": abs(float(value))})
    correlations = pd.DataFrame(pairs).sort_values("abs_correlation", ascending=False) if pairs else pd.DataFrame()

    outputs = {
        "data_overview.csv": overview,
        "column_profile.csv": profile,
        "target_profile.csv": targets,
        "numeric_correlations.csv": correlations,
    }
    written = []
    for name, table in outputs.items():
        path = tables / name
        table.to_csv(path, index=False, encoding="utf-8-sig")
        written.append(path)
    manifest = {
        "data_path": data_path.name,
        "data_sha256": overview.iloc[0]["sha256"],
        "rows": len(frame),
        "columns": len(frame.columns),
        "targets": TARGETS,
        "audit_status": "completed",
    }
    log_path = root / "outputs" / "logs" / "fusai_data_audit_manifest.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    written.append(log_path)
    return written
