from __future__ import annotations

from pathlib import Path
import pandas as pd

TASKS = {
    "task1": "Sleep_Quality_Score",
    "task2": "Productivity_Score",
    "task3": "Health_Score",
}

def validate_protocol(data_path: Path) -> dict[str, object]:
    frame = pd.read_csv(data_path)
    if frame["Person_ID"].isna().any() or frame["Person_ID"].duplicated().any():
        raise ValueError("Person_ID 必须非空且唯一")
    missing = {task: target for task, target in TASKS.items() if target not in frame.columns}
    if missing:
        raise ValueError(f"缺少目标字段: {missing}")
    return {"rows": len(frame), "columns": len(frame.columns), "targets": TASKS}
