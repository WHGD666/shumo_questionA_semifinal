#!/usr/bin/env python3
"""复赛A题三任务统一推理入口。

本脚本只做推理，不做任何训练；输入含额外列时自动忽略，
缺少必需列时给出明确报错，输出保持 Person_ID 顺序不变。

用法：
    python predict.py --task task1 --input input.csv --output task1_predictions.csv
    python predict.py --task all --input input.csv --output-dir outputs
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "source"))

ID_COLUMN = "Person_ID"
MANIFEST_PATH = ROOT / "model_manifest.json"


def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        raise SystemExit(f"缺少提交包清单: {MANIFEST_PATH}")
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def predict_task(manifest: dict, task: str, input_path: Path, output_path: Path) -> Path:
    entry = manifest["tasks"][task]
    frame = pd.read_csv(input_path)
    if ID_COLUMN not in frame.columns:
        raise SystemExit(f"输入缺少标识符列 {ID_COLUMN}")
    if frame[ID_COLUMN].isna().any() or frame[ID_COLUMN].duplicated().any():
        raise SystemExit(f"{ID_COLUMN} 必须唯一且非空")
    missing = [c for c in entry["required_columns"] if c not in frame.columns]
    if missing:
        raise SystemExit(
            f"{task} 输入缺少必需列 {len(missing)} 个: {', '.join(missing)}；"
            f"其余多余列已自动忽略"
        )
    model = joblib.load(ROOT / entry["model_filename"])
    prediction = model.predict(frame)
    output = frame[[ID_COLUMN]].copy()
    output[entry["prediction_column"]] = prediction
    output.to_csv(output_path, index=False, encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="复赛A题统一推理入口（仅推理，不训练）")
    parser.add_argument("--task", required=True, choices=["task1", "task2", "task3", "all"])
    parser.add_argument("--input", required=True, help="输入CSV路径")
    parser.add_argument("--output", help="单任务输出CSV路径")
    parser.add_argument("--output-dir", help="task=all 时的输出目录，默认与输入文件同目录")
    args = parser.parse_args()

    manifest = load_manifest()
    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"输入文件不存在: {input_path}")

    tasks = ["task1", "task2", "task3"] if args.task == "all" else [args.task]
    for task in tasks:
        if args.task == "all":
            output_dir = Path(args.output_dir) if args.output_dir else input_path.resolve().parent
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{task}_predictions.csv"
        else:
            if not args.output:
                raise SystemExit("单任务模式必须提供 --output")
            output_path = Path(args.output)
        written = predict_task(manifest, task, input_path, output_path)
        print(f"{task} 预测完成: {written}")


if __name__ == "__main__":
    main()
