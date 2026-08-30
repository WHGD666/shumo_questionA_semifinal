#!/usr/bin/env python3
"""提交包完整性自检：模型哈希、字段契约与依赖版本。

用法：在提交包目录内执行  python verify_package.py
全部通过时输出 PASS 并返回退出码 0。
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "source"))

MANIFEST_PATH = ROOT / "model_manifest.json"
REQUIREMENTS_PATH = ROOT / "requirements.txt"


def check_dependencies() -> list[str]:
    problems: list[str] = []
    for line in REQUIREMENTS_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, _, expected = line.partition("==")
        try:
            installed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            problems.append(f"缺少依赖: {name}=={expected}")
            continue
        if installed != expected:
            problems.append(f"依赖版本不一致: {name} 期望 {expected}，实际 {installed}")
    return problems


def main() -> int:
    if not MANIFEST_PATH.exists():
        print("FAIL: 缺少 model_manifest.json")
        return 1
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    failures: list[str] = []
    for task, entry in manifest["tasks"].items():
        model_path = ROOT / entry["model_filename"]
        if not model_path.exists():
            failures.append(f"{task}: 模型文件缺失 {entry['model_filename']}")
            continue
        digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
        if digest != entry["model_sha256"]:
            failures.append(f"{task}: 模型哈希与清单不一致")
            continue
        try:
            import joblib

            model = joblib.load(model_path)
        except Exception as error:  # noqa: BLE001 - 自检需要报告任何加载失败
            failures.append(f"{task}: 模型加载失败: {error}")
            continue
        if list(model.required_columns_) != list(entry["required_columns"]):
            failures.append(f"{task}: 必需字段与清单不一致")
            continue
        print(f"PASS {task}: {entry['selected_model']} 必需字段 {entry['required_column_count']} 个")

    failures.extend(check_dependencies())
    for problem in failures:
        print(f"FAIL {problem}")
    if failures:
        return 1
    print("提交包完整性检查全部通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
