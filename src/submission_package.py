from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import joblib

from src.final_evaluation_models import ID_COLUMN, TARGETS, TASKS
from src.final_evaluation_protocol import sha256_file
from src.final_release import load_release_config, validate_release_gate


RELEASE_MANIFEST_PATH = Path("outputs/logs/final_release_model_training_manifest.json")
PREDICT_ENTRYPOINT_NAME = "predict.py"
VERIFY_ENTRYPOINT_NAME = "verify_package.py"
README_NAME = "README.md"
REQUIREMENTS_NAME = "requirements.txt"


def _strict_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"JSON包含非标准常量: {value}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def _copy_tree(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns("__pycache__", "*.py[cod]"),
    )


def _example_inputs(frame_root: Path, package_root: Path) -> dict[str, Path]:
    """生成20行示例输入：全列一份，外加每任务掩去自身目标的严格契约一份。

    注意：其他任务的目标字段可能是本任务模型的必需特征（契约允许），
    因此每个任务只能保证自身目标缺失，示例按此口径生成。
    """
    import pandas as pd

    frame = pd.read_csv(frame_root / "data" / "raw" / "A题数据集.csv")
    sample = frame.sort_values(ID_COLUMN).head(20)
    paths: dict[str, Path] = {}
    full_path = package_root / "example_input.csv"
    sample.to_csv(full_path, index=False, encoding="utf-8")
    paths["all"] = full_path
    for task, target in TARGETS.items():
        masked_path = package_root / f"example_input_{task}.csv"
        sample.drop(columns=[target]).to_csv(masked_path, index=False, encoding="utf-8")
        paths[task] = masked_path
    return paths


def build_submission_package(root: Path, overwrite: bool = False) -> list[Path]:
    gate = validate_release_gate(root)
    config = gate["config"]
    submission = config["submission"]
    package_root = root / submission["package_directory"]
    results_root = root / submission["results_directory"]

    release_manifest_path = root / RELEASE_MANIFEST_PATH
    if not release_manifest_path.exists():
        raise FileNotFoundError("缺少最终发布模型训练清单，请先训练最终发布模型")
    release_manifest = _strict_json(release_manifest_path)

    planned = [
        package_root / config["artifacts"][task]["filename"] for task in TASKS
    ] + [package_root / "source", package_root / "model_manifest.json", results_root]
    if not overwrite and any(path.exists() for path in planned):
        raise FileExistsError("提交包已存在；确认重建时请使用 --overwrite")

    package_root.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    manifest_tasks: dict[str, dict[str, Any]] = {}
    for task in TASKS:
        artifact = config["artifacts"][task]
        release_entry = release_manifest["tasks"][task]
        if release_entry["selected_model"] != artifact["selected_model"]:
            raise ValueError(f"{task}发布清单所选模型与冻结配置不一致")
        source_model = root / release_entry["model_path"]
        if sha256_file(source_model) != release_entry["model_sha256"]:
            raise ValueError(f"{task}发布模型哈希与训练清单不一致，禁止打包")
        target_model = package_root / artifact["filename"]
        shutil.copyfile(source_model, target_model)
        if sha256_file(target_model) != release_entry["model_sha256"]:
            raise ValueError(f"{task}提交包内模型哈希发生变化")

        model = joblib.load(target_model)
        required = list(model.required_columns_)
        target = artifact["target"]
        if target in required or ID_COLUMN in required:
            raise ValueError(f"{task}提交包模型字段契约包含目标或标识符")
        manifest_tasks[task] = {
            "target": target,
            "selected_model": artifact["selected_model"],
            "prediction_column": f"{target}_prediction",
            "model_filename": artifact["filename"],
            "model_sha256": release_entry["model_sha256"],
            "required_columns": required,
            "required_column_count": len(required),
            "training_rows": release_entry["training_rows"],
        }
        written.append(target_model)

    _copy_tree(root / "src", package_root / "source" / "src")
    written.append(package_root / "source" / "src")

    package_manifest = {
        "run_id": release_manifest["run_id"],
        "status": "submission_package_ready",
        "experiment": "final_release_submission_package",
        "training_role": release_manifest["training_role"],
        "holdout_run_id": release_manifest["holdout_run_id"],
        "registry_run_id": release_manifest["registry_run_id"],
        "release_training_manifest_sha256": sha256_file(release_manifest_path),
        "data_sha256": release_manifest["data_sha256"],
        "input_contract": {
            "mode": config["input_contract"]["mode"],
            "id_column": ID_COLUMN,
            "accepts_extra_columns": bool(
                config["input_contract"]["accepts_extra_columns"]
            ),
            "description": config["input_contract"]["description"],
            "ambiguity_disclosure": config["input_contract"]["ambiguity_disclosure"],
        },
        "tasks": manifest_tasks,
    }
    manifest_path = package_root / "model_manifest.json"
    manifest_path.write_text(
        json.dumps(package_manifest, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    written.append(manifest_path)

    example_inputs = _example_inputs(root, package_root)
    written.extend(example_inputs.values())

    for task in TASKS:
        prediction = _predict_with_package_model(
            package_root, task, example_inputs[task]
        )
        output_path = results_root / f"example_{task}_predictions.csv"
        prediction.to_csv(output_path, index=False, encoding="utf-8")
        written.append(output_path)

    return written


def _predict_with_package_model(
    package_root: Path, task: str, input_path: Path
):
    import pandas as pd

    manifest = _strict_json(package_root / "model_manifest.json")
    model = joblib.load(package_root / manifest["tasks"][task]["model_filename"])
    frame = pd.read_csv(input_path)
    prediction = model.predict(frame)
    output = frame[[ID_COLUMN]].copy()
    output[manifest["tasks"][task]["prediction_column"]] = prediction
    return output


def verify_submission_package(root: Path) -> dict[str, Any]:
    config = load_release_config(root)
    package_root = root / config["submission"]["package_directory"]
    manifest = _strict_json(package_root / "model_manifest.json")
    findings: dict[str, Any] = {"tasks": {}}
    for task in TASKS:
        entry = manifest["tasks"][task]
        model_path = package_root / entry["model_filename"]
        if not model_path.exists():
            raise FileNotFoundError(f"{task}模型文件缺失: {entry['model_filename']}")
        model_hash = sha256_file(model_path)
        if model_hash != entry["model_sha256"]:
            raise ValueError(f"{task}模型哈希与提交包清单不一致")
        model = joblib.load(model_path)
        if list(model.required_columns_) != list(entry["required_columns"]):
            raise ValueError(f"{task}模型必需字段与清单不一致")
        findings["tasks"][task] = {
            "model_hash_verified": True,
            "required_column_count": len(entry["required_columns"]),
        }
    findings["status"] = "verified"
    return findings
