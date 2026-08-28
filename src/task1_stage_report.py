from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _read_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: float, digits: int = 6) -> str:
    return f"{float(value):.{digits}f}"


def validate_stage_sources(root: Path) -> tuple[pd.DataFrame, dict, dict]:
    registry_path = root / "outputs" / "tables" / "task1_experiment_registry.csv"
    freeze_path = root / "outputs" / "logs" / "task1_frozen_candidate_manifest.json"
    stability_path = root / "outputs" / "logs" / "task1_non_sleep_residual_stability_manifest.json"
    registry = pd.read_csv(registry_path)
    freeze = _read_manifest(freeze_path)
    stability = _read_manifest(stability_path)
    selected = registry.loc[registry["decision"] == "selected"]
    if len(selected) != 1:
        raise ValueError("任务一实验登记表必须且只能包含一个 selected 候选")
    selected_row = selected.iloc[0]
    if freeze.get("status") != "development_candidate_frozen":
        raise ValueError("任务一开发集候选尚未冻结")
    if freeze.get("holdout_evaluated") is not False or stability.get("holdout_evaluated") is not False:
        raise ValueError("任务一留出集状态异常")
    if selected_row["run_id"] != freeze.get("selection_run_id"):
        raise ValueError("报告来源中的候选运行编号不一致")
    if abs(float(selected_row["oof_r2"]) - float(freeze["selected_oof_metrics"]["r2"])) > 1e-12:
        raise ValueError("报告来源中的候选 R2 不一致")
    return registry, freeze, stability


def build_report_text(registry: pd.DataFrame, freeze: dict, stability: dict) -> str:
    selected = registry.loc[registry["decision"] == "selected"].iloc[0]
    strict = registry.loc[registry["decision"] == "scientific_comparator"].iloc[0]
    lines = [
        "# 复赛任务一阶段实验报告",
        "",
        "> 本报告记录开发集模型选择过程，不包含最终留出集或隐藏测试集成绩。所有指标均由固定开发集的折外预测计算。",
        "",
        "## 1. 任务与验证协议",
        "",
        "任务一以 `Sleep_Quality_Score` 为连续预测目标，主要评价指标为决定系数 R²，同时记录 MAE 与 RMSE。原始数据共有10000行，其中8000行被固定为开发集，2000行被封存为最终留出集。开发集使用固定五折划分，所有预处理、编码、样条变换和模型拟合均限定在相应训练折内完成。",
        "",
        "按照赛题要求，模型不使用睡眠时长、入睡时间、起床时间、夜间醒来次数等直接睡眠字段。当前竞赛候选使用54个规则允许的非睡眠特征。模型选择期间没有读取或使用留出集标签。",
        "",
        "## 2. 受控实验过程",
        "",
        "|序号|实验|特征契约|最佳模型|OOF R²|OOF MAE|OOF RMSE|结论|",
        "|---:|---|---|---|---:|---:|---:|---|",
    ]
    for row in registry.itertuples(index=False):
        lines.append(
            f"|{row.order}|{row.experiment}|{row.feature_contract}|{row.best_model}|"
            f"{_fmt(row.oof_r2)}|{_fmt(row.oof_mae)}|{_fmt(row.oof_rmse)}|{row.decision}|"
        )
    lines.extend([
        "",
        "线性模型首先建立了稳定基准。固定强树模型没有超过线性基准，表明该任务的主要信息结构并非由高阶树分裂主导。随后引入样条变换，用低阶平滑非线性描述连续健康与行为变量，OOF R²得到稳定提高。样条参数仅在预先限定的邻域内搜索，并按照停止规则选定4个结点、2阶样条和Ridge系数1.0。",
        "",
        "在样条主模型确定后，使用内层四折交叉拟合生成训练残差，再以LightGBM学习剩余非线性结构。残差模型的预测按0.75权重叠加至主模型预测。由于残差标签来自内层折外预测，而不是主模型对自身训练样本的拟合值，因此避免了训练内残差造成的信息乐观偏差。",
        "",
        "## 3. 冻结候选及稳定性",
        "",
        f"最终冻结的开发集候选为 `Spline Ridge + LightGBM residual × 0.75`。其OOF R²为 **{_fmt(selected.oof_r2)}**，MAE为 **{_fmt(selected.oof_mae)}**，RMSE为 **{_fmt(selected.oof_rmse)}**。候选选择运行编号为 `{freeze['selection_run_id']}`。",
        "",
        f"为检验残差模型对内层划分随机性的敏感程度，在不改变模型结构和权重的前提下使用5个固定随机种子重复内层交叉拟合。平均OOF R²为 **{_fmt(stability['mean_oof_r2'])}**，标准差为 **{_fmt(freeze['stability_metrics']['std_oof_r2'])}**；相对基础样条模型的最小R²提升为 **{_fmt(stability['minimum_delta_vs_base_r2'])}**。5次结果均为正提升，稳定性确认通过。确认运行编号为 `{freeze['confirmation_run_id']}`。",
        "",
        "## 4. 代理变量敏感性",
        "",
        f"竞赛候选保留了规则允许的 `Health_Score`、`Fitness_Level`、`Healthy_Aging_Score` 和 `Wellness_Category` 四个综合变量。为说明模型的信息来源，另行构建剔除上述变量的科学对照。相同样条结构下，对照OOF R²为 **{_fmt(strict.oof_r2)}**，相较完整契约下降 **{_fmt(freeze['proxy_disclosure']['proxy_r2_gap'])}**。",
        "",
        "该差异说明综合健康代理承载了较多与睡眠质量相关的信息。论文中应明确披露这一现象，并同时报告完整特征竞赛候选和代理剔除对照，不能把两者混写成同一种特征条件。",
        "",
        "## 5. 阶段结论与后续门禁",
        "",
        "任务一开发集模型探索已经停止，候选结构、参数、特征契约、残差权重和内层种子均已冻结。继续围绕同一开发集扩大搜索可能引入验证集过拟合，因此不再进行无边界调参。",
        "",
        "当前尚未训练最终部署模型，也没有启封2000行最终留出集。待任务二和任务三的开发集候选同样冻结后，再统一执行一次性留出集评估。留出集结果与比赛隐藏测试集结果均应单独标识，不得将本报告的开发集OOF指标表述为官方成绩。",
        "",
    ])
    return "\n".join(lines)


def build_evidence_text(registry: pd.DataFrame, freeze: dict) -> str:
    lines = [
        "# 任务一论文证据索引",
        "",
        "本索引用于定位论文表格、结论和复现实验所对应的机器可读证据。论文数据应从下列文件生成，不应手工维护另一套指标。",
        "",
        "|证据主题|主要文件|用途|",
        "|---|---|---|",
        "|冻结候选与指标|`outputs/logs/task1_frozen_candidate_manifest.json`|最终候选结构、OOF指标、稳定性摘要、代理披露和留出集状态|",
        "|完整实验路径|`outputs/tables/task1_experiment_registry.csv`|论文模型对比表、实验取舍与运行编号|",
        "|最终候选折级表现|`outputs/tables/task1_non_sleep_nested_residual_correction_fold_metrics.csv`|五折R²、MAE、RMSE及波动分析|",
        "|最终候选OOF预测|`outputs/predictions/task1_non_sleep_nested_residual_correction_oof_predictions.csv`|残差图、预测—真值图及误差分析|",
        "|多种子稳定性|`outputs/tables/task1_non_sleep_residual_stability_seed_summary.csv`|随机种子敏感性表|",
        "|代理变量敏感性|`outputs/tables/task1_spline_proxy_validation_summary_metrics.csv`|完整契约与代理剔除契约对照|",
        "|冻结数据划分|`data/splits/split_manifest.json`、`data/splits/split_assignments.csv`|证明开发集、留出集和五折划分固定|",
        "|原始数据审计|`outputs/tables/data_overview.csv`、`outputs/tables/column_profile.csv`|数据规模、缺失情况和字段类型描述|",
        "",
        "## 正式运行编号",
        "",
    ]
    for row in registry.itertuples(index=False):
        lines.append(f"- `{row.experiment}`：`{row.run_id}`")
    lines.extend([
        "",
        "## 使用约束",
        "",
        f"- 开发集候选选择运行：`{freeze['selection_run_id']}`。",
        f"- 稳定性确认运行：`{freeze['confirmation_run_id']}`。",
        "- 当前所有论文证据均来自开发集OOF结果。",
        "- `holdout_evaluated=false`，不得提前生成最终留出集结论。",
        "- 代理变量完整契约与代理剔除科学对照必须分别标注。",
        "",
    ])
    return "\n".join(lines)


def build_task1_stage_report(root: Path, overwrite: bool = False) -> list[Path]:
    registry, freeze, stability = validate_stage_sources(root)
    report_path = root / "docs" / "task1_stage_report.md"
    evidence_path = root / "docs" / "task1_evidence_index.md"
    paper_table_path = root / "paper" / "materials" / "tables" / "task1_experiment_registry.csv"
    paths = [report_path, evidence_path, paper_table_path]
    if not overwrite and any(path.exists() for path in paths):
        raise FileExistsError("任务一阶段材料已存在；确认重建时请使用 --overwrite")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    paper_table_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(build_report_text(registry, freeze, stability), encoding="utf-8")
    evidence_path.write_text(build_evidence_text(registry, freeze), encoding="utf-8")
    registry.to_csv(paper_table_path, index=False, encoding="utf-8-sig")
    return paths
