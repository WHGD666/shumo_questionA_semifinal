from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _read_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: float, digits: int = 6) -> str:
    return f"{float(value):.{digits}f}"


def validate_stage_sources(root: Path) -> tuple[pd.DataFrame, dict, dict, pd.DataFrame]:
    registry = pd.read_csv(root / "outputs" / "tables" / "task2_experiment_registry.csv")
    freeze = _read_manifest(root / "outputs" / "logs" / "task2_frozen_candidate_manifest.json")
    learning = _read_manifest(root / "outputs" / "logs" / "task2_cyclic_elastic_learning_curve_manifest.json")
    curve = pd.read_csv(root / "outputs" / "tables" / "task2_cyclic_elastic_learning_curve_aggregate.csv")
    selected = registry.loc[registry["decision"] == "selected"]
    if len(selected) != 1:
        raise ValueError("任务二实验登记表必须且只能包含一个 selected 候选")
    if freeze.get("status") != "development_candidate_frozen":
        raise ValueError("任务二开发集候选尚未冻结")
    if freeze.get("holdout_evaluated") is not False or learning.get("holdout_evaluated") is not False:
        raise ValueError("任务二留出集状态异常")
    if selected.iloc[0]["run_id"] != freeze.get("selection_run_id"):
        raise ValueError("任务二报告来源中的候选运行编号不一致")
    if learning.get("diagnosis", {}).get("practical_plateau_supported") is not True:
        raise ValueError("任务二学习曲线没有通过停止规则")
    if abs(float(selected.iloc[0]["oof_adjusted_r2_raw"]) - float(freeze["selected_oof_metrics"]["adjusted_r2_raw"])) > 1e-12:
        raise ValueError("任务二报告来源中的调整 R2 不一致")
    return registry, freeze, learning, curve


def build_report_text(registry: pd.DataFrame, freeze: dict, learning: dict, curve: pd.DataFrame) -> str:
    selected = registry.loc[registry["decision"] == "selected"].iloc[0]
    competition = registry.loc[registry["decision"] == "scientific_comparator"].iloc[0]
    lines = [
        "# 复赛任务二阶段实验报告",
        "",
        "> 本报告仅记录固定开发集上的模型选择和诊断结果，不包含最终留出集或比赛隐藏测试集成绩。所有比较均使用同一份五折划分和折外预测。",
        "",
        "## 1. 任务与验证协议",
        "",
        "任务二以 `Productivity_Score` 为连续预测目标，主要指标为按原始工程字段数量计算的调整决定系数，辅助记录原始R²、MAE与RMSE。10000条样本中，8000条被固定为开发集，2000条被封存为一次性最终留出集；当前所有任务二实验均未使用留出集标签。",
        "",
        "模型禁止使用 `Person_ID` 和目标字段本身。竞赛契约保留全部其他规则允许字段；科学代理剔除契约进一步排除 `Health_Score`、`Fitness_Level`、`Healthy_Aging_Score` 和 `Wellness_Category` 四个综合变量。原始入睡与起床时刻在每个训练折内转换为正余弦周期特征后删除，58个源字段形成60个工程字段。缺失值填补、标准化和类别编码均仅在相应训练折内拟合。",
        "",
        "## 2. 受控实验过程",
        "",
        "|序号|实验|特征契约|最佳模型|OOF调整R²|OOF R²|OOF MAE|OOF RMSE|结论|",
        "|---:|---|---|---|---:|---:|---:|---:|---|",
    ]
    for row in registry.itertuples(index=False):
        lines.append(
            f"|{row.order}|{row.experiment}|{row.feature_contract}|{row.best_model}|"
            f"{_fmt(row.oof_adjusted_r2_raw)}|{_fmt(row.oof_r2)}|{_fmt(row.oof_mae)}|"
            f"{_fmt(row.oof_rmse)}|{row.decision}|"
        )
    lines.extend([
        "",
        "原始时间基线中，PLS在科学代理剔除契约下取得最高调整R²。将两个时刻转换为周期坐标后，固定ElasticNet超过基线；随后仅在预先限定的30组参数中搜索，确定 `alpha=0.01`、`l1_ratio=0.9`。该模型使用L1与L2混合正则化，在高维独热编码和相关行为指标并存时能够同时完成系数收缩与稳定估计。",
        "",
        "为检验线性稀疏结构是否遗漏可泛化的非线性关系，依次比较了LightGBM/CatBoost、嵌套交叉拟合残差修正、定向样条加性模型、PLS/PCR潜因子模型以及RBF/Nystroem核近似。上述路线均未超过冻结ElasticNet，相关失败结果继续保留为模型选择与消融证据。",
        "",
        "## 3. 冻结候选",
        "",
        f"任务二冻结候选为科学代理剔除契约下的 `ElasticNet(alpha=0.01, l1_ratio=0.9)`。其开发集OOF调整R²为 **{_fmt(selected['oof_adjusted_r2_raw'])}**，原始R²为 **{_fmt(selected['oof_r2'])}**，MAE为 **{_fmt(selected['oof_mae'])}**，RMSE为 **{_fmt(selected['oof_rmse'])}**。候选选择运行编号为 `{freeze['selection_run_id']}`。",
        "",
        f"包含四个综合代理的竞赛契约取得调整R² **{_fmt(competition['oof_adjusted_r2_raw'])}** 和原始R² **{_fmt(competition['oof_r2'])}**，均未超过代理剔除契约。因此最终选择并不是用科学解释性牺牲预测精度，而是在略高开发集指标的同时减少对综合代理的依赖。",
        "",
        "## 4. 学习曲线与停止规则",
        "",
        "冻结模型在20%、40%、60%、80%和100%外层训练数据上分别使用5个固定种子构造嵌套子样本。该实验只改变训练样本量，不重新选择参数。",
        "",
        "|训练比例|每折训练样本|平均训练R²|平均OOF调整R²|平均OOF R²|OOF R²标准差|相对前一比例增量|",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in curve.itertuples(index=False):
        delta = "—" if pd.isna(row.delta_oof_r2_vs_previous_fraction) else _fmt(row.delta_oof_r2_vs_previous_fraction)
        lines.append(
            f"|{int(round(100 * row.training_fraction))}%|{int(row.training_rows_per_fold)}|"
            f"{_fmt(row.mean_train_r2)}|{_fmt(row.mean_oof_adjusted_r2_raw)}|"
            f"{_fmt(row.mean_oof_r2)}|{_fmt(row.std_oof_r2)}|{delta}|"
        )
    diagnosis = learning["diagnosis"]
    lines.extend([
        "",
        f"训练比例从80%增加到100%时，OOF R²仅增加 **{_fmt(diagnosis['gain_oof_r2_80_to_100'])}**；全量训练时训练—验证R²差距为 **{_fmt(diagnosis['full_train_validation_r2_gap'])}**。两项均低于预先冻结的0.005和0.03阈值，实际性能平台诊断通过。确认运行编号为 `{freeze['confirmation_run_id']}`。",
        "",
        "这一结果表明，当前模型没有明显过拟合，继续在相同字段和开发集上扩大模型搜索的预期收益很小。该结论只表示当前数据与特征条件下的实际平台，不构成数学上的绝对性能上限。新增直接相关测量变量、更大样本或不同数据生成机制仍可能改变上限。",
        "",
        "## 5. 阶段结论与后续门禁",
        "",
        "任务二的特征契约、周期时间编码、模型类别、正则化参数和开发集指标现已冻结。不得再根据当前开发集继续扩大参数搜索，也不得把本报告中的OOF指标表述为官方成绩。",
        "",
        "当前尚未训练最终部署模型，也没有启封2000条最终留出集。待任务三开发集候选同样冻结后，再统一进入最终模型训练与一次性留出集评估。",
        "",
    ])
    return "\n".join(lines)


def build_evidence_text(registry: pd.DataFrame, freeze: dict) -> str:
    lines = [
        "# 任务二论文证据索引",
        "",
        "本索引用于将任务二论文结论定位到机器可读证据。指标应从下列文件生成，不应手工维护另一套数值。",
        "",
        "|证据主题|主要文件|用途|",
        "|---|---|---|",
        "|冻结候选与指标|`outputs/logs/task2_frozen_candidate_manifest.json`|候选结构、OOF指标、代理披露、学习曲线确认和留出集状态|",
        "|完整实验路径|`outputs/tables/task2_experiment_registry.csv`|论文模型对比表、实验取舍、运行编号和证据路径|",
        "|冻结候选折级指标|`outputs/tables/task2_cyclic_elastic_tuning_fold_metrics.csv`|五折调整R²、R²、MAE、RMSE及波动|",
        "|冻结候选紧凑OOF预测|`outputs/predictions/task2_frozen_candidate_oof_predictions.csv`|预测—真值图、残差图和误差分析|",
        "|学习曲线汇总|`outputs/tables/task2_cyclic_elastic_learning_curve_aggregate.csv`|训练规模、泛化差距与平台诊断|",
        "|学习曲线重复结果|`outputs/tables/task2_cyclic_elastic_learning_curve_repeat_metrics.csv`|五个固定子采样种子的稳定性|",
        "|特征契约与代理审计|`outputs/tables/task2_feature_contract.csv`、`outputs/tables/task2_proxy_audit.csv`|允许字段、综合代理和高信号字段披露|",
        "|共线性诊断|`outputs/tables/task2_vif.csv`、`outputs/tables/task2_numeric_collinearity.csv`|VIF与数值字段相关性说明|",
        "|冻结数据划分|`data/splits/split_manifest.json`、`data/splits/split_assignments.csv`|证明开发集、留出集和五折划分固定|",
        "|原始数据审计|`outputs/tables/data_overview.csv`、`outputs/tables/column_profile.csv`|数据规模、缺失情况和字段类型描述|",
        "",
        "## 正式运行编号",
        "",
    ]
    seen: set[tuple[str, str]] = set()
    for row in registry.itertuples(index=False):
        item = (row.experiment, row.run_id)
        if item not in seen:
            lines.append(f"- `{row.experiment}`：`{row.run_id}`")
            seen.add(item)
    lines.extend([
        "",
        "## 使用约束",
        "",
        f"- 开发集候选选择运行：`{freeze['selection_run_id']}`。",
        f"- 学习曲线确认运行：`{freeze['confirmation_run_id']}`。",
        "- 当前所有论文证据均来自开发集OOF或其子样本诊断。",
        "- `holdout_evaluated=false`，不得提前生成最终留出集结论。",
        "- 调整R²以60个工程原始字段作为主要惩罚维度；独热变换宽度只作为诊断。",
        "- 完整竞赛契约与综合代理剔除契约必须分别标注。",
        "",
    ])
    return "\n".join(lines)


def build_task2_stage_report(root: Path, overwrite: bool = False) -> list[Path]:
    registry, freeze, learning, curve = validate_stage_sources(root)
    report_path = root / "docs" / "task2_stage_report.md"
    evidence_path = root / "docs" / "task2_evidence_index.md"
    paper_registry_path = root / "paper" / "materials" / "tables" / "task2_experiment_registry.csv"
    paper_curve_path = root / "paper" / "materials" / "tables" / "task2_learning_curve.csv"
    paths = [report_path, evidence_path, paper_registry_path, paper_curve_path]
    if not overwrite and any(path.exists() for path in paths):
        raise FileExistsError("任务二阶段材料已存在；确认重建时请使用 --overwrite")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    paper_registry_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(build_report_text(registry, freeze, learning, curve), encoding="utf-8")
    evidence_path.write_text(build_evidence_text(registry, freeze), encoding="utf-8")
    registry.to_csv(paper_registry_path, index=False, encoding="utf-8-sig")
    curve.to_csv(paper_curve_path, index=False, encoding="utf-8-sig")
    return paths
