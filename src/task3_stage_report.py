from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _read_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: float, digits: int = 6) -> str:
    return f"{float(value):.{digits}f}"


def validate_stage_sources(root: Path) -> tuple[pd.DataFrame, dict, dict, dict]:
    registry = pd.read_csv(root / "outputs" / "tables" / "task3_experiment_registry.csv")
    freeze = _read_manifest(root / "outputs" / "logs" / "task3_frozen_candidate_manifest.json")
    protocol = _read_manifest(root / "outputs" / "logs" / "task3_protocol_audit_manifest.json")
    refinement = _read_manifest(
        root / "outputs" / "logs" / "task3_health_additive_refinement_manifest.json"
    )
    selected = registry.loc[registry["decision"] == "selected"]
    scientific = registry.loc[registry["decision"] == "scientific_comparator"]
    if len(selected) != 1 or len(scientific) != 1:
        raise ValueError("任务三登记表必须包含一个竞赛候选和一个科学对照")
    if freeze.get("status") != "development_candidate_frozen":
        raise ValueError("任务三开发集候选尚未冻结")
    if any(
        manifest.get("holdout_evaluated") is not False
        for manifest in (freeze, protocol, refinement)
    ):
        raise ValueError("任务三留出集状态异常")
    if selected.iloc[0]["run_id"] != freeze.get("selection_run_id"):
        raise ValueError("任务三报告来源中的候选运行编号不一致")
    stop = freeze.get("stop_confirmation", {})
    if stop.get("material_improvement") is not False:
        raise ValueError("任务三边界确认没有通过停止规则")
    if abs(float(selected.iloc[0]["oof_r2"]) - float(freeze["selected_oof_metrics"]["r2"])) > 1e-12:
        raise ValueError("任务三报告来源中的候选 R2 不一致")
    checks = protocol.get("deterministic_proxy_checks", {})
    if not checks or not all(checks.values()):
        raise ValueError("任务三确定性代理审计没有全部通过")
    return registry, freeze, protocol, refinement


def build_proxy_comparison(registry: pd.DataFrame) -> pd.DataFrame:
    selected = registry.loc[registry["decision"] == "selected"].iloc[0]
    scientific = registry.loc[registry["decision"] == "scientific_comparator"].iloc[0]
    rows = [
        {
            "role": "competition_candidate",
            "feature_contract": selected["feature_contract"],
            "model": selected["best_model"],
            "oof_r2": float(selected["oof_r2"]),
            "oof_mae": float(selected["oof_mae"]),
            "oof_rmse": float(selected["oof_rmse"]),
            "r2_gap_vs_scientific": float(selected["oof_r2"] - scientific["oof_r2"]),
            "healthy_aging_score_included": True,
        },
        {
            "role": "scientific_comparator",
            "feature_contract": scientific["feature_contract"],
            "model": scientific["best_model"],
            "oof_r2": float(scientific["oof_r2"]),
            "oof_mae": float(scientific["oof_mae"]),
            "oof_rmse": float(scientific["oof_rmse"]),
            "r2_gap_vs_scientific": 0.0,
            "healthy_aging_score_included": False,
        },
    ]
    return pd.DataFrame(rows)


def build_report_text(registry: pd.DataFrame, freeze: dict, protocol: dict) -> str:
    selected = registry.loc[registry["decision"] == "selected"].iloc[0]
    scientific = registry.loc[registry["decision"] == "scientific_comparator"].iloc[0]
    proxy = freeze["proxy_disclosure"]
    stop = freeze["stop_confirmation"]
    lines = [
        "# 复赛任务三阶段实验报告",
        "",
        "> 本报告记录固定开发集上的模型选择、代理审计和停止依据，不包含最终留出集或比赛隐藏测试集成绩。所有指标均来自相同五折划分下的折外预测。",
        "",
        "## 1. 任务与验证协议",
        "",
        "任务三以 `Health_Score` 为连续预测目标，主要评价指标为决定系数R²，同时记录MAE与RMSE。原始10000条样本中，8000条固定为开发集，2000条封存为一次性最终留出集；开发集继续使用预先生成的固定五折，全部填补、标准化、类别编码和样条变换均只在相应训练折内拟合。",
        "",
        "模型不使用 `Person_ID` 和目标字段本身。原始入睡与起床时刻被转换为正余弦周期坐标后删除。任务三开发阶段没有读取留出集标签，也没有根据留出集结果调整特征、结点数或正则化参数。",
        "",
        "## 2. 目标代理审计与双特征契约",
        "",
        "审计发现 `Fitness_Level` 与 `Wellness_Category` 都是 `Health_Score` 按固定阈值生成的类别字段，并且在当前数据中逐行完全一致。这两个字段属于可确定性重建目标区间的直接代理，因此从所有正式任务三模型中排除。",
        "",
        f"`Healthy_Aging_Score` 与目标的开发集Pearson相关系数为 **{_fmt(protocol['healthy_aging_pearson_correlation'])}**，单变量线性R²为 **{_fmt(protocol['healthy_aging_univariate_linear_r2'])}**。该字段不是目标的确定性复制，但承载了很强的综合健康信息。因此建立两套契约：竞赛契约保留该规则允许字段；科学契约进一步将其剔除，用于披露模型对综合代理的依赖。",
        "",
        "## 3. 受控实验过程",
        "",
        "|序号|实验|角色|特征契约|最佳模型|OOF R²|OOF MAE|OOF RMSE|结论|",
        "|---:|---|---|---|---|---:|---:|---:|---|",
    ]
    for row in registry.itertuples(index=False):
        lines.append(
            f"|{row.order}|{row.experiment}|{row.role}|{row.feature_contract}|{row.best_model}|"
            f"{_fmt(row.oof_r2)}|{_fmt(row.oof_mae)}|{_fmt(row.oof_rmse)}|{row.decision}|"
        )
    lines.extend([
        "",
        "线性基线中，OLS、Ridge与ElasticNet结果接近，而ExtraTrees明显较低，说明目标的主体结构接近加性关系。小范围线性调参仅带来低于预设阈值的提升，没有被视为实质突破。",
        "",
        "对线性模型折外残差进行诊断后，BMI、体重和年龄等变量表现出明显的平方项与折点结构。随后仅对有诊断证据的连续变量使用低阶样条，其余连续变量保持线性、类别变量保持独热编码。定向加性模型在两套契约下均产生超过阈值的提升，验证了平滑非线性假设。",
        "",
        "## 4. 冻结竞赛候选",
        "",
        f"冻结竞赛候选为 `{selected['best_model']}`，使用竞赛代理保留契约。其样条结点数为4、次数为2，ElasticNet参数为 `alpha=0.003`、`l1_ratio=0.9`。开发集OOF R²为 **{_fmt(selected['oof_r2'])}**，MAE为 **{_fmt(selected['oof_mae'])}**，RMSE为 **{_fmt(selected['oof_rmse'])}**；五折R²标准差为 **{_fmt(selected['fold_r2_std'])}**。候选选择运行编号为 `{freeze['selection_run_id']}`。",
        "",
        f"科学代理剔除对照为 `{scientific['best_model']}`，OOF R²为 **{_fmt(scientific['oof_r2'])}**，MAE为 **{_fmt(scientific['oof_mae'])}**，RMSE为 **{_fmt(scientific['oof_rmse'])}**。竞赛候选相对科学对照的R²差为 **{_fmt(proxy['competition_minus_scientific_oof_r2'])}**，该差值主要反映 `Healthy_Aging_Score` 所携带的综合健康信息，不能全部归因于算法复杂度。",
        "",
        "## 5. 边界确认与停止依据",
        "",
        f"初始加性网格的最优结点数位于4结点边界，因此额外比较4、5和6个结点。边界网格最高结果为 `{stop['best_boundary_model']}`，OOF R²为 **{_fmt(stop['best_boundary_oof_r2'])}**，相对冻结4结点候选仅提高 **{_fmt(stop['delta_vs_selected_oof_r2'])}**，低于预先设定的 **{_fmt(stop['minimum_material_improvement'])}** 门槛。为降低开发集过拟合和部署复杂度，保留更简单的4结点模型。确认运行编号为 `{freeze['confirmation_run_id']}`。",
        "",
        "该停止规则表示继续扩大当前样条网格缺乏足够证据，并不意味着数学意义上的绝对性能上限。新的直接测量变量、更大样本或不同数据分布仍可能改变可达到的效果。",
        "",
        "## 6. 阶段结论与后续门禁",
        "",
        "任务三的目标代理处理、竞赛与科学特征契约、样条字段、结点数、次数、正则化参数和开发集候选现已冻结。开发集搜索到此停止，后续不得根据同一开发集继续扩大参数网格。",
        "",
        "当前尚未训练最终部署模型，也没有启封2000条最终留出集。本报告中的OOF结果仅用于内部候选选择与论文方法论说明，不得表述为比赛官方成绩。",
        "",
    ])
    return "\n".join(lines)


def build_evidence_text(registry: pd.DataFrame, freeze: dict) -> str:
    lines = [
        "# 任务三论文证据索引",
        "",
        "本索引用于将任务三论文结论定位到机器可读证据。所有指标应从下列冻结文件生成，不应手工维护另一套数值。",
        "",
        "|证据主题|主要文件|用途|",
        "|---|---|---|",
        "|冻结候选与指标|`outputs/logs/task3_frozen_candidate_manifest.json`|竞赛候选、科学对照、代理披露、停止依据和留出集状态|",
        "|完整实验路径|`outputs/tables/task3_experiment_registry.csv`|论文模型对比表、实验取舍、运行编号和证据路径|",
        "|候选五折指标|`outputs/tables/task3_health_targeted_additive_models_fold_metrics.csv`|五折R²、MAE、RMSE和模型波动|",
        "|竞赛候选紧凑OOF|`outputs/predictions/task3_frozen_candidate_oof_predictions.csv`|预测—真值图、残差图和误差分析|",
        "|科学对照紧凑OOF|`outputs/predictions/task3_frozen_scientific_comparator_oof_predictions.csv`|代理敏感性和误差对照|",
        "|代理比较表|`paper/materials/tables/task3_proxy_comparison.csv`|竞赛契约与代理剔除契约的直接对比|",
        "|目标代理审计|`outputs/tables/task3_proxy_category_ranges.csv`、`outputs/logs/task3_protocol_audit_manifest.json`|确定性分箱代理与Healthy_Aging关联说明|",
        "|特征契约|`outputs/tables/task3_feature_contract.csv`|允许、排除和强代理字段清单|",
        "|边界确认|`outputs/tables/task3_health_additive_refinement_summary_metrics.csv`|4至6结点对比与停止门槛|",
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
        f"- 样条边界确认运行：`{freeze['confirmation_run_id']}`。",
        "- 当前所有论文指标均为固定开发集OOF结果。",
        "- `holdout_evaluated=false`，不得提前生成最终留出集结论。",
        "- `Fitness_Level` 与 `Wellness_Category` 是确定性目标分箱代理，不能作为任务三正式特征。",
        "- `Healthy_Aging_Score` 保留于竞赛契约，但其信息贡献必须与科学代理剔除对照同时披露。",
        "- 竞赛候选结果、科学对照结果与未来官方隐藏测试结果必须分别标注。",
        "",
    ])
    return "\n".join(lines)


def build_task3_stage_report(root: Path, overwrite: bool = False) -> list[Path]:
    registry, freeze, protocol, _ = validate_stage_sources(root)
    proxy_comparison = build_proxy_comparison(registry)
    report_path = root / "docs" / "task3_stage_report.md"
    evidence_path = root / "docs" / "task3_evidence_index.md"
    paper_registry_path = root / "paper" / "materials" / "tables" / "task3_experiment_registry.csv"
    proxy_path = root / "paper" / "materials" / "tables" / "task3_proxy_comparison.csv"
    paths = [report_path, evidence_path, paper_registry_path, proxy_path]
    if not overwrite and any(path.exists() for path in paths):
        raise FileExistsError("任务三阶段材料已存在；确认重建时请使用 --overwrite")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    paper_registry_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(build_report_text(registry, freeze, protocol), encoding="utf-8")
    evidence_path.write_text(build_evidence_text(registry, freeze), encoding="utf-8")
    registry.to_csv(paper_registry_path, index=False, encoding="utf-8-sig")
    proxy_comparison.to_csv(proxy_path, index=False, encoding="utf-8-sig")
    return paths
