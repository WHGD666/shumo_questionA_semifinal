# 任务三论文证据索引

本索引用于将任务三论文结论定位到机器可读证据。所有指标应从下列冻结文件生成，不应手工维护另一套数值。

|证据主题|主要文件|用途|
|---|---|---|
|冻结候选与指标|`outputs/logs/task3_frozen_candidate_manifest.json`|竞赛候选、科学对照、代理披露、停止依据和留出集状态|
|完整实验路径|`outputs/tables/task3_experiment_registry.csv`|论文模型对比表、实验取舍、运行编号和证据路径|
|候选五折指标|`outputs/tables/task3_health_targeted_additive_models_fold_metrics.csv`|五折R²、MAE、RMSE和模型波动|
|竞赛候选紧凑OOF|`outputs/predictions/task3_frozen_candidate_oof_predictions.csv`|预测—真值图、残差图和误差分析|
|科学对照紧凑OOF|`outputs/predictions/task3_frozen_scientific_comparator_oof_predictions.csv`|代理敏感性和误差对照|
|代理比较表|`paper/materials/tables/task3_proxy_comparison.csv`|竞赛契约与代理剔除契约的直接对比|
|目标代理审计|`outputs/tables/task3_proxy_category_ranges.csv`、`outputs/logs/task3_protocol_audit_manifest.json`|确定性分箱代理与Healthy_Aging关联说明|
|特征契约|`outputs/tables/task3_feature_contract.csv`|允许、排除和强代理字段清单|
|边界确认|`outputs/tables/task3_health_additive_refinement_summary_metrics.csv`|4至6结点对比与停止门槛|
|冻结数据划分|`data/splits/split_manifest.json`、`data/splits/split_assignments.csv`|证明开发集、留出集和五折划分固定|
|原始数据审计|`outputs/tables/data_overview.csv`、`outputs/tables/column_profile.csv`|数据规模、缺失情况和字段类型描述|

## 正式运行编号

- `regression_baseline_competition`：`20260829T234029+0800_task3_baseline_33b2dd82`
- `regression_baseline_scientific`：`20260829T234029+0800_task3_baseline_33b2dd82`
- `linear_tuning_competition`：`20260829T234620+0800_task3_linear_tuning_7e5c7f11`
- `linear_tuning_scientific`：`20260829T234620+0800_task3_linear_tuning_7e5c7f11`
- `targeted_additive_competition`：`20260829T235336+0800_task3_additive_c92b6dee`
- `targeted_additive_scientific`：`20260829T235336+0800_task3_additive_c92b6dee`
- `additive_boundary_competition`：`20260829T235738+0800_task3_additive_refinement_b1e98c2f`
- `additive_boundary_scientific`：`20260829T235738+0800_task3_additive_refinement_b1e98c2f`

## 使用约束

- 开发集候选选择运行：`20260829T235336+0800_task3_additive_c92b6dee`。
- 样条边界确认运行：`20260829T235738+0800_task3_additive_refinement_b1e98c2f`。
- 当前所有论文指标均为固定开发集OOF结果。
- `holdout_evaluated=false`，不得提前生成最终留出集结论。
- `Fitness_Level` 与 `Wellness_Category` 是确定性目标分箱代理，不能作为任务三正式特征。
- `Healthy_Aging_Score` 保留于竞赛契约，但其信息贡献必须与科学代理剔除对照同时披露。
- 竞赛候选结果、科学对照结果与未来官方隐藏测试结果必须分别标注。
