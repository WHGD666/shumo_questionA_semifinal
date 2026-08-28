# 任务一论文证据索引

本索引用于定位论文表格、结论和复现实验所对应的机器可读证据。论文数据应从下列文件生成，不应手工维护另一套指标。

|证据主题|主要文件|用途|
|---|---|---|
|冻结候选与指标|`outputs/logs/task1_frozen_candidate_manifest.json`|最终候选结构、OOF指标、稳定性摘要、代理披露和留出集状态|
|完整实验路径|`outputs/tables/task1_experiment_registry.csv`|论文模型对比表、实验取舍与运行编号|
|最终候选折级表现|`outputs/tables/task1_non_sleep_nested_residual_correction_fold_metrics.csv`|五折R²、MAE、RMSE及波动分析|
|最终候选OOF预测|`outputs/predictions/task1_non_sleep_nested_residual_correction_oof_predictions.csv`|残差图、预测—真值图及误差分析|
|多种子稳定性|`outputs/tables/task1_non_sleep_residual_stability_seed_summary.csv`|随机种子敏感性表|
|代理变量敏感性|`outputs/tables/task1_spline_proxy_validation_summary_metrics.csv`|完整契约与代理剔除契约对照|
|冻结数据划分|`data/splits/split_manifest.json`、`data/splits/split_assignments.csv`|证明开发集、留出集和五折划分固定|
|原始数据审计|`outputs/tables/data_overview.csv`、`outputs/tables/column_profile.csv`|数据规模、缺失情况和字段类型描述|

## 正式运行编号

- `linear_tuning`：`20260828T211938+0800_task1_linear_tuning_e70a65d2`
- `fixed_strong_models`：`20260828T213416+0800_task1_strong_fixed_5c5d02a3`
- `structured_models`：`20260828T220025+0800_task1_structured_071eedc4`
- `spline_tuning`：`20260828T221213+0800_task1_spline_tuning_553589a2`
- `spline_refinement`：`20260828T221950+0800_task1_non_sleep_spline_refinement_8e4d9323`
- `spline_proxy_validation`：`20260828T235920+0800_task1_spline_proxy_77c43179`
- `nested_residual_correction`：`20260829T001923+0800_task1_nested_residual_1daa2d59`

## 使用约束

- 开发集候选选择运行：`20260829T001923+0800_task1_nested_residual_1daa2d59`。
- 稳定性确认运行：`20260829T002532+0800_task1_residual_stability_fe7d4957`。
- 当前所有论文证据均来自开发集OOF结果。
- `holdout_evaluated=false`，不得提前生成最终留出集结论。
- 代理变量完整契约与代理剔除科学对照必须分别标注。
