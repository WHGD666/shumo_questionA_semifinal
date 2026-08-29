# 任务二论文证据索引

本索引用于将任务二论文结论定位到机器可读证据。指标应从下列文件生成，不应手工维护另一套数值。

|证据主题|主要文件|用途|
|---|---|---|
|冻结候选与指标|`outputs/logs/task2_frozen_candidate_manifest.json`|候选结构、OOF指标、代理披露、学习曲线确认和留出集状态|
|完整实验路径|`outputs/tables/task2_experiment_registry.csv`|论文模型对比表、实验取舍、运行编号和证据路径|
|冻结候选折级指标|`outputs/tables/task2_cyclic_elastic_tuning_fold_metrics.csv`|五折调整R²、R²、MAE、RMSE及波动|
|冻结候选紧凑OOF预测|`outputs/predictions/task2_frozen_candidate_oof_predictions.csv`|预测—真值图、残差图和误差分析|
|学习曲线汇总|`outputs/tables/task2_cyclic_elastic_learning_curve_aggregate.csv`|训练规模、泛化差距与平台诊断|
|学习曲线重复结果|`outputs/tables/task2_cyclic_elastic_learning_curve_repeat_metrics.csv`|五个固定子采样种子的稳定性|
|特征契约与代理审计|`outputs/tables/task2_feature_contract.csv`、`outputs/tables/task2_proxy_audit.csv`|允许字段、综合代理和高信号字段披露|
|共线性诊断|`outputs/tables/task2_vif.csv`、`outputs/tables/task2_numeric_collinearity.csv`|VIF与数值字段相关性说明|
|冻结数据划分|`data/splits/split_manifest.json`、`data/splits/split_assignments.csv`|证明开发集、留出集和五折划分固定|
|原始数据审计|`outputs/tables/data_overview.csv`、`outputs/tables/column_profile.csv`|数据规模、缺失情况和字段类型描述|

## 正式运行编号

- `multicollinearity_baselines`：`20260829T172431+0800_task2_baseline_c64a3f82`
- `cyclic_time_encoding`：`20260829T172956+0800_task2_cyclic_time_af600f17`
- `elastic_tuning`：`20260829T173352+0800_task2_elastic_tuning_59b765b7`
- `competition_contract_comparator`：`20260829T173352+0800_task2_elastic_tuning_59b765b7`
- `fixed_strong_models`：`20260829T173919+0800_task2_strong_fixed_fbe8122d`
- `nested_residual_correction`：`20260829T175953+0800_task2_nested_residual_d5661a2c`
- `targeted_additive_models`：`20260829T182540+0800_task2_additive_86e17dc7`
- `latent_factor_models`：`20260829T182928+0800_task2_latent_factor_f7423cdb`
- `kernel_approximation_models`：`20260829T183308+0800_task2_kernel_22f5be57`
- `learning_curve`：`20260829T190021+0800_task2_learning_curve_3d139646`

## 使用约束

- 开发集候选选择运行：`20260829T173352+0800_task2_elastic_tuning_59b765b7`。
- 学习曲线确认运行：`20260829T190021+0800_task2_learning_curve_3d139646`。
- 当前所有论文证据均来自开发集OOF或其子样本诊断。
- `holdout_evaluated=false`，不得提前生成最终留出集结论。
- 调整R²以60个工程原始字段作为主要惩罚维度；独热变换宽度只作为诊断。
- 完整竞赛契约与综合代理剔除契约必须分别标注。
