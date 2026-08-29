# 复赛任务三阶段实验报告

> 本报告记录固定开发集上的模型选择、代理审计和停止依据，不包含最终留出集或比赛隐藏测试集成绩。所有指标均来自相同五折划分下的折外预测。

## 1. 任务与验证协议

任务三以 `Health_Score` 为连续预测目标，主要评价指标为决定系数R²，同时记录MAE与RMSE。原始10000条样本中，8000条固定为开发集，2000条封存为一次性最终留出集；开发集继续使用预先生成的固定五折，全部填补、标准化、类别编码和样条变换均只在相应训练折内拟合。

模型不使用 `Person_ID` 和目标字段本身。原始入睡与起床时刻被转换为正余弦周期坐标后删除。任务三开发阶段没有读取留出集标签，也没有根据留出集结果调整特征、结点数或正则化参数。

## 2. 目标代理审计与双特征契约

审计发现 `Fitness_Level` 与 `Wellness_Category` 都是 `Health_Score` 按固定阈值生成的类别字段，并且在当前数据中逐行完全一致。这两个字段属于可确定性重建目标区间的直接代理，因此从所有正式任务三模型中排除。

`Healthy_Aging_Score` 与目标的开发集Pearson相关系数为 **0.938235**，单变量线性R²为 **0.880284**。该字段不是目标的确定性复制，但承载了很强的综合健康信息。因此建立两套契约：竞赛契约保留该规则允许字段；科学契约进一步将其剔除，用于披露模型对综合代理的依赖。

## 3. 受控实验过程

|序号|实验|角色|特征契约|最佳模型|OOF R²|OOF MAE|OOF RMSE|结论|
|---:|---|---|---|---|---:|---:|---:|---|
|1|regression_baseline_competition|baseline|competition_proxy_inclusive|elastic_net_fixed|0.979621|1.383688|1.748385|baseline_control|
|2|regression_baseline_scientific|proxy_sensitivity_baseline|scientific_proxy_removed|elastic_net_fixed|0.927502|2.617234|3.297721|scientific_baseline|
|3|linear_tuning_competition|bounded_hyperparameter_tuning|competition_proxy_inclusive|elastic_a0p003_l1_0p9|0.979663|1.381874|1.746618|rejected_below_threshold|
|4|linear_tuning_scientific|bounded_hyperparameter_tuning|scientific_proxy_removed|elastic_a0p01_l1_0p9|0.927670|2.614934|3.293886|rejected_below_threshold|
|5|targeted_additive_competition|candidate_selection|competition_proxy_inclusive|additive_elastic_k4_d2_a0p003|0.980700|1.352934|1.701476|selected|
|6|targeted_additive_scientific|scientific_proxy_comparator|scientific_proxy_removed|additive_ridge_k4_d2_a0p3|0.938393|2.427497|3.039948|scientific_comparator|
|7|additive_boundary_competition|boundary_refinement|competition_proxy_inclusive|additive_elastic_k6_d2_a0p001|0.980796|1.351314|1.697256|confirmed_stop|
|8|additive_boundary_scientific|boundary_refinement|scientific_proxy_removed|additive_ridge_k4_d2_a0p3|0.938393|2.427497|3.039948|confirmed_stop|

线性基线中，OLS、Ridge与ElasticNet结果接近，而ExtraTrees明显较低，说明目标的主体结构接近加性关系。小范围线性调参仅带来低于预设阈值的提升，没有被视为实质突破。

对线性模型折外残差进行诊断后，BMI、体重和年龄等变量表现出明显的平方项与折点结构。随后仅对有诊断证据的连续变量使用低阶样条，其余连续变量保持线性、类别变量保持独热编码。定向加性模型在两套契约下均产生超过阈值的提升，验证了平滑非线性假设。

## 4. 冻结竞赛候选

冻结竞赛候选为 `additive_elastic_k4_d2_a0p003`，使用竞赛代理保留契约。其样条结点数为4、次数为2，ElasticNet参数为 `alpha=0.003`、`l1_ratio=0.9`。开发集OOF R²为 **0.980700**，MAE为 **1.352934**，RMSE为 **1.701476**；五折R²标准差为 **0.000715**。候选选择运行编号为 `20260829T235336+0800_task3_additive_c92b6dee`。

科学代理剔除对照为 `additive_ridge_k4_d2_a0p3`，OOF R²为 **0.938393**，MAE为 **2.427497**，RMSE为 **3.039948**。竞赛候选相对科学对照的R²差为 **0.042308**，该差值主要反映 `Healthy_Aging_Score` 所携带的综合健康信息，不能全部归因于算法复杂度。

## 5. 边界确认与停止依据

初始加性网格的最优结点数位于4结点边界，因此额外比较4、5和6个结点。边界网格最高结果为 `additive_elastic_k6_d2_a0p001`，OOF R²为 **0.980796**，相对冻结4结点候选仅提高 **0.000096**，低于预先设定的 **0.000200** 门槛。为降低开发集过拟合和部署复杂度，保留更简单的4结点模型。确认运行编号为 `20260829T235738+0800_task3_additive_refinement_b1e98c2f`。

该停止规则表示继续扩大当前样条网格缺乏足够证据，并不意味着数学意义上的绝对性能上限。新的直接测量变量、更大样本或不同数据分布仍可能改变可达到的效果。

## 6. 阶段结论与后续门禁

任务三的目标代理处理、竞赛与科学特征契约、样条字段、结点数、次数、正则化参数和开发集候选现已冻结。开发集搜索到此停止，后续不得根据同一开发集继续扩大参数网格。

当前尚未训练最终部署模型，也没有启封2000条最终留出集。本报告中的OOF结果仅用于内部候选选择与论文方法论说明，不得表述为比赛官方成绩。
