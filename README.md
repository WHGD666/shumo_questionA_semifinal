# 复赛 A 题独立工程

本目录用于复赛 A 题（健康生活方式与早起习惯数据分析），和初赛工程完全分离。
三个回归任务的实验、一次性留出集评估、最终发布模型与统一提交包均已完成。

## 复赛任务与最终结果

| 任务 | 目标 | 竞赛模型 | 开发集OOF主指标 | 内部留出集主指标 | 留出集MAE |
|---|---|---|---:|---:|---:|
| 任务一 | `Sleep_Quality_Score` | `lightgbm_residual_w0p75` | R² 0.793587 | R² 0.794940 | 0.568353 |
| 任务二 | `Productivity_Score` | `elastic_a0p01_l1_0p9` | 调整R² 0.620162 | 调整R² 0.607313 | 0.574338 |
| 任务三 | `Health_Score` | `additive_elastic_k4_d2_a0p003` | R² 0.980700 | R² 0.980747 | 1.365843 |

以上全部为内部评估结果，**不是组委会官方隐藏测试成绩**。
唯一权威数字来源是 `outputs/tables/final_results_registry.csv`；
任何文档、模型卡或论文材料引用指标时必须从该注册表生成，不得手工另写。

代理变量披露与无代理科学对照结果同样记录在结果注册表中
（`proxy_policy`、`scientific_comparator_name` 等字段）。

## 关键冻结事实

- 数据 10,000 行，固定划分 8,000 条开发集 / 2,000 条留出集；
  开发集五折（每折 1,600 条），划分文件为 `data/splits/split_assignments.csv`。
- 一次性留出集评估已完成并永久封存：
  `outputs/logs/final_holdout_consumed.json`（run `20260830T115206+0800_final_holdout_7abae5b7`），
  评估后未做任何调参或模型更换。
- 最终发布模型用全部 10,000 条有标签数据按冻结结构重拟合
  （run `20260830T132906+0800_final_release_9a88322f`），
  模型文件位于 `outputs/models/release/`（不入库，可由脚本重建）。
- 推理契约：每任务独立测试，仅自身目标允许缺失，
  其他任务的目标字段可作为必需特征；输入多余列自动忽略，
  输出保持 `Person_ID` 顺序不变。

## 目录结构

```text
data/
├─ raw/                    原始数据，只读（不入库）
├─ processed/              可复现的数据处理产物
└─ splits/                 固定划分（已冻结入库）
configs/                   三任务实验与发布配置
docs/                      审计记录、阶段报告、评估协议
src/                       数据审计、协议、建模与打包模块
scripts/                   审计、训练、评估、打包脚本
tests/                     自动化测试（当前 151 项全部通过）
outputs/
├─ tables/                 指标与注册表
├─ predictions/            冻结候选 OOF 与留出集预测
├─ models/                 evaluation 与 release 模型（不入库）
└─ logs/                   运行清单与历史记录
submission/
├─ model/                  统一提交包（模型、predict.py、自检、契约）
└─ results/                示例预测输出
paper/materials/           论文图表与证据材料（未开始，见门禁说明）
```

## 常用命令

```bash
# 全量测试
python -m unittest discover -s tests

# 重建最终发布模型（结构参数冻结，仅重拟合）
python scripts/train_final_release_models.py --overwrite

# 重建统一提交包
python scripts/build_submission_package.py --overwrite

# 提交包内自检与推理
cd submission/model
python verify_package.py
python predict.py --task all --input example_input.csv --output-dir outputs
```

环境：Python 3.12，依赖版本见 `submission/model/requirements.txt`。
训练历史与逐次运行清单见 `outputs/logs/` 及 `outputs/logs/history/`。

## 门禁说明

论文表格、论文图片与论文证据索引的生成代码尚未创建。
按 `docs/final_evaluation_protocol.md` 第 5 节，进入该阶段前必须停止并
获得用户明确确认（该门禁由自动化测试校验）。
