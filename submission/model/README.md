# 复赛A题统一模型调用包

三个回归任务的最终发布模型与统一推理入口。模型结构和参数保持冻结，
发布模型在一次性留出集评估完成后用全部 10,000 条有标签数据重新拟合；
留出集结果未参与任何模型选择或调参。

## 目录结构

```text
task1.joblib / task2.joblib / task3.joblib   三个任务的发布模型
predict.py                                   统一推理入口（仅推理，不训练）
verify_package.py                            完整性自检（哈希、字段契约、依赖）
model_manifest.json                          运行编号、模型哈希、必需字段契约
requirements.txt                             依赖锁定版本（Python 3.12）
source/src/                                  模型反序列化所需的源代码副本
example_input.csv                            20行全列示例输入（可直接用于 --task all）
example_input_task1/2/3.csv                  分别掩去自身目标列的严格契约示例
```

## 快速开始

```bash
pip install -r requirements.txt
python verify_package.py
python predict.py --task task1 --input example_input.csv --output task1_predictions.csv
python predict.py --task all   --input example_input.csv --output-dir outputs
```

## 输入契约

- 必须提供标识符列 `Person_ID`，要求唯一且非空；
- 每个任务独立评估，只要求该任务模型的必需字段齐全
  （逐字段清单见 `model_manifest.json` 中各任务的 `required_columns`）；
- **自身预测目标列允许缺失，预测逻辑不读取自身目标**；
  注意其他任务的目标字段可能是本任务模型的必需特征
  （例如任务1需要 `Productivity_Score` 与 `Health_Score`），
  逐任务必需字段以 `model_manifest.json` 为准；
- 输入中的多余列自动忽略；缺少必需列时报错并列出缺失字段；
- 输出保持输入行的顺序与 `Person_ID` 不变，预测列名为
  `Sleep_Quality_Score_prediction`、`Productivity_Score_prediction`、
  `Health_Score_prediction`。

## 结果说明

- 模型哈希以 `model_manifest.json` 为准，与训练清单
  `final_release_model_training_manifest.json` 一致；
- 本包内所有指标均为内部开发集与一次性内部留出集结果，
  不是组委会官方隐藏测试成绩；
- 本包仅用于推理，禁止用隐藏测试数据重新训练或据此调参。
