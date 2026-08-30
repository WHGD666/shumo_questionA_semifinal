# 复赛提交材料

- `model/` 统一模型调用包：三个任务的最终发布模型、统一推理入口
  `predict.py`、完整性自检 `verify_package.py`、字段契约与依赖锁定。
  用法与输入契约见 `model/README.md`。
- `results/` 预测结果：当前保存由 `example_input.csv` 生成的三任务示例
  预测，用于核对输出格式；隐藏测试集的正式预测由组委会数据生成。
