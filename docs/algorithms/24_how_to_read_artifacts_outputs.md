# 24. 如何阅读 `artifacts/` 下的所有产物

## 为什么要先学会看 `artifacts/`

很多初学者训练完模型后，最容易迷糊的不是模型代码，而是：

- 训练到底输出了什么
- 哪个文件才是最重要的
- 哪些文件是给机器读的
- 哪些文件是给人看的

而 `artifacts/` 正是这些训练产物的主要落点。

## 一个实验目录里通常有什么

当前项目里，一次训练通常会生成一个实验目录，例如：

```text
artifacts/20260322-10-4-cpu/
```

里面常见的文件包括：

- `best_fusion_model.pt`
- `training_history.json`
- `training_metadata.json`
- `training_session_manifest.json`
- `evaluation_report.json`

## `best_fusion_model.pt` 是什么

这是当前实验保存下来的最佳模型权重。它不是简单的“最后一轮模型”，而是训练过程中按当前模型选择规则保存下来的 best checkpoint。

## `training_history.json` 看什么

这个文件记录每个 epoch 的主要指标，例如：

- `train_loss`
- `train_accuracy`
- `train_macro_f1`
- `val_loss`
- `val_accuracy`
- `val_macro_f1`
- `learning_rate`

它最适合回答：

- 模型是不是在收敛
- 哪一轮开始明显变好
- 有没有出现过拟合迹象

## `training_metadata.json` 看什么

这个文件更像一份训练摘要。通常会告诉你：

- 当前配置是什么
- 类别映射是什么
- 样本数大概是多少
- 数据切分是什么

## `training_session_manifest.json` 为什么重要

它记录的是：

- 哪些 session 进了训练集
- 哪些进了验证集
- 哪些进了测试集

如果你要比较两次实验是否公平，首先就应该看这份文件是不是一致。

## `evaluation_report.json` 看什么

它是独立评估阶段输出的结果摘要，最适合回答：

- 最终测试集效果到底是多少
- 每一类表现如何
- 哪几类容易混淆

## 推荐的阅读顺序

```text
training_metadata.json
-> training_session_manifest.json
-> training_history.json
-> evaluation_report.json
-> best_fusion_model.pt
```

## `artifacts/` 和 `reports/real/` 是什么关系

你可以这样理解：

- `artifacts/`：原始实验产物目录
- `reports/real/`：从实验产物生成出来的可视化图表目录

## 一个最实用的判断方式

如果你只想快速判断某个实验目录是否完整，可直接检查这 5 个文件是否都在：

- `best_fusion_model.pt`
- `training_history.json`
- `training_metadata.json`
- `training_session_manifest.json`
- `evaluation_report.json`

## 对应源码

- `src/training/trainer.py`
- `src/training/evaluator.py`
- `src/reporting/report_builder.py`


`artifacts/` 不是杂物堆，而是一次实验从训练过程、数据切分、最佳模型到最终评估的完整证据链。