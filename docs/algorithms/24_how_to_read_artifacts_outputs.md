# 24. 如何阅读 `artifacts/` 下的所有产物

## 为什么要先学会看 `artifacts/`

很多初学者训练完模型后，最容易迷糊的不是模型代码，而是：

- 训练到底输出了什么
- 哪个文件才是最重要的
- 哪些文件是给机器读的
- 哪些文件是给人看的

而 `artifacts/` 正是这些训练产物的主要落点。

所以学会读 `artifacts/`，基本就学会了“怎么看一次实验到底发生了什么”。

## `artifacts/` 里通常有什么

当前项目里，一次训练通常会生成一个实验目录，例如：

```text
artifacts/real_train_5ep_single_scale_tuned/
```

这个目录下常见的文件包括：

- `best_fusion_model.pt`
- `training_history.json`
- `training_metadata.json`
- `training_session_manifest.json`
- `evaluation_report.json`

你可以把它理解成：

- 一个实验目录 = 一次训练的完整结果包

## 先建立一个最重要的概念

`artifacts/` 里的文件，主要分成两类：

### 1. 给程序继续使用的文件

例如：

- `best_fusion_model.pt`
- `training_session_manifest.json`
- `evaluation_report.json`

### 2. 给人查看和分析的文件

例如：

- `training_history.json`
- `training_metadata.json`

也就是说，`artifacts/` 既是机器产物目录，也是实验记录目录。

## 第 1 个最重要的文件：`best_fusion_model.pt`

这是当前实验保存下来的**最佳模型权重**。

它不是“最后一轮模型”这个概念那么简单，而是训练过程中按照当前模型选择规则保存下来的 best checkpoint。

你可以把它理解成：

- 当前这次实验最值得拿去评估、推理或继续分析的模型文件

如果你后面要执行：

- `manage.py evaluate`

通常最先用的就是它。

## 第 2 个最重要的文件：`training_history.json`

这个文件记录的是训练过程中每个 epoch 的主要指标。

当前常见内容包括：

- `epoch`
- `train_loss`
- `train_accuracy`
- `train_macro_f1`
- `val_loss`
- `val_accuracy`
- `val_macro_f1`
- `learning_rate`

这个文件最适合回答的问题是：

- 模型是不是在收敛
- 哪一轮开始明显变好
- 有没有出现过拟合迹象
- 学习率变化和指标变化是否一致

## 第 3 个关键文件：`training_metadata.json`

这个文件更像一份训练摘要。

它通常会告诉你：

- 类别映射是什么
- 当前配置是什么
- 数据规模大概是多少
- 本次训练的一些核心元信息是什么

如果你后面忘了这次实验到底是几类任务、用的什么配置，通常应该先看它。

## 第 4 个关键文件：`training_session_manifest.json`

这个文件非常重要，但初学者很容易忽略。

它记录的是：

- 哪些 session 进了训练集
- 哪些 session 进了验证集
- 哪些 session 进了测试集

为什么它这么重要？

因为当前项目是按 `session` 切分，而不是按 clip 随机打散。

所以如果你要比较两次实验是否公平，最先要检查的往往就是：

- 它们的 `training_session_manifest.json` 是否一致

## 第 5 个关键文件：`evaluation_report.json`

这个文件是独立评估阶段输出的结果摘要。

它最适合回答：

- 最终测试集效果到底是多少
- 每一类表现如何
- 哪几类容易混淆

当前常见字段包括：

- `accuracy`
- `macro_precision`
- `macro_recall`
- `macro_f1`
- `loss`
- `labels`
- `confusion_matrix`
- `per_class`
- `evaluated_sessions`

## 推荐的阅读顺序

如果你想快速看懂一次实验，推荐按这个顺序看：

```text
training_metadata.json
-> training_session_manifest.json
-> training_history.json
-> evaluation_report.json
-> best_fusion_model.pt
```

为什么这样看？

- 先看这次实验是什么
- 再看数据怎么切的
- 再看训练过程如何
- 再看最终测试结果
- 最后确认最优权重文件在哪里

## 如果你只想知道“这次实验好不好”

那最推荐先看两份文件：

- `training_history.json`
- `evaluation_report.json`

前者告诉你：

- 训练过程是不是健康

后者告诉你：

- 最终测试到底好不好

## 如果你想比较两次实验谁更好

推荐按下面顺序比较：

1. 先比 `training_session_manifest.json`
2. 再比 `training_metadata.json`
3. 再比 `evaluation_report.json`
4. 最后再看 `training_history.json`

这是因为：

- 数据切分不同，结果就不能简单横向比
- 配置不同，结果也不能只看一个分数

所以真正公平的比较一定要先确认实验条件是否一致。

## `artifacts/` 和 `reports/real/` 是什么关系

这两个目录很容易混。

你可以这样理解：

- `artifacts/`：原始实验产物目录
- `reports/real/`：从实验产物生成出来的可视化图表目录

也就是说：

- `training_history.json` 在 `artifacts/`
- `training_curves.png` 在 `reports/real/`

前者更适合程序和精细分析，后者更适合人工快速查看。

## 什么时候说明一次训练“没跑完整”

如果你看到某个实验目录里：

- 没有 `best_fusion_model.pt`
- 没有 `training_history.json`
- 没有 `evaluation_report.json`

那通常说明：

- 训练中断了
- 或者评估还没跑
- 或者输出目录不是完整实验目录

## 一个最实用的判断方式

如果你只想快速判断某个实验目录是否完整，可直接检查这 5 个文件是否都在：

- `best_fusion_model.pt`
- `training_history.json`
- `training_metadata.json`
- `training_session_manifest.json`
- `evaluation_report.json`

如果这五项都在，通常说明：

- 训练跑完了
- 评估也跑完了
- 后续分析已经有足够依据

## 对应源码

- `src/training/trainer.py`
- `src/training/evaluator.py`
- `src/reporting/report_builder.py`

## 初学者最该记住的一句话

`artifacts/` 不是杂物堆，而是一次实验从训练过程、数据切分、最佳模型到最终评估的完整证据链。
