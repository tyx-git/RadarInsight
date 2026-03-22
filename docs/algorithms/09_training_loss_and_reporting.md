# 09. 训练损失、评估和报告生成

## 当前训练并不是只看一个损失

当前模型训练里有三类输出：

- 主分类输出
- 轨迹辅助头输出
- RD 辅助头输出

这样做的目的，是防止训练后期某一条分支完全失效。

## 为什么要辅助头

如果只看融合头，可能出现一种情况：

- 模型靠其中一个模态就能暂时把损失降下来
- 另一个模态逐渐退化成“陪跑”

辅助头的作用，就是让每一条分支都保持基本可分性。

## 当前还有哪些稳定训练手段

- `label smoothing`
- `AdamW`
- `class-balanced loss`
- `warmup + cosine scheduler`
- `gradient clipping`
- 会话级切分
- 独立评估

## 评估为什么重要

当前项目不只输出 `accuracy`，还输出：

- `macro_f1`
- `per_class`
- `confusion_matrix`

这很重要，因为微动识别不能只看总体准确率。
很多时候真正有价值的是：

- 鸟类和旋翼是否更容易分开
- 固定翼和其他目标是否仍然混淆

## 报告生成现在做什么

当前项目把真实训练与评估结果可视化输出到：

- `reports/real/`

这里只保留真实环境下的训练与评估图表，避免混入与真实结果无关的材料。

## 对应源码

- `src/training/trainer.py`
- `src/training/evaluator.py`
- `src/reporting/report_builder.py`
