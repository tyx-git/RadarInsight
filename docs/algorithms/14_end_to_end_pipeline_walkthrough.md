# 14. 全链路 walkthrough：从 `.mat` 到最终分类结果，中间到底发生了什么

## 第 1 步：原始数据进入 `pro/raw`

项目当前不直接对接雷达设备，而是读取已经完成雷达信号处理后的 `.mat` 文件。

原始目录分成两部分：

- `pro/raw/TR`：轨迹相关数据
- `pro/raw/RD`：实际交接时命名为 RD，但核心字段是 `DPL` 复信号

这里最重要的一点是：

**训练时不直接读取 `pro/raw`，而是先做一次标准化处理。**

## 第 2 步：`prepare-pro` 把原始数据整理成标准 session

入口：`manage.py prepare-pro`

它做的事包括：

1. 扫描 TR / RD 文件并自动配对。
2. 从文件名提取类别标签。
3. 从 TR 提取轨迹表。
4. 从 `DPL` 生成 RD 帧序列。
5. 生成 clip 标签、RD clip、质检结果。
6. 把标准 session 复制到 `data/sessions`。

处理完以后，每个 session 都会变成统一结构，后续训练只需要认这个标准结构。

## 第 3 步：`data/sessions` 成为训练入口

为什么 `pro` 和 `data` 要分开？

因为两者职责不同：

- `pro` 负责数据处理和清洗。
- `data` 负责给训练框架提供稳定输入。

这样做虽然看起来有一点冗余，但目录边界非常清楚。

## 第 4 步：Dataset 把 session 读成模型可用样本

位置：`src/training/dataset.py`

在这一层，一个样本会被整理成：

- `track_features`
- `rd_features`
- `label`
- `has_rd`

也就是说，Dataset 的任务不是“存数据”，而是“把标准文件变成张量”。

## 第 5 步：轨迹分支提取宏观运动特征

位置：`src/models/temporal_encoder.py`

轨迹分支重点回答的问题是：

- 这个目标整体怎么动？
- 速度、加速度、转向和抖动有什么规律？

它更偏行为建模。

## 第 6 步：RD 分支提取微动特征

位置：`src/models/cnn_encoder.py`

RD 分支重点回答的问题是：

- 这个目标有没有稳定的微动纹理？
- 谐波、侧带、局部峰值结构像什么？
- 时间上这些结构是否持续存在？

它更偏微动建模。

## 第 7 步：融合层把宏观行为和微动细节放在一起看

位置：`src/models/fusion_classifier.py`

当前不是简单拼接，而是双向 `Cross-Attention`。

所以最终分类不是：

- 只看轨迹
- 或只看 RD

而是：

- 让两边互相解释，再做最终判断

## 第 8 步：训练器负责真正把模型训起来

位置：`src/training/trainer.py`

训练器负责：

- 构建 dataloader
- 按 session 分层切分 train / val / test
- 计算主损失和辅助损失
- 保存 best checkpoint
- 记录 history 和评估报告

## 第 9 步：评估器单独验证模型

位置：`src/training/evaluator.py`

训练结束后，还会单独读取 best checkpoint 再跑一次评估。

这一步很重要，因为它能避免“训练过程中最后一轮内存状态”和“最终 checkpoint”不一致的问题。

## 第 10 步：报告模块把结果画成图表

位置：`src/reporting/report_builder.py`

图表不是模型的一部分，但它对理解模型是否真的变好了非常重要。

常见图表包括：

- 训练曲线
- 混淆矩阵
- 各类别指标图
- 总体指标图

## 你应该怎么用这篇文档

如果你只记不住一堆模块名，那就记住下面这条主线：

```text
TR / DPL mat
 -> prepare-pro
 -> standardized session folders
 -> dataset tensors
 -> track branch + RD branch
 -> cross-attention fusion
 -> classifier
 -> train / evaluate / report
```


这个项目不是“一个模型文件”在工作，而是一条从原始雷达 `.mat` 到最终分类报告的完整算法链。
