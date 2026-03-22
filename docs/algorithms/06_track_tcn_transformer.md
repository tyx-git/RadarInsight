# 06. 轨迹分支：Residual TCN + Transformer

## 这部分为什么存在

RD 分支负责微动识别，但很多类别并不能只靠微动分开。

例如：

- 固定翼和旋翼在行为模式上通常不一样
- 鸟类和无人机在轨迹稳定性上通常也不一样

所以还需要轨迹分支去建模宏观运动规律。

## 为什么不用简单 RNN

早期版本用过 `GRU`，但它的表达能力有限。

现在改成：

```text
Feature Projection -> Residual TCN -> Transformer Encoder -> Attention Pooling
```

## 各部分分别做什么

### 输入投影

把低维轨迹特征映射到更适合深层网络处理的空间。

### Residual TCN

它更擅长抓局部时间模式，例如：

- 速度突然变化
- 转向突然变化
- 抖动节律

### Transformer Encoder

它更适合抓长时间依赖，例如：

- 整段轨迹的整体行为模式
- 关键时间点之间的关系

### Attention Pooling

不是简单平均所有时刻，而是让模型自动学会：

- 哪些时间片更重要
- 哪些片段更能代表目标行为

## 结果

轨迹分支现在不是简单辅助分支，而是能提供一条完整的行为特征链路。

## 对应源码

- `src/models/temporal_encoder.py`
