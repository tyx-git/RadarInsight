# 12. 项目里的 Transformer 骨干到底在做什么

## 先把 Transformer 想成什么

Transformer 最核心的能力不是“记忆”，而是“建立关系”。

它会问一个问题：

**当前这个时间点，应该重点参考序列中的哪些其他位置？**

这就是注意力机制的本质。

## 在本项目里，Transformer 不只出现一次

项目里和 Transformer 思想直接相关的地方有两处：

1. 轨迹分支里的 `Transformer Encoder`
2. 融合层里的双向 `Cross-Attention`

这两者都属于 Transformer 家族，只是用途不同。

## 轨迹分支里的 Transformer：做自注意力

在 `src/models/temporal_encoder.py` 里，轨迹 token 会送进 `Transformer Encoder`。

这里的作用是：

- 让每个时间点都能参考整段轨迹其他时间点。
- 让模型自动决定哪些时间段之间关系最重要。
- 把局部运动模式拼成全局行为模式。

你可以把它理解成：

- `TCN` 先看局部片段
- `Transformer` 再看整段关系

## 融合层里的 Cross-Attention：做跨模态注意力

在 `src/models/fusion_classifier.py` 里，项目不是把轨迹向量和 RD 向量直接拼接，而是让两种模态先“互相提问”。

### 轨迹问 RD

轨迹 token 会问：

- 在我当前看到的这段运动行为背景下，RD 微动里哪些片段最关键？

### RD 问轨迹

RD token 会问：

- 在我当前看到的这些微动结构下，轨迹中哪些时间片能解释它？

所以它不是简单加权，而是真正的信息交互。

## 为什么这比简单拼接强

简单拼接只是在最后把两个向量堆起来。

这意味着：

- 轨迹和 RD 彼此看不到对方的细节。
- 模型只能在最后一层“勉强融合”。

而 Cross-Attention 是在更前面就让两边开始交换信息。

对于高精度任务，这通常更强。

## 你不需要死记的数学，但要懂的三个词

### Query

谁在提问。

### Key

可供匹配的索引。

### Value

真正要取回来的信息。

在本项目里，你可以简单理解成：

- Query：我现在最想知道什么
- Key：我应该去哪里找
- Value：我最终取回来的有用内容

## 为什么项目没有直接用纯 Transformer 做整个主干

因为当前真实数据规模还不够大，且 RD 数据有很强的局部纹理和频谱结构。

在这种条件下，直接上纯 Transformer 往往：

- 对数据规模更敏感
- 训练更难稳定
- 不一定比成熟 CNN/TCN 骨干更划算

所以项目当前采用的是更稳的组合：

- RD：CNN / X3D 风格主干
- 轨迹：TCN + Transformer
- 融合：Cross-Attention

## 对应源码

- `src/models/temporal_encoder.py`
- `src/models/fusion_classifier.py`


项目里的 Transformer 不负责“替代所有骨干”，而是负责把序列内部关系和跨模态关系建模得更聪明。
