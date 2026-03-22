# 13. RD 强建模分阶段理解

## 为什么要分阶段看

因为当前 RD 分支不是“换了一个更大的 CNN”这么简单，而是沿着同一条主线逐层增强。

如果不按阶段理解，很容易看花：

- 输入表示在增强
- 主干前置模块在增强
- 主干内部在做深层编码
- 融合层还在继续利用 RD 信息

所以这篇文档专门按阶段拆开。

## 阶段 1：把原始 RD 变成更适合学习的 6 通道张量

位置：`src/training/dataset.py`

默认主路径的 `6` 通道是：

- `real`
- `imag`
- `log amplitude`
- `temporal difference`
- `background suppressed`
- `local contrast`

这一阶段的目标是：

- 保留复数信息
- 强调时间变化
- 压制背景
- 突出局部结构

## 阶段 2：微动增强层 `RadarMicroMotionEnhancer`

位置：`src/models/cnn_encoder.py`

这一层的作用是先把更像雷达微动的线索提亮出来，例如：

- 谐波结构
- 时间变化结构
- 多普勒方向敏感结构

你可以把它理解成“先做一次雷达专用预增强”。

## 阶段 3：谱轴精炼层 `SpectralAxisRefiner`

位置：`src/models/cnn_encoder.py`

这一层进一步强调：

- 距离轴方向变化
- 多普勒轴方向变化
- 时间-多普勒联合模式

这是把雷达谱图的物理结构显式写进网络的一步。

## 阶段 4：背景 / 局部对比并行支路 `BackgroundContrastRefiner`

位置：`src/models/cnn_encoder.py`

这一层让主干同时看到两个更有针对性的视角：

- 背景抑制后的视角
- 局部对比突出的视角

它的意义不是替代主干，而是为主干补充更清晰的目标结构表达。

## 阶段 5：X3D-M 风格主干做深层时空编码

位置：`src/models/cnn_encoder.py`

前面几步都还是“增强”，真正完成高层判别特征提取的是这里的 `X3D-M` 风格主干。

它会把前面增强后的 RD 特征编码成：

- RD token
- RD pooled feature

然后分别送给：

- 融合层
- RD 辅助头

## 当前这条链路说明了什么

当前 RD 强建模已经不是早期那种“幅度图 + 普通 CNN”的简单方案，而是：

- 先做复数感知输入表示
- 再做雷达导向增强
- 再做深层时空编码

这就是当前默认主路径的完整思路。

## 对应源码

- `src/training/dataset.py`
- `src/models/cnn_encoder.py`
- `mdopt.md`

## 初学者最该记住的一句话

RD 强建模不是单独某一层的工作，而是从输入表示到前置增强，再到时空主干，整条链一起优化。
