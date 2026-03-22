# 05. X3D-M 风格 RD 主骨干

## 这部分在做什么

RD 主骨干负责把一段 RD clip 序列编码成高层时空特征。

当前项目里，这个主骨干是官方 `X3D` 风格的 `X3D-M`，位置在：

- `src/models/cnn_encoder.py`

## 为什么不是普通 2D CNN

如果只用普通 2D CNN，模型通常只能看到“单帧纹理”。

但 RD 微动识别真正关心的是：

- 条纹是否随时间持续变化
- 谐波是否周期性出现
- 局部峰值是否稳定存在

所以当前采用的是 3D CNN，把：

- 时间维
- 距离维
- 多普勒维

一起建模。

## 当前输入不是原始单通道图

当前默认主路径下，`X3D-M` 并不是直接接收一张单通道幅度图，而是接收经过数据集层构造的 `6` 通道 RD 表示：

- `real`
- `imag`
- `log amplitude`
- `temporal difference`
- `background suppressed`
- `local contrast`

也就是说，主干前面已经有一层“更懂雷达”的表示构造。

## 当前结构可以怎么理解

当前 `src/models/cnn_encoder.py` 里的主链大致分成四段：

1. `RadarMicroMotionEnhancer`
2. `SpectralAxisRefiner`
3. `BackgroundContrastRefiner`
4. `X3D-M` 风格主干

### 1. `RadarMicroMotionEnhancer`

先把更像微动的结构强化出来，例如：

- 谐波相关模式
- 时间变化结构
- 多普勒方向敏感结构

### 2. `SpectralAxisRefiner`

进一步强调雷达谱图的物理结构，尤其是：

- 距离轴方向变化
- 多普勒轴方向变化
- 时间与多普勒的联合结构

### 3. `BackgroundContrastRefiner`

在进入主干前，再从两个视角补一次增强：

- 背景抑制视角
- 局部对比视角

### 4. `X3D-M` 主干

这才是真正负责做深层时空特征提取的部分。它会把前面增强后的特征逐层编码成更高层的判别表示。

## 最后输出是什么

当前 RD 主骨干最后会输出两类结果：

- `temporal tokens`
- `pooled feature`

前者进入跨模态融合层，后者进入 RD 辅助头。

## 为什么这条链路现在更强

因为它不再是“通用 3D CNN 直接吃谱图”，而是：

- 输入层先保留更多对微动有价值的信息
- 主干前先做雷达导向增强
- 再让 `X3D-M` 做深层时空编码

这比最初的轻量 CNN 基线强得多，也更适合当前真实数据环境。

## 对应源码

- `src/models/cnn_encoder.py`

## 初学者最该记住的一句话

`X3D-M` 在这个项目里不是孤立工作的，它前面已经接了一整套雷达微动增强链路。
