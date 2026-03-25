# Training Framework

## Overview

当前训练框架围绕真实雷达 `.mat` 数据构建，核心目标是把真实处理后的 `TR` 与 `DPL` 数据稳定地送入高精度融合模型，并输出可复现的训练、评估与真实图表结果。

当前主模型由五部分组成：

- `X3D-M` 风格 RD 编码器
- `Residual TCN + Transformer Encoder` 轨迹编码器
- 双向 `Cross-Attention` 融合层
- 面向未来 `5` 大类、`20+` 子类扩展的层级式表示头
- 粗类映射 + 细粒度边界监督 + `detail logits` 残差注入
- 基于验证集的组内 logit 校准

当前主训练链路固定为：

```text
manage.py prepare-pro
-> data/sessions
-> manage.py train
-> manage.py evaluate
-> manage.py report
```

## Current RD Main Path

当前 RD 主路径仍然是**单尺度主链优先**的 `DPL -> RD`，但训练输入已经同时保留：

- 对齐后的局部目标 ROI
- 同一时间窗内的全局 RD 上下文
- 更丰富的复数/相位/结构特征通道

当前默认 `12` 通道输入优先覆盖：

- `real`
- `imag`
- `log amplitude`
- `temporal difference`
- `background suppressed`
- `local contrast`
- `local doppler symmetric`
- `phase difference`
- `coherence`
- `global log amplitude`
- `global background suppressed`
- `global doppler symmetric`

## RD Encoder

`src/models/cnn_encoder.py` 中的 `X3DMicroMotionEncoder` 当前包含：

- 微动增强层
- 谱轴精炼层
- `background/local contrast` 增强支路
- 显式的谱结构增强支路：
  - `RotorMicroMotionBranch`
  - `FixedWingStructureBranch`
- `SpectralStructureEnhancer`
- `X3D-M` 风格主干
- 结构嵌入 token 注入

也就是说，当前 RD 分支不再只是“把谱图送进 CNN”，而是显式把“旋翼微动结构”和“固定翼稳定谱脊结构”作为可学习先验送入主干。

## Track Encoder

`src/models/temporal_encoder.py` 中的 `TrackTemporalEncoder` 当前包含：

- 输入投影
- `Residual TCN`
- `Transformer Encoder`
- 显式运动模式分支：
  - 连续性分支
  - 平滑性分支
  - 机动模式分支
- 全局统计特征门控融合
- 模式 token 注入与最终汇聚

当前轨迹分支不再只依赖通用时序编码，而是显式建模：

- 持续性
- 平滑性
- 机动强度与变化方式

## Hierarchical Representation And Boundary Supervision

当前融合头已经引入层级式表示拆分：

- `coarse feature`：负责更稳定的粗类语义分离
- `detail feature`：负责粗类内部的细粒度边界表达
- `coarse logits`：监督粗类级稳定性
- `detail logits`：监督粗类内部的细粒度边界
- 最终 `logits`：由 `coarse + detail` 共同决定，并叠加 `detail logits` 残差

当前配置中通过 `coarse_label_map` 指定粗类映射。当前公开数据集默认是：

- `Bird -> BiologicalTarget`
- `FixedWing -> AircraftTarget`
- `RotaryDrone -> AircraftTarget`

因此当前细粒度边界监督的核心目标，就是在 `AircraftTarget` 内部稳定拉开 `FixedWing` 与 `RotaryDrone`。

后续接入真实大规模数据集时，可以把 `coarse_label_map` 扩展为 `5` 大类，把现有 `detail feature` 和 `detail logits` 自然扩展到 `20+` 子类，而不需要推倒当前主干重来。

## Training Loop

`src/training/trainer.py` 当前支持：

- `class-balanced loss`
- `label smoothing`
- `warmup + cosine` 学习率调度
- `gradient clipping`
- `macro-F1` 跟踪
- `session` 级 `train / val / test` 切分
- 按类别分层的 session 划分
- 轨迹辅助头、RD 辅助头、粗类辅助头、细粒度边界辅助头联合训练
- 粗类组内 hardest-negative 边界 margin 约束
- `detail logits` 残差注入最终输出
- 基于验证集的粗类内 logit bias 校准
- 独立评估与真实图表生成

## Recommended Defaults

当前默认配置包括：

- `sequence_length: 16`
- `rd_frames: 24`
- `rd_height: 64`
- `rd_width: 64`
- `rd_input_channels: 12`
- `detail_boundary_loss_weight: 0.2`
- `detail_logit_residual_weight: 0.2`

当前公开雷达数据集用于算法训练与结构验证，不等同于最终真实部署环境数据集。

当前真实工程目标不是为三类闭集打补丁，而是逐步验证：

- 哪类表示设计能稳定提升真实泛化能力
- 哪类层级结构能自然扩展到未来约 `33 万` 数据规模、`5` 大类、`20+` 子类

## Commands

准备真实数据：

```bash
python3 manage.py prepare-pro
```

训练模型：

```bash
python3 manage.py train --epochs 5 --batch-size 4 --device cuda
```

评估模型：

```bash
python3 manage.py evaluate artifacts/my_run/best_fusion_model.pt --device cuda
```

生成真实图表：

```bash
python3 manage.py report artifacts/my_run
```

校准模型：

```bash
python3 manage.py calibrate artifacts/my_run/best_fusion_model.pt --device cuda
```
