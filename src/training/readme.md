# Training Framework

## Overview

当前训练框架围绕真实雷达 `.mat` 数据构建，核心目标是把真实处理后的 `TR` 与 `DPL` 数据稳定地送入高精度融合模型，并输出可复现的训练、评估与真实图表结果。

当前主模型由四部分组成：

- 官方 `X3D` 风格 `X3D-M` RD 编码器
- `Residual TCN + Transformer Encoder` 轨迹编码器
- 双向 `Cross-Attention` 融合层
- 主分类头与辅助分支头

当前主训练链路固定为：

```text
manage.py prepare-pro
-> data/sessions
-> manage.py train
-> manage.py evaluate
-> manage.py report
```

## Current RD Main Path

当前 RD 主路径是**单尺度** `DPL -> RD`。

默认 `6` 通道输入为：

- `real`
- `imag`
- `log amplitude`
- `temporal difference`
- `background suppressed`
- `local contrast`

这 6 个通道分别负责：

- 保留复数信息
- 压缩动态范围
- 强调时间变化
- 压制背景
- 突出局部谱结构

## RD Encoder

`src/models/cnn_encoder.py` 中的 `X3DMicroMotionEncoder` 当前包含：

- 微动增强层
- 谱轴精炼层
- `background/local contrast` 并行增强支路
- `X3D-M` 风格主干

也就是说，当前 RD 分支不是直接把谱图塞进普通 CNN，而是先做雷达微动导向增强，再做时空编码。

## Track Encoder

`src/models/temporal_encoder.py` 中的 `TrackTemporalEncoder` 组合了：

- 输入投影
- `Residual TCN`
- `Transformer Encoder`
- 注意力池化与统计汇聚

它负责建模轨迹的宏观运动规律，例如速度变化、加速度变化、转向趋势和轨迹抖动。

## Training Loop

`src/training/trainer.py` 当前支持：

- `class-balanced loss`
- `label smoothing`
- `warmup + cosine` 学习率调度
- `gradient clipping`
- `macro-F1` 跟踪
- `session` 级 `train / val / test` 切分
- 按类别分层的 session 划分
- 独立评估与真实图表生成

## Recommended Defaults

当前默认配置包括：

- `sequence_length: 24`
- `rd_frames: 24`
- `rd_height: 64`
- `rd_width: 64`
- `rd_input_channels: 6`

当前真实数据标签来自文件名，因此是 session 级弱监督标签。它可以训练当前模型，但不能等价视为逐帧精标数据集。

## Commands

准备真实数据：

```bash
.\.venv\Scripts\python.exe manage.py prepare-pro
```

训练模型：

```bash
.\.venv\Scripts\python.exe manage.py train --epochs 30 --batch-size 4 --device cpu
```

评估模型：

```bash
.\.venv\Scripts\python.exe manage.py evaluate artifacts/my_run/best_fusion_model.pt --device cpu
```

生成真实图表：

```bash
.\.venv\Scripts\python.exe manage.py report artifacts/my_run
```
