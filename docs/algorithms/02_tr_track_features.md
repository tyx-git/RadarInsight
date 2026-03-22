# 02. TR 轨迹数据与时序特征

## 这一步在做什么

`TR` 文件中包含轨迹相关字段，例如距离、速度、方位、俯仰、SNR 和时间戳。

项目会把这些字段整理成 `track_data.csv`，再从中提取轨迹时序特征。

## 轨迹分支为什么重要

轨迹分支主要负责学习目标的宏观行为模式，例如：

- 速度变化快不快
- 轨迹抖动大不大
- 是否存在停走行为
- 转向是否频繁

这些特征对区分：

- 鸟类盘旋
- 旋翼无人机悬停或侧飞
- 固定翼稳定飞行

很有帮助。

## 当前使用了哪些基础字段

当前会写入 `track_data.csv` 的核心字段包括：

- `t`
- `frame_id`
- `track_id`
- `range_m`
- `azimuth_deg`
- `radial_speed_mps`
- `snr_db`
- `elevation_deg`

## 为什么还要做派生特征

原始值只告诉你“这一帧是什么样”，但模型真正需要的是“它怎么变化”。

所以 `src/training/dataset.py` 里又派生了：

- `delta_range_m`
- `delta_azimuth_deg`
- `speed_delta_mps`
- `acceleration_mps2`
- `turn_rate_degps`
- `range_jitter_m`
- `snr_delta_db`

这些特征能把目标行为描述得更完整。

## 一个直观理解

假设两个目标的平均速度都差不多：

- 目标 A：速度稳定、方位变化平滑
- 目标 B：速度有起伏、方位抖动明显

如果只看 `radial_speed_mps`，它们可能很像。
但如果加上 `speed_delta_mps`、`turn_rate_degps`、`range_jitter_m`，模型就更容易把它们区分开。

## 对应源码

- `src/prodata/mat_session_builder.py`
- `src/training/dataset.py`
