# 27. CVAT 怎么用在这个项目里：一个面向小白的实操思路

## CVAT 能不能直接拿来标这个项目

能用，但需要先理解它的角色。

CVAT 本质上是一个图像 / 视频标注平台，而当前项目的数据原始形式主要是：

- `.mat`
- `CSV`
- `JSONL`
- `HDF5`

所以 CVAT **不能直接拿 `.h5` 或 `.mat` 当标注界面使用**。

更合理的做法是：

- 先把 RD clip 转成图像序列或短视频
- 再把这些可视化结果送进 CVAT 做人工标注

## 对这个项目来说，CVAT 最适合标什么

当前最适合 CVAT 标的是：

- **clip 级 RD 可视化序列**

也就是：

- 把一个 `rd_clips.h5` 里的 clip 导出成 PNG 序列或短视频
- 在 CVAT 中按 clip 或按帧做标签

## 当前仓库里有哪些现成基础

### `tools/visualize_clips.py`

当前仓库已有一个基础可视化脚本：

- `tools/visualize_clips.py`

它当前可以：

- 读取 `radar/rd/rd_clips.h5`
- 输出预览图到 `qc/` 目录

### `export_to_cvat()` 目前还是占位

在：

- `src/labeling/truth_matcher.py`

里有 `export_to_cvat()`，但目前还是 `pass`，说明完整导出 / 回写流程还没接好。

## 推荐的 CVAT 使用方式

### 方案 A：按 clip 做分类标签

这是当前最适合本项目的方式：

1. 从 `rd_clips.h5` 导出 clip 预览图或短视频
2. 每个 clip 作为一个任务送入 CVAT
3. 在 CVAT 中给每个 clip 打类别标签
4. 导出标注结果
5. 再转换回项目自己的标签格式

### 方案 B：按帧做更细粒度标签

如果后面希望更精细，可以进一步在 CVAT 中：

- 按帧标可见区域
- 按帧标目标类别
- 标出目标开始出现、稳定出现、消失的时间段

这种方式更细，但成本也更高。

## 在 CVAT 里应该怎么建项目

建议标签至少定义为：

- `Bird`
- `RotaryDrone`
- `FixedWing`
- `Other`
- `Unknown`

还可以增加属性：

- `confidence`: `high / medium / low`
- `visible_quality`: `clear / weak / noisy`
- `review_status`: `draft / reviewed`

## 当前最推荐的起点

如果你现在就要开始用 CVAT，最推荐：

- 先把 RD clip 导出成图片或视频
- 用 CVAT 做 clip 级分类标签
- 暂时不追求逐帧框选
- 先把 clip 级精标链路跑通

## 对应文件

- `tools/visualize_clips.py`
- `src/labeling/truth_matcher.py`
- `src/prodata/mat_session_builder.py`


CVAT 不是直接拿来读雷达 `.mat` 的，而是拿来标“你先导出的雷达可视化结果”的。