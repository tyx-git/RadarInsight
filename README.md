﻿# RadarInsight
## 项目简介
RadarInsight 是一个面向真实雷达 `.mat` 数据的目标识别训练框架。当前项目不直接连接雷达硬件，而是基于雷达信号处理后的 `TR` 与 `RD` 数据开展数据整理、标准数据集构建、模型训练、独立评估和真实图表生成。
## Author
TTxyz
## 当前主链路
项目当前固定主链路如下：
```text
pro/raw/TR + pro/raw/RD
-> prepare-pro
-> pro/sessions
-> data/sessions
-> train / evaluate / report / calibrate
```
其中：`TR` 提供轨迹时序信息；`RD` 文件内部核心为复数 `DPL` 数据，后续通过 `DPL -> RD` 处理链生成可训练的距离-多普勒表示；训练和评估只读取 `data/sessions/`。
## 当前主模型
当前最优主模型由四部分组成：
- 轨迹分支：`Residual TCN + Transformer`
- RD 分支：目标对齐局部 ROI + 全局上下文 + `X3D-M` 风格 3D CNN
- 融合层：双向 `Cross-Attention` + 模态可靠性门控
- 输出层：粗细粒度层级分类 + 细分类余弦分类头 + 组内校准
这条路线的目标不是贴合当前小样例数据集，而是服务未来 `3万+` 样本、`5` 大类、`20+` 子类数据集。
## 目录职责
- `pro/raw/`：原始 `.mat` 数据输入目录
- `pro/sessions/`：数据处理阶段的中间 session 目录
- `data/sessions/`：训练与评估唯一读取的数据集目录
- `src/`：数据处理、模型、训练、评估源码
- `docs/algorithms/`：算法、标注、泛化、调试等讲解文档
- `artifacts/`：checkpoint、训练历史、评估结果、校准产物
- `reports/real/`：真实图表输出目录
- `debug.md`：架构演进、问题分析和迭代记录
## 数据与标签
当前真实数据放在：
- `pro/raw/TR/*.mat`
- `pro/raw/RD/*.mat`
当前标签从文件名提取并标准化为：
- `Bird`
- `FixedWing`
- `RotaryDrone`
需要注意：当前标签仍属于 `session` 级弱监督标签，不是逐帧人工精标。项目后续会面向更大规模、多主类、多子类数据集演进。
## 标准数据集结构
训练最终读取的是 `data/sessions/<session_id>/`。典型结构如下：
```text
data/sessions/<session_id>/
├─ meta_session.json
├─ labels/
│  ├─ labels_trackclip.jsonl
│  └─ labels_matched.jsonl
├─ qc/
│  └─ qc_report.json
└─ radar/
   ├─ track/track_data.csv
   └─ rd/
      ├─ rd_data.h5
      ├─ rd_clips.h5
      └─ rd_alignment_diagnostics.jsonl
```
## 核心命令
### 1. 从原始数据构建标准数据集
```bash
.\.venv\Scripts\python.exe manage.py prepare-pro
```
如果只想处理指定 session，可以在命令后追加 `session_id` 列表。
### 2. 查看当前可训练 session
```bash
.\.venv\Scripts\python.exe manage.py sessions --include-rd
```
### 3. 训练模型
```bash
.\.venv\Scripts\python.exe manage.py train --epochs 5 --batch-size 4 --device cpu
```
如果不手动传 `--output-dir`，输出目录会自动按以下格式命名：
`YYYYMMDDHHMM-epochs-batch-device`
例如：`202603252213-5-4-cpu`
### 4. 独立评估 checkpoint
```bash
.\.venv\Scripts\python.exe manage.py evaluate artifacts/my_run/best_fusion_model.pt --device cpu
```
### 5. 生成真实图表
```bash
.\.venv\Scripts\python.exe manage.py report artifacts/my_run
```
### 6. 进行组内校准
```bash
.\.venv\Scripts\python.exe manage.py calibrate artifacts/my_run/best_fusion_model.pt --output-dir artifacts/calibrated/my_run-calibrated --device cpu
```
## 常用训练参数
- `--epochs`：训练轮数
- `--batch-size`：批大小
- `--learning-rate`：学习率
- `--num-workers`：数据加载进程数
- `--device`：训练设备，如 `cpu` 或 `cuda`
- `--output-dir`：训练产物输出目录
- `--include-rd / --no-rd`：是否启用 RD 分支
## 训练切分策略
当前训练框架固定为：
- 按 `session` 级别切分 `train / val / test`
- 按类别做分层切分
- 避免同一 session 的 clip 同时出现在训练集和测试集
- 自动导出 `training_session_manifest.json`
这一步对泛化能力非常关键，因此不能改成随机 clip 打散切分。
## 图表与报告
当前项目只保留真实结果输出：
- 图表输出目录：`reports/real/`
- 报告输入目录：`artifacts/<run_name>/`
当前不再保留 fake 数据、fake 报告和相关代码链路。
## GitHub 发布建议
以下目录属于本地运行数据，默认不应提交到 GitHub：
- `pro/raw/`
- `pro/sessions/`
- `data/sessions/`
- `artifacts/`
- `reports/real/`
这些目录保留在本地即可，仓库应主要提交源码、配置和文档。
## 配置文件
统一配置文件是 `config/system_config.yaml`。当前最重要的两部分是：
- `pro_processing`：控制 `pro/raw -> pro/sessions -> data/sessions`
- `training`：控制模型训练参数、输入尺寸和损失配置
重点配置包括：
- `pro_processing.range_scale_to_meter`
- `pro_processing.doppler_window`
- `pro_processing.doppler_fft_size`
- `training.sequence_length`
- `training.rd_frames`
- `training.rd_height`
- `training.rd_width`
- `training.rd_input_channels`
- `training.detail_boundary_loss_weight`
- `training.detail_cosine_scale`
## 当前仓库已经具备的能力
- 真实 `.mat` 数据接入
- `TR/RD` 自动配对
- `DPL -> RD` 转换
- 标准 session 数据集生成
- `session` 级分层训练切分
- 独立评估与真实图表生成
- 组内校准与精度修正
- 面向初学者的系统算法文档
## 推荐阅读入口
如果你想快速理解项目，建议按这个顺序阅读：
- `explain.md`
- `mdopt.md`
- `src/training/readme.md`
- `docs/algorithms/README.md`
- `docs/algorithms/LinkAll.md`
- `docs/algorithms/31_generalization_learning_in_this_project.md`
- `docs/algorithms/32_how_to_judge_real_generalization_improvement.md`
- `docs/algorithms/33_what_is_a_session_and_why_it_matters.md`
- `docs/algorithms/34_session_clip_track_sample_label_relationships.md`
- `docs/algorithms/35_how_to_read_training_history_and_choose_the_next_experiment.md`
- `docs/algorithms/36_current_best_model_full_explanation.md`
- `docs/algorithms/37_training_parameters_and_metrics_explained.md`
- `docs/algorithms/38_how_202603252213_5_4_cpu_calibrated_model_works.md`


## Real Excel Track Pipeline
Use `prepare-tsv` to convert paired Excel files under `pro/excel` into standardized TSV sessions under `pro/tsv`, then sync them into `data/tsv` for training.

The current default training route is the track-only TSV pipeline under `data/tsv`.



## Environment Variables
??????????????????????????
- `TRACK=true|false`
  - ???????????
  - ????`true`?
- `RD=true|false`
  - ?????? RD ???
  - ????`true`?
- `TRACK=true RD=true`
  - ????????? RD ???
- `TRACK=true RD=false`
  - ????????
- `TRACK=false RD=true`
  - ??? RD ???
- `TRACK=false RD=false`
  - ????????????????
- ????????? Excel ??????????????????????????????????? RD ?????????

## ????????
???? TSV ?????????? A ???????`HumanTarget / BirdTarget / DroneTarget / AircraftTarget / SurfaceTarget / UnknownTarget / NoiseTarget`?

????????? Excel ????????????????????????? `data/tsv`?
- ???
  - `set TRACK=true && set RD=false && .\.venv\Scripts\python.exe manage.py train --data-root data/tsv --epochs 5 --batch-size 4 --device cpu`
- ???
  - `set TRACK=true && set RD=false && .\.venv\Scripts\python.exe manage.py evaluate artifacts\<run>\best_fusion_model.pt --data-root data/tsv --device cpu`
- ???
  - `set TRACK=true && set RD=false && .\.venv\Scripts\python.exe manage.py calibrate artifacts\<run>\best_fusion_model.pt --data-root data/tsv --device cpu`
- ???????
  - `set TRACK=true && set RD=false && .\.venv\Scripts\python.exe manage.py report artifacts\<run>`
- ????????
  - `set TRACK=true && set RD=false && .\.venv\Scripts\python.exe manage.py sessions --data-root data/tsv`

