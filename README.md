# RadarInsight

RadarInsight 是一个面向真实雷达 `.mat` 数据的目标识别训练框架。当前仓库**不再直接对接雷达设备**，也**不再保留伪数据展示链路**，而是围绕真实数据处理、标准数据集构建、模型训练、评估与真实报告生成展开。

## Author

TTxyz

## 当前主路径

当前主训练路径固定为：

```text
pro/raw/TR + pro/raw/RD
-> prepare-pro
-> pro/sessions
-> data/sessions
-> train / evaluate / report
```

其中：

- `TR`：提取轨迹时序数据
- `RD`：文件名虽为 RD，但内部核心字段是复数 `DPL`，当前按 TD 类复信号处理后，通过 FFT 生成距离-多普勒谱图
- `data/sessions/`：训练与评估唯一读取的数据根目录

## 当前模型主方案

当前高精度主模型为：

- 轨迹分支：`Residual TCN + Transformer`
- RD 分支：官方 `X3D` 风格 `X3D-M`
- 融合层：双向 `Cross-Attention`

当前默认 RD 主路径是**单尺度** `DPL -> RD`，默认 `6` 通道输入为：

- `real`
- `imag`
- `log amplitude`
- `temporal difference`
- `background suppressed`
- `local contrast`

## 目录职责

- `pro/raw/`：真实原始 `.mat` 数据输入目录
- `pro/sessions/`：真实数据处理后的中间会话目录
- `data/sessions/`：标准训练数据集目录
- `artifacts/`：checkpoint、history、评估结果输出目录
- `reports/real/`：真实训练与评估图表
- `src/`：数据处理、模型、训练与评估源码
- `docs/algorithms/`：面向初学者的算法讲解文档
- `needtosettle/`：暂不进入主链路的历史模块与待定内容

## 原始数据与标签

当前真实数据位于：

- `pro/raw/TR/*.mat`
- `pro/raw/RD/*.mat`

当前标签直接从文件名提取并标准化为：

- `Bird`
- `RotaryDrone`
- `FixedWing`

需要明确：当前标签属于**session 级弱监督标签**，不是人工逐帧精标。

## 标准数据集结构

训练最终读取的是 `data/sessions/<session_id>/`，典型结构如下：

```text
data/sessions/<session_id>/
├─ meta_session.json
├─ radar/
│  ├─ track/track_data.csv
│  └─ rd/
│     ├─ rd_data.h5
│     └─ rd_clips.h5
├─ labels/
│  ├─ labels_trackclip.jsonl
│  └─ labels_matched.jsonl
└─ qc/
   └─ qc_report.json
```

训练最关键的输入文件是：

- `radar/track/track_data.csv`
- `radar/rd/rd_clips.h5`
- `labels/labels_trackclip.jsonl`
- `labels/labels_matched.jsonl`

## 真实数据准备

### 1. 安装依赖

```bash
.\.venv\Scripts\pip.exe install -r requirements.txt
```

### 2. 放入真实 `.mat` 数据

目录结构应为：

```text
pro/raw/
├─ TR/*.mat
└─ RD/*.mat
```

### 3. 构建标准 session

```bash
.\.venv\Scripts\python.exe manage.py prepare-pro
```

该命令会：

- 自动配对 `TR/RD`
- 解析文件名提取标签
- 从 `TR` 生成轨迹表
- 从 `DPL` 生成 RD 帧与 RD clip
- 生成标准 session
- 复制到 `data/sessions/`

如果只处理部分 session：

```bash
.\.venv\Scripts\python.exe manage.py prepare-pro 20230705160021_rotarydrone_13_6837
```

## 查看会话状态

```bash
.\.venv\Scripts\python.exe manage.py sessions --include-rd
```

它会输出：

- 已可训练的 session
- 每个 session 的 `track clips` 数量
- 标签覆盖情况

## 训练模型

训练始终从 `data/sessions/` 读取标准数据集。

### 全量训练

```bash
.\.venv\Scripts\python.exe manage.py train --epochs 30 --batch-size 4 --device cpu
```

### 指定 session 训练

```bash
.\.venv\Scripts\python.exe manage.py train 20230705160021_rotarydrone_13_6837 20230930155519_bird_03_39 --epochs 10 --batch-size 2 --device cpu
```

### 常用参数

- `--epochs`：训练轮数
- `--batch-size`：批大小
- `--learning-rate`：学习率
- `--num-workers`：数据加载线程数
- `--device`：训练设备，如 `cpu` 或 `cuda`
- `--output-dir`：checkpoint 与历史输出目录

## 训练切分策略

当前训练框架已经固定为：

- 按 `session` 级切分 `train / val / test`
- 按类别做分层切分
- 避免同一 session 样本同时进入训练集与测试集
- 自动导出 `training_session_manifest.json`

## 评估模型

```bash
.\.venv\Scripts\python.exe manage.py evaluate artifacts/my_run/best_fusion_model.pt --device cpu
```

如果训练目录中存在 `training_session_manifest.json`，评估会优先读取其中的 `test_sessions`。

## 生成真实图表

```bash
.\.venv\Scripts\python.exe manage.py report artifacts/my_run
```

## GitHub 发布建议

当前仓库已经按发布友好的方式整理：

- 真实原始数据放在 `pro/raw/`
- 中间处理结果放在 `pro/sessions/`
- 训练数据集放在 `data/sessions/`
- 训练产物放在 `artifacts/`
- 真实图表放在 `reports/real/`

以上目录都属于本地运行数据，默认不应提交到 GitHub。

当前报告系统**只保留真实环境图表**，输出到：

- `reports/real/`

## 配置文件

统一配置文件是 `config/system_config.yaml`。

当前最重要的两部分：

- `pro_processing`：控制 `pro/raw -> pro/sessions -> data/sessions`
- `training`：控制模型训练参数与输入尺寸

重点配置包括：

- `pro_processing.range_scale_to_meter`
- `pro_processing.doppler_window`
- `pro_processing.doppler_fft_size`
- `training.sequence_length`
- `training.rd_frames`
- `training.rd_height`
- `training.rd_width`
- `training.rd_input_channels`

## 当前状态

当前仓库已经具备：

- 真实 `.mat` 数据接入
- `TR/RD` 自动配对
- `DPL -> RD` 谱图生成
- 标准 session 数据集生成
- 会话级分层训练切分
- 真实评估与真实图表生成

如果你要继续深入理解项目，建议继续阅读：

- `explain.md`
- `mdopt.md`
- `src/training/readme.md`
- `docs/algorithms/README.md`
