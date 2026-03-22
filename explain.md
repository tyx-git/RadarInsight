# RadarInsight 项目讲解

这份文档面向第一次接触本仓库的同学。目标不是逐行翻译代码，而是帮助你快速建立全局地图：项目现在处理什么数据、数据怎么流动、每个目录负责什么、当前主模型为什么这样设计。

## 1. 项目现在在做什么

当前项目已经完全围绕**真实雷达信号处理后的 `.mat` 数据**展开，不再以伪数据或雷达直连接口为主。

主链路现在是：

```text
pro/raw/TR + pro/raw/RD
-> prepare-pro
-> pro/sessions
-> data/sessions
-> train / evaluate / report
```

这里的两类核心输入是：

- `TR`：提取轨迹时序信息
- `RD`：文件名虽为 RD，但内部核心字段是复数 `DPL`，当前按 TD 类复信号处理，再通过 FFT 生成距离-多普勒谱图

标签当前直接从文件名提取，因此是**session 级弱监督标签**，不是人工逐帧精标。

## 2. 为什么要分 `pro` 和 `data`

这是当前工程结构里非常重要的一点。

- `pro/`：负责真实数据处理
- `data/`：负责提供稳定训练数据集

也就是说：

- `pro/raw/` 放原始 `.mat`
- `pro/sessions/` 放处理中间结果
- `data/sessions/` 放训练真正读取的标准数据集

这样做的好处是：

- 数据处理和模型训练边界清楚
- 重跑处理流程不会直接污染训练根目录
- 训练代码永远只依赖统一格式的数据集

## 3. 目录职责

### `pro/`

当前只负责真实数据处理。

- `pro/raw/TR/`：原始 `TR` `.mat`
- `pro/raw/RD/`：原始 `RD` 命名 `.mat`，内部是 `DPL` 复数数据
- `pro/sessions/`：处理后的中间会话目录

### `data/`

`data/sessions/` 是最终标准训练数据集目录。

训练与评估都只从这里读取，典型 session 会包含：

- `track_data.csv`
- `rd_data.h5`
- `rd_clips.h5`
- `labels_trackclip.jsonl`
- `labels_matched.jsonl`
- `qc_report.json`

### `src/`

这里是主代码区，最值得优先看的部分包括：

- `src/prodata/mat_session_builder.py`
- `src/training/dataset.py`
- `src/models/cnn_encoder.py`
- `src/models/temporal_encoder.py`
- `src/models/fusion_classifier.py`
- `src/training/trainer.py`
- `src/training/evaluator.py`

### `reports/real/`

这里只保留真实训练与评估图表。

### `needtosettle/`

这里存放当前不进入主链路的历史模块和待定内容，例如旧雷达接入接口。

## 4. 原始数据是怎么进入训练链路的

### 第一步：配对 `TR/RD`

这一步在 `src/prodata/mat_session_builder.py` 中完成。

它会：

- 扫描 `pro/raw/TR/*.mat`
- 扫描 `pro/raw/RD/*.mat`
- 按文件名一一配对
- 从文件名中提取并标准化标签：
  - `Bird`
  - `RotaryDrone`
  - `FixedWing`

### 第二步：从 `TR` 提取轨迹时序

当前会读取这些关键字段：

- `R_m`
- `V_m`
- `A_m`
- `E_m`
- `SNR`
- `DATA_time`
- `Iframecnt`

最终会整理成 `track_data.csv`，并统一使用米等标准单位进入训练。

### 第三步：从 `DPL` 生成 RD 图序列

当前默认主路径是**单尺度** `DPL -> RD`：

- 先沿距离维做 FFT
- 再沿多普勒维做 FFT
- 生成 RD 帧序列
- 再切成 `rd_clips.h5`

这一步的目标不是生成“好看图片”，而是生成适合微动识别的时空输入。

## 5. 标准 session 为什么重要

标准 session 是整个训练框架的统一输入协议。

一旦进入 `data/sessions/<session_id>/`，后续训练代码就不再关心原始 `.mat` 长什么样，只关心：

- 轨迹表有没有
- RD clip 有没有
- 标签文件有没有
- 质检报告有没有

这让整个工程非常稳定。

## 6. 当前模型结构

当前主模型仍然保持高精度路线：

- RD 主干：`X3D-M` 风格 3D CNN
- 轨迹主干：`Residual TCN + Transformer`
- 融合层：双向 `Cross-Attention`

也就是说，项目核心不是单模态分类，而是：

**轨迹宏观行为 + RD 微动细节** 的融合识别。

## 7. 轨迹分支在做什么

轨迹分支负责建模宏观运动规律，例如：

- 速度变化
- 加速度变化
- 转向趋势
- 轨迹抖动

当前它并不是简单看原始轨迹点，而是结合派生特征后，再通过：

- `Residual TCN`
- `Transformer Encoder`
- `Attention Pooling`

完成时序建模。

## 8. RD 分支在做什么

RD 分支负责建模微动特征，例如：

- 谐波结构
- 侧带结构
- 局部峰值
- 时间上的持续微动模式

当前默认单尺度主路径使用的 6 通道是：

- `real`
- `imag`
- `log amplitude`
- `temporal difference`
- `background suppressed`
- `local contrast`

并且在进入 `X3D-M` 前，当前还做了多级 RD 强建模：

- `RadarMicroMotionEnhancer`
- `SpectralAxisRefiner`
- `BackgroundContrastRefiner`

所以当前 RD 分支已经不是“普通 CNN 看图”，而是带有明显雷达微动先验的时空建模链路。

## 9. 为什么要用双向 Cross-Attention

因为轨迹和 RD 各自只看到问题的一半。

- 轨迹更擅长看宏观行为
- RD 更擅长看微动细节

如果只做简单拼接，两个模态之间没有真正交流。

双向 `Cross-Attention` 的作用是：

- 让轨迹去查询 RD 中哪些片段最关键
- 让 RD 反过来参考轨迹上下文

最后形成更稳的融合判断。

## 10. 当前训练和评估怎么保证更可信

当前训练框架有几个关键工程决策：

- 按 `session` 切分 `train / val / test`
- 按类别做分层 session 切分
- 使用 `class-balanced loss`
- 使用 `label smoothing`
- 使用 `warmup + cosine scheduler`
- 使用 `gradient clipping`
- 使用独立评估器重新读取 best checkpoint 做测试

这些设计共同目标只有一个：

**让结果更稳，而不是只让一次训练跑出好看数字。**

## 11. 当前最关键的源码位置

如果你要快速理解主链路，优先看：

- `src/prodata/mat_session_builder.py`
- `src/training/dataset.py`
- `src/models/cnn_encoder.py`
- `src/models/temporal_encoder.py`
- `src/models/fusion_classifier.py`
- `src/training/trainer.py`
- `src/training/evaluator.py`
- `manage.py`

如果你是初学者，建议配合阅读：

- `docs/algorithms/README.md`

## 12. 一句话总结整个仓库

如果只记一句话，请记住：

**这个仓库现在的核心，是把真实雷达 `.mat` 数据处理成标准训练数据集，再用“轨迹时序 + RD 微动”的高精度融合模型完成识别。**
