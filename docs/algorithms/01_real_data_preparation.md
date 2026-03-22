# 01. 真实数据准备与会话构建

## 这一步在做什么

这一步负责把 `pro/raw/TR/*.mat` 和 `pro/raw/RD/*.mat` 变成标准化训练会话目录。

核心目标是把“原始真实 `.mat` 文件”整理成训练代码能直接读取的结构：

```text
data/sessions/<session_id>/
├─ meta_session.json
├─ radar/track/track_data.csv
├─ radar/rd/rd_data.h5
├─ radar/rd/rd_clips.h5
├─ labels/labels_trackclip.jsonl
├─ labels/labels_matched.jsonl
└─ qc/qc_report.json
```

## 为什么不能直接拿 `.mat` 训练

因为训练代码需要的是统一格式的数据集，而不是各种来源、各种字段名、各种长度都不同的原始文件。

如果不先做标准化，会出现几个问题：

- 不同文件字段不一致
- 长度不一致
- 标签提取规则不一致
- 后续训练代码会越来越混乱

所以项目把“原始数据处理”和“模型训练”分成两层：

- `pro/`：处理真实数据
- `data/`：提供标准训练数据集

## 当前是怎么配对 TR 和 RD 的

配对逻辑在 `src/prodata/mat_session_builder.py` 中完成。

它会：

1. 扫描 `pro/raw/TR/*.mat`
2. 扫描 `pro/raw/RD/*.mat`
3. 从文件名中提取时间戳、标签、片段编号
4. 把同一个 `session_id` 的 `TR/RD` 配成一对

当前标签从文件名规范化为：

- `Bird`
- `RotaryDrone`
- `FixedWing`

## 会话构建后的意义

一旦会话构建完成，后面的训练代码就不需要再知道 `.mat` 里原始字段怎么命名了。

训练只关心：

- `track_data.csv` 有没有
- `rd_clips.h5` 有没有
- `labels_matched.jsonl` 有没有

这就是数据工程的意义：

**把复杂原始输入变成统一的训练接口。**

## 对应源码

- `src/prodata/mat_session_builder.py`
- `manage.py` 中的 `prepare-pro`
