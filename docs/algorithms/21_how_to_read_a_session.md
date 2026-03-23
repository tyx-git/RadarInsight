# 21. 如何完整读懂一个 session：从文件夹到模型输入

## 为什么要先学会“看一个 session”

很多初学者一上来就想看模型代码，但在真实项目里，更常见的问题其实是：

- 这个 session 里到底有什么数据
- 哪些文件是模型真正会读取的
- 这个 session 是否真的可以训练

所以学会读一个标准 session，是理解整个项目最实用的一步。

## 一个标准 session 长什么样

当前训练读取的是：

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

你可以把这个目录理解成一个“小型训练单元”。

## 推荐阅读顺序

如果你第一次检查某个 session，建议按这个顺序读：

```text
meta_session.json
-> qc_report.json
-> track_data.csv
-> labels_trackclip.jsonl
-> labels_matched.jsonl
-> rd_clips.h5
```

## `meta_session.json` 是什么

这是最适合当起点的文件。它更像一份摘要信息，帮助你快速知道：

- 当前 session 的标签
- 轨迹大致规模
- RD 数据大致形状
- 当前处理链路的一些基础元信息

## `track_data.csv` 看什么

这是轨迹分支最核心的来源。最应该检查：

- 行数是否足够
- `track_id` 是否正常
- 时间列是否递增
- `range_m / azimuth_deg / radial_speed_mps / snr_db` 是否像真实值

如果这里明显异常，后面的轨迹分支再强也学不好。

## `labels_trackclip.jsonl` 看什么

这个文件更像“样本切片说明书”，它描述：

- 一条轨迹被切成了哪些 clip
- 每个 clip 的时间窗是什么
- sample_id 如何命名

如果你想知道“为什么一个 session 会有这么多训练样本”，这里最值得看。

## `labels_matched.jsonl` 看什么

这个文件更接近训练标签的落地结果。你可以把它理解成：

- 当前样本最终拿什么标签参与训练
- 当前标签证据来自哪里

在当前项目里，它主要反映 session 级弱监督标签是如何落地到样本上的。

## `rd_clips.h5` 看什么

这是 RD 分支训练时真正读取的核心文件。最应该关心：

- 文件是否存在
- 每个 clip 的 shape 是否合理
- dtype 是否合理
- clip 数量是否和标签记录对得上

如果你只想确认 RD 链路是不是通的，这个文件通常是第一检查点。

## `qc_report.json` 看什么

这个文件更像一张“入场检查表”。如果 QC 本身就显示问题很多，那后续训练结果通常也不可靠。

## 如何快速判断一个 session 是否可训练

可以按下面 5 条快速判断：

1. `qc_report.json` 正常
2. `track_data.csv` 有足够轨迹行
3. `labels_trackclip.jsonl` 有 clip
4. `labels_matched.jsonl` 有标签
5. `rd_clips.h5` 存在且 shape 正常

## 对应源码

- `src/prodata/mat_session_builder.py`
- `src/training/dataset.py`
- `src/training/session_manager.py`


调模型之前，先确认一个 session 里的数据和标签到底是否完整，这一步往往比盲改网络更重要。