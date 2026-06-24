# RadarInsight 最新复审结论

更新时间：2026-05-06

当前仍在训练期，目标优先级仍是最高精度、强鲁棒性、强泛化；部署吞吐暂不作为主约束。

基于最新系统产物：

- `artifacts/20260506-032514/system_config_smoke/evaluation_report.json`
- `tests/ground_system_alignment_report.json`
- `tests/system_failure_root_cause_report.json`

## 已确认修复

1. Expert2 不再全判 `NonAircraft`。
2. Aircraft 在 System 端不再被完全压制。
3. reroute 分支仍在代码里，但当前实际选中的融合模式是 `ungated`，`reroute_rate = 0`，所以它不是当前主因。

## 仍然卡精度的主因

### P0 Ground/Expert4 在 System 最终仲裁中被压制

最新结果：

- System final `ground_group_f1 = 0.4138`
- standalone Expert4 on System test split `ground_group_f1 = 0.5641`
- System representation + Expert4 head `ground_group_f1 = 0.5405`
- 诊断结论 `primary = global_arbitration_suppression`

这说明问题不在地面头本体，而是在 System 级最终组装里被吃掉了。

关键代码点：

- `src/core/system_trainer.py` 的训练损失里没有 groundness supervision，ground 分支只吃 3 类分类损失。
- `config/config.json` 里 `groundness_protection_enabled = false`，等于把保护链路关掉了。
- `src/core/system.py` 和 `src/core/system_trainer.py` 仍然让 ground 分支和低空/Aircraft 直接竞争全局 logits。

结论：

ground 头在独立或对照条件下是能工作的，但 System 联合训练后被漂移，最终仲裁又没有保护它。

### P1 弱源仍然拖 macro

最新源域表现：

- `raw_tr_rd` accuracy `0.5455`
- `excel_track` accuracy `0.7273`
- `cq08_track` accuracy `0.9270`
- `source_balanced_accuracy = 0.7333`
- `min_source_accuracy = 0.5455`

这说明 source skew 仍然很重，`raw_tr_rd` 还是当前最弱源。

### P1 Expert1 还没到低空安全级别

最新 Expert1 仍有明显短板：

- `expert1_low_altitude_macro_f1 = 0.7970`
- `ClutterTarget recall = 0.5714`
- `UnknownTarget recall = 0.6667`

低空主类已经可用，但 RD 分支还没达到“显著强于轨迹”的目标。

## 当前系统结论

最新 System：

- accuracy `0.8614`
- macro F1 `0.6112`
- route accuracy `0.7319`
- Aircraft `precision = 0.5714`, `recall = 0.8`
- `non_aircraft_as_aircraft_rate = 0.0092`

所以现在的瓶颈不是 Aircraft 了，而是 ground 分支和弱源泛化。

## 优先修复顺序

1. 先修 System 里的 ground 链路
   - 要么冻结/保护导入的 Expert4 头
   - 要么把 groundness supervision 加回 System 训练
   - 在 groundness 仍不可分前，不要盲目打开 `groundness_protection_enabled`
2. 再收紧选模目标
   - 让 source 维度的弱源指标进入更硬的惩罚
   - 不要只看总体 macro F1
3. 最后继续补 Expert1 的 RD 分支和长尾类

---

# 2026-05-10 Expert1 V2 暂存实施计划

## 架构目标

```text
无 RD：trajectory-only branch -> final logits
有 RD：trajectory branch + RD evidence branch -> TR+RD joint branch -> final logits
```

## 本轮先落地的最小闭环

1. 保留当前 `traj_backbone`、`traj_head`、`traj_residual_head`。
2. 保留当前 RD encoder 和 RD auxiliary heads。
3. 新增 TR+RD joint head：
   - 输入 `traj_features`
   - 输入 `rd_features_vec`
   - 输入 `abs(traj_features - rd_features_vec)`
   - 输入 `traj_features * rd_features_vec`
   - 输出 `joint_logits`
4. 新增 reliability head：
   - 输出 `track_weight` / bounded `rd_weight`
   - 无 RD 时 `rd_weight = 0`
5. 修改 `dual_main` 语义：
   - 无 RD 样本 final 使用 `traj_logits`
   - 有 RD 样本 final 使用 `traj_logits + rd_weight * (joint_logits - traj_logits)`
   - RD specialist residual 只保留小权重且乘 `rd_weight`
6. 暂不删除旧 RD auxiliary 和 hard-pair 报告，避免破坏现有 trainer 指标。

## 后续训练目标

- `trajectory-only macro F1` 不低于当前稳定基线。
- `joint-on-RD macro F1 > trajectory-on-RD macro F1`。
- `final_net_gain_rate_on_rd >= 0`。
- 三组 hard pair final net gain 均不为负。

---

# 2026-05-11 Expert1 V2 TR+RD 全量训练暂存

## 最新结构

```text
无 RD：
    trajectory-only branch -> final logits

有 RD：
    trajectory sequence tokens + RD sequence tokens
    -> cross-modal fusion
    -> TR+RD joint logits
    -> final logits
```

关键约束：

- 不再用轻量快速训练作为候选结论。
- 不再把 joint 分支隔离成只报告不训练。
- 有 RD 样本的主分类 loss 直接监督 TR+RD 混合分支。
- 无 RD 样本 final 必须严格回退到 `traj_logits`。

## 全量训练配置

配置文件：

```text
tmp/expert1_v2_tr_rd_full_config.json
```

```text
output_dir = artifacts/expert1_v2_tr_rd_full
epochs = 12
max_train_batches_per_epoch = 0
quick_val_max_batches = 0
full_val_interval_epochs = 1
joint_logit_delta_scale = 1.0
low_altitude_deep_reroute_aux_weight = 0.35
low_altitude_rd_forced_fusion_weight = 0.9
low_altitude_rd_gate_aux_weight = 0.35
```

## 训练命令

```powershell
.\.venv\Scripts\python.exe -m src.core.train_expert1 --config tmp\expert1_v2_tr_rd_full_config.json
```

训练完成后必须把对应 `artifacts/<timestamp>/expert1_v2_tr_rd_full/` 下的完整性能文件复制到 `performance_records/<timestamp>/expert1_v2_tr_rd_full/`：

- `training_summary.json`
- `evaluation_report.json`
- `training_history.json`
- 本次配置文件副本

---

# 2026-05-12 Expert1 V2 stable full 暂存

## 目的

20260512-124930 证明 TR+RD joint 有效，但 stage2 打开过猛，trajectory anchor 和 without-RD/excel 表现被压低。下一轮不改变“有 RD 走 TR+RD、无 RD 走 trajectory”的架构，只修训练稳定性。

## 暂存配置

配置文件：

```text
tmp/expert1_v2_tr_rd_stable_full_config.json
```

关键参数：

```text
output_dir = artifacts/expert1_v2_tr_rd_stable_full
epochs = 18
stage1_traj_only_epochs = 3
stage2_ramp_epochs = 4
low_altitude_traj_supervision_weight = 0.55
low_altitude_traj_aux_ratio = 0.55
low_altitude_rd_aux_ratio = 0.45
low_altitude_rd_forced_fusion_weight = 0.45
low_altitude_deep_reroute_aux_weight = 0.20
low_altitude_traj_anchor_weight = 0.35
low_altitude_traj_anchor_start_epoch = 2
low_altitude_rd_teacher_warmup_epochs = 2
excel_source_boost = 2.6
raw_source_boost = 2.5
```

代码侧已同步：

- `low_altitude_traj_supervision_weight` 持续监督 `traj_logits`。
- strong fusion / joint / deep reroute direct loss 已乘 `stage2_scale`。
- `low_altitude_traj_anchor_start_epoch` 已实际参与 anchor KL 启动条件。

## 训练命令

```powershell
.\.venv\Scripts\python.exe -m src.core.train_expert1 --config tmp\expert1_v2_tr_rd_stable_full_config.json
```

训练完成后把完整性能文件复制到：

```text
performance_records/<timestamp>/expert1_v2_tr_rd_stable_full/
```

必须保存：

- `training_summary.json`
- `evaluation_report.json`
- `training_history.json`
- `training_split_manifest.json`
- `reports/real/<timestamp>/expert1_v2_tr_rd_stable_full.md`
- `reports/real/<timestamp>/expert1_v2_tr_rd_stable_full.json`
- `tmp/expert1_v2_tr_rd_stable_full_config.json`
# 2026-05-12 Expert1 latency benchmark

## 脚本

```powershell
.\.venv\Scripts\python.exe tmp\benchmark_expert1_latency.py --config tmp\expert1_v2_tr_rd_stable_full_config.json --checkpoint artifacts\20260512-124930\expert1_v2_tr_rd_full\best_model.pt --device cuda --batch-size 8 --batches 16 --warmup 3 --modes v2_joint traj_skip_rd --output tmp\expert1_latency_cuda.json
```

当前训练进程占用 GPU 时不要跑 CUDA 基准；否则吞吐和时延会被训练污染。

## CPU smoke 结果

命令：

```powershell
.\.venv\Scripts\python.exe tmp\benchmark_expert1_latency.py --config tmp\expert1_v2_tr_rd_stable_full_config.json --checkpoint artifacts\20260512-124930\expert1_v2_tr_rd_full\best_model.pt --device cpu --batch-size 2 --batches 2 --warmup 0 --modes v2_joint traj_only traj_skip_rd --output tmp\expert1_latency_cpu_smoke.json
```

结果只验证脚本和量级，不代表部署 CUDA 性能：

| mode | samples/s | avg latency / sample | avg batch latency |
|---|---:|---:|---:|
| v2_joint | 12.77 | 78.33 ms | 156.66 ms |
| traj_only but RD encoded | 14.40 | 69.46 ms | 138.93 ms |
| traj_skip_rd optimized | 314.03 | 3.18 ms | 6.37 ms |

说明：

- `traj_skip_rd` 是真正跳过 RD encoder 和 joint fusion 的无 RD 快路径。
- `v2_joint` 包含 trajectory、RD summary、RD Transformer、RD patch CNN、cross-modal joint head。
- 当前模型参数量：24,456,017。

## 外部对照口径

UAVDetect 论文按 batch=10 报告了 RAD-DAR / UAVDetect / VGG 在 edge/GPU 上的 batch latency 和 FPS。该类方法多为 spectrogram CNN，输入和任务比本项目的“轨迹 + RD + MoE/联合证据”更轻，因此只能作为量级参考，不是严格同任务同硬件对比。

参考：

- UAVDetect: A Channel Attention based CNN for Improving UAV Classification in Radar. `https://www.mdpi.com/1424-8220/24/17/5805`

---

# 2026-05-13 Expert1 部署式窗口切分暂存

## 生成窗口索引

```powershell
.\.venv\Scripts\python.exe -m src.process.build_expert1_window_index --window-seconds 10 --stride-seconds 2 --min-points 5
```

输出：

```text
data/processed/expert1_window_index.csv
data/manifests/expert1_window_split_manifest.json
data/processed/expert1_window_summary.json
```

当前统计：

```text
total_windows = 16271
trainable_windows = 15793
diagnostic_windows = 478
parent_tracks = 2048
parent_leakage_count = 0
trainable train/val/test = 11426 / 2300 / 2067
```

校验命令：

```powershell
.\.venv\Scripts\python.exe -c "import pandas as pd; df=pd.read_csv('data/processed/expert1_window_index.csv'); train=df[df.is_trainable.astype(str).str.lower().isin(['true','1','yes'])]; print(len(train[train.window_duration_seconds>10.000001])); print(len(train[train.window_point_count<5])); print(df.groupby('split_group_id').split_name.nunique().gt(1).sum())"
```

期望输出：

```text
0
0
0
```

## 窗口全量训练配置

配置文件：

```text
tmp/expert1_v2_window_stable_full_config.json
```

关键参数：

```text
index_name = expert1_window_index.csv
output_dir = artifacts/expert1_v2_window_stable_full
epochs = 18
batch_size = 8
stage1_traj_only_epochs = 3
stage2_ramp_epochs = 4
parent_window_balance_power = 0.5
```

训练命令：

```powershell
.\.venv\Scripts\python.exe -m src.core.train_expert1 --config tmp\expert1_v2_window_stable_full_config.json
```

本轮训练结束后保存性能快照到：

```text
performance_records/<timestamp>/expert1_v2_window_stable_full/
```

必须保存：

- `training_summary.json`
- `evaluation_report.json`
- `training_history.json`
- `training_split_manifest.json`
- `reports/real/<timestamp>/expert1_v2_window_stable_full.md`
- `reports/real/<timestamp>/expert1_v2_window_stable_full.json`
- `tmp/expert1_v2_window_stable_full_config.json`

重点看：

- 窗口级 macro F1 / macro recall。
- Bird / Unknown / Clutter recall。
- without-RD window accuracy。
- source-balanced window macro F1。
- parent mean-logit / majority-vote / first-window。
- stable decision latency。

## Expert1 RD pipeline 首轮优化暂存

配置文件：

```text
tmp/expert1_v2_window_stable_full_config.json
```

新增关键参数：

```text
rd_encoded_cache_size = 256
enable_window_rd_time_crop = true
rd_window_crop_min_frames = 2
rd_joint_segment_count = 5
enable_rd_segment_joint_tokens = true
```

本轮已实现：

- 同一父 RD tensor / summary 只加载和编码一次，窗口样本复用编码结果。
- 窗口化索引下按窗口时间范围裁切 RD，避免每个 10 秒窗口继承完整父 RD。
- TR+RD joint cross-attention 前将 RD token 聚合成最多 5 个 segment token。
- Expert1 报告写入 `rd_pipeline` 配置，System 接回链路同步接入 segment token 参数。

已做验证：

```text
.\.venv\Scripts\python.exe -m compileall -q src\core\data\torch_dataset.py src\core\expert1_data.py src\core\expert1_config.py src\core\expert1_trainer.py src\core\experts.py src\core\system.py src\core\system_config.py src\core\system_trainer.py
```

结果：

```text
通过
```

小 batch 前向结果：

```text
logits (2, 5)
joint_logits (2, 5)
tr_rd_active [True, True]
rd_lengths 示例 [44, 47]
rd_patch_lengths 示例 [44, 47]
rd_deep_patch_lengths 示例 [54, 58]
```

注意：

- 不要把这版理解成最终 RD segment embedding cache；当前是训练可用的首轮内存缓存和 token 聚合。
- 下一轮若继续优化，应优先做持久化 segment embedding、显式 RD 时间轴对齐、deep patch 降采样消融、异步预取。

---

# 2026-05-15 Expert1 RD Segment Store Smoke 暂存

新增测试文件：

```text
tests/test_expert1_rd_segment_store_smoke.py
```

覆盖链路：

```text
build_expert1_rd_segment_store
  -> build_window_segment_map
  -> WindowAssembler
  -> TorchTrackDataset(use_rd_segment_store=True)
  -> collate_track_batch
```

已修复问题：

- CSV manifest 中的 `"False"` 字符串不能通过 `bool("False")` 判断，否则 summary RD 会被错误标记为多模态 RD。
- RD segment store 使用 `overwrite=True` 重建时，需要清理当前 store 目录内旧的 `*_segments.pt`，避免残留 shard 干扰部署排查。

验证结果：

```text
.\.venv\Scripts\python.exe -m compileall -q src\process\build_expert1_rd_segment_store.py src\core\data\rd_segment_store.py src\core\data\torch_dataset.py tests\test_expert1_rd_segment_store_smoke.py
通过

.\.venv\Scripts\python.exe -m unittest tests.test_expert1_rd_segment_store_smoke -v
Ran 4 tests in 1.215s
OK
```

---

# 2026-05-15 Expert1 窗口训练加速暂存

已落地：

- 训练前统计不再扫真实 DataLoader：
  - class weights -> `index_df`
  - RD class weights -> `index_df`
  - RD coverage -> `index_df`
- source normalization stats 改为只读轨迹 CSV，并缓存到：

```text
data/processed/expert1_window_index_source_feature_stats.json
```

- DataLoader 配置新增：

```text
pin_memory
persistent_workers
prefetch_factor
```

- 训练器 CUDA 搬运新增：

```text
non_blocking_h2d
```

- 验证节奏新增：

```text
full_val_warmup_epochs
full_val_interval_epochs = 3
quick_val_max_batches = 80
```

- 专家一新增整 batch 无 RD 快路径：

```text
batch 内全部 rd_is_multimodal=false -> 直接 trajectory-only，不运行 RD encoder / patch CNN / joint fusion
```

验证：

```text
compileall 通过
tests.test_expert1_rd_segment_store_smoke: Ran 6 tests, OK
tmp/expert1_v2_window_speed_smoke_config.json: 1-batch 训练 smoke 完整写出 summary/report
```

注意：

- 正式 RD segment store 已构建完成：1228 个 parent shard、9994 个 2s segment，store 总大小约 18.68GB，`failed_parent_count=0`。
- 正式 `expert1_window_segment_map.csv` 已生成完成：5798 个有 RD 可训练窗口全部映射成功，`missing_window_count=0`。
- `tmp/expert1_v2_window_stable_full_config.json` 已切到 `use_rd_segment_store=true`。
- 前向 smoke 通过：同一 batch 中 RD 样本 `tr_rd_active=True`、纯轨迹样本 `tr_rd_active=False`，`logits_shape=(2,5)`。

---

# 2026-05-16 Expert1 stage2 降速暂存

结论：

- epoch 1-3 快，是因为 `stage2=off`。
- epoch 4 之后慢，是因为 `stage2=on` 后每个含 RD batch 额外跑 `deep_reroute` 辅助前向。
- 短 profiling 显示主瓶颈不是 DataLoader：`get_batch` 约 0.07s 到 0.20s。
- 修改前 deep auxiliary batch 常见 5s 到 8s，极端 20s+。

已改：

```text
low_altitude_deep_reroute_aux_interval = 4
low_altitude_deep_reroute_max_frames = 32
```

- `deep_reroute` auxiliary 每 4 个 batch 运行一次。
- deep auxiliary 输入最多 32 帧。
- RD / patch / deep patch 各自独立下采样，避免时间轴长度不同导致 CUDA index 越界。
- stage1 `disable_tr_rd_fusion=True` 时直接 trajectory-only，不再跑 RD 分支。

验证：

```text
compileall 通过
tests.test_expert1_rd_segment_store_smoke: Ran 6 tests, OK
stage2 profiling：batch 约 0.2s 到 0.35s
```
