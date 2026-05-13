# Expert1 V2 TR+RD Full 训练快照

训练目录：`artifacts/20260511-224549/expert1_v2_tr_rd_full`

报告：

- `reports/real/20260511-224549/expert1_v2_tr_rd_full.md`
- `reports/real/20260511-224549/expert1_v2_tr_rd_full.json`

归档内容：

- `training_summary.json`
- `training_history.json`
- `training_split_manifest.json`
- `evaluation_report.json`
- `real_report.json`
- `real_report.md`
- `expert1_v2_tr_rd_full_config.json`

## 关键配置

| 项 | 值 |
|---|---:|
| epochs | 12 |
| batch size | 8 |
| stage1 trajectory-only epochs | 1 |
| stage2 ramp epochs | 1 |
| joint_logit_delta_scale | 1.0 |
| rd_specialist_residual_scale | 0.5 |
| max_train_batches_per_epoch | 0 |
| quick_val_max_batches | 0 |
| full_val_interval_epochs | 1 |
| train / val / test | 1432 / 306 / 310 |
| test RD / no-RD | 237 / 73 |
| trainable parameters | 24,409,745 |

## 测试集指标

| 指标 | 值 |
|---|---:|
| final accuracy | 0.8710 |
| final macro F1 | 0.7971 |
| trajectory accuracy | 0.8161 |
| trajectory macro F1 | 0.7614 |
| trajectory macro F1 on RD | 0.7756 |
| joint accuracy on RD | 0.9030 |
| joint macro F1 on RD | 0.8216 |
| joint - trajectory macro F1 on RD | +0.0460 |
| RD-only accuracy on RD | 0.6920 |
| RD-only macro F1 on RD | 0.6431 |
| final rescue / harm / net on RD | 22 / 5 / +17 |
| RD-only rescue / harm / net on RD | 12 / 45 / -33 |
| with RD accuracy | 0.9030 |
| without RD accuracy | 0.7671 |
| source-balanced macro F1 | 0.4990 |

## 按来源

| 来源 | count | final acc | final macro F1 | trajectory macro F1 |
|---|---:|---:|---:|---:|
| cq08_track | 233 | 0.9142 | 0.8287 | 0.7791 |
| excel_track | 73 | 0.7671 | 0.5881 | 0.5881 |
| raw_tr_rd | 4 | 0.2500 | 0.0800 | 0.1333 |

## 结论

这次全量训练证明当前 Expert1 V2 的大方向成立：有 RD 样本走 TR+RD joint 后，`joint-on-RD macro F1` 高于 `trajectory-on-RD macro F1`，final 在 RD 样本上产生正净收益。

主要问题不是 joint 架构失效，而是旧选模门槛仍把 RD-only 的净收益当作硬约束。新架构中 RD-only 只是 auxiliary evidence，候选选择应以 `joint/final` 相对 trajectory 的收益为主。

当前不能直接接入 System 的风险是弱源仍不稳：`raw_tr_rd` 测试只有 4 条，final 低于 trajectory；`excel_track` 与 20260510 稳定基线基本持平但略低。
