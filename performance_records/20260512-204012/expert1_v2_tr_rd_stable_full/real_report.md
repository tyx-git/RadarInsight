# 实验报告：expert1_v2_tr_rd_stable_full

## 总体指标
- accuracy: 0.8645
- macro_precision: 0.7405
- macro_recall: 0.7917
- macro_f1: 0.7526
- route_accuracy: 1.0000
- loss: 0.8775
- skipped_non_finite_batches: 0

## 设备信息
- requested_device: cuda
- resolved_device: cuda
- cuda_available: True
- cuda_device_name: NVIDIA GeForce RTX 4060 Laptop GPU

## 训练产物
- best_checkpoint: artifacts\20260512-204012\expert1_v2_tr_rd_stable_full\best_model.pt
- history_file: artifacts\20260512-204012\expert1_v2_tr_rd_stable_full\training_history.json
- manifest_file: artifacts\20260512-204012\expert1_v2_tr_rd_stable_full\training_split_manifest.json
- evaluation_report: artifacts\20260512-204012\expert1_v2_tr_rd_stable_full\evaluation_report.json

## 各类别指标
| 类别 | support | precision | recall | f1 |
|---|---:|---:|---:|---:|
| DroneTarget | 151 | 0.9156 | 0.9338 | 0.9246 |
| BirdTarget | 80 | 0.9048 | 0.7125 | 0.7972 |
| BalloonTarget | 54 | 1.0000 | 0.9630 | 0.9811 |
| ClutterTarget | 7 | 0.4444 | 0.5714 | 0.5000 |
| UnknownTarget | 18 | 0.4375 | 0.7778 | 0.5600 |

## 主要混淆对
| 真实类别 | 预测类别 | 次数 |
|---|---|---:|
| BirdTarget | UnknownTarget | 11 |
| BirdTarget | DroneTarget | 10 |
| DroneTarget | BirdTarget | 5 |
| DroneTarget | UnknownTarget | 4 |
| BirdTarget | ClutterTarget | 2 |
| ClutterTarget | UnknownTarget | 2 |
| UnknownTarget | DroneTarget | 2 |
| UnknownTarget | ClutterTarget | 2 |
| DroneTarget | ClutterTarget | 1 |
| BalloonTarget | BirdTarget | 1 |

## 路由表现
| 路由 | support | correct | recall |
|---|---:|---:|---:|
| low_altitude_suspect | 310 | 310 | 1.0000 |

## Expert1 V2 分支
| 分支 | accuracy | macro_f1 |
|---|---:|---:|
| final | 0.8645 | 0.7526 |
| trajectory | 0.8581 | 0.7358 |
| TR+RD joint | 0.8645 | 0.7500 |
| RD evidence | 0.6419 | 0.5032 |
- joint_on_rd: accuracy=0.8734, macro_f1=0.7618, minus_traj_macro_f1=0.0158
- rd_evidence_on_rd: accuracy=0.6835, macro_f1=0.5677
| RD evidence 贡献指标 | 值 |
|---|---:|
| rd_rescue_count | 9 |
| rd_harm_count | 52 |
| rd_rescue_rate_on_rd | 0.0380 |
| rd_harm_rate_on_rd | 0.2194 |
| rd_net_gain_count | -43 |
| rd_net_gain_rate_on_rd | -0.1814 |
| rd_rescue_unadopted_count | 6 |
| rd_harm_adopted_count | 4 |
| known_confidence_mean | 0.7994 |
| residual_score_mean | 0.0925 |
| residual_score_on_residual_mean | 0.4465 |
| traj_residual_accuracy | 0.9129 |
| rd_residual_accuracy_on_rd | 0.9072 |
| rd_unknown_clutter_detail_accuracy_on_residual_rd | 0.6818 |
| rd_clutter_morphology_accuracy_on_residual_rd | 0.6818 |
| rd_unknown_clutter_detail_clutter_recall | 0.1429 |
| rd_clutter_morphology_clutter_recall | 0.4286 |

| bounded RD residual 权重 | on-RD 平均权重 |
|---|---:|
| DroneBird | 0.1717 |
| UnknownClutter | 0.1717 |
| Balloon | 0.1717 |

| RD hard pair | support | traj acc | rd acc | final acc | rd rescue | rd harm | rd net | final net |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DroneTarget__vs__BirdTarget | 162 | 0.8580 | 0.6975 | 0.8580 | 6 | 32 | -26 | 0 |
| UnknownTarget__vs__ClutterTarget | 22 | 0.7273 | 0.6818 | 0.7727 | 2 | 3 | -1 | 1 |
| BirdTarget__vs__ClutterTarget | 63 | 0.6984 | 0.5714 | 0.6508 | 4 | 12 | -8 | -3 |

## 按来源统计
- source_balanced_accuracy: 0.7385
- min_source_accuracy: 0.5000
- source_balanced_macro_f1: 0.5217
- min_source_macro_f1: 0.2000
| 来源 | count | final acc | final F1 | traj F1 | joint F1 | joint-on-RD F1 | RD-evidence-on-RD F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| cq08_track | 233 | 0.8798 | 0.7659 | 0.7501 | 0.7659 | 0.7659 | 0.5663 |
| excel_track | 73 | 0.8356 | 0.5991 | 0.5991 | 0.5919 | 0.0000 | 0.0000 |
| raw_tr_rd | 4 | 0.5000 | 0.2000 | 0.2000 | 0.2000 | 0.2000 | 0.2933 |

## 来源 x 类别指标
| 来源 | 类别 | support | predicted | precision | recall | f1 |
|---|---|---:|---:|---:|---:|---:|

## 典型错分样本（前20条）
| sample_id | source | 真实类别 | 预测类别 | 真实路由 | 初始路由 | 最终路由 | 回退路径 | reject reason | low reject score |
|---|---|---|---|---|---|---|---|---|---:|
| 20230929161757|Bird|12|6527 | raw_tr_rd | BirdTarget | DroneTarget | low_altitude_suspect | low_altitude_suspect | low_altitude_suspect |  |  | 0.0000 |
| 20231114155403|Bird|01|8175 | raw_tr_rd | BirdTarget | DroneTarget | low_altitude_suspect | low_altitude_suspect | low_altitude_suspect |  |  | 0.0000 |
| 20241128_191432|batch694 | excel_track | DroneTarget | UnknownTarget | low_altitude_suspect | low_altitude_suspect | low_altitude_suspect |  |  | 0.0000 |
| 20241130_213057|batch803 | excel_track | BirdTarget | UnknownTarget | low_altitude_suspect | low_altitude_suspect | low_altitude_suspect |  |  | 0.0000 |
| 20250327_191432|batch625 | excel_track | UnknownTarget | DroneTarget | low_altitude_suspect | low_altitude_suspect | low_altitude_suspect |  |  | 0.0000 |
| 20250328_210801|batch210 | excel_track | DroneTarget | UnknownTarget | low_altitude_suspect | low_altitude_suspect | low_altitude_suspect |  |  | 0.0000 |
| 20250331_113300|batch197 | excel_track | DroneTarget | BirdTarget | low_altitude_suspect | low_altitude_suspect | low_altitude_suspect |  |  | 0.0000 |
| 20250331_113300|batch431 | excel_track | DroneTarget | BirdTarget | low_altitude_suspect | low_altitude_suspect | low_altitude_suspect |  |  | 0.0000 |
| 20250331_113300|batch514 | excel_track | BirdTarget | DroneTarget | low_altitude_suspect | low_altitude_suspect | low_altitude_suspect |  |  | 0.0000 |
| 20250401_091500|batch58 | excel_track | BirdTarget | DroneTarget | low_altitude_suspect | low_altitude_suspect | low_altitude_suspect |  |  | 0.0000 |
| 20250403_190653|batch691 | excel_track | UnknownTarget | DroneTarget | low_altitude_suspect | low_altitude_suspect | low_altitude_suspect |  |  | 0.0000 |
| 20250403_190653|batch746 | excel_track | BirdTarget | DroneTarget | low_altitude_suspect | low_altitude_suspect | low_altitude_suspect |  |  | 0.0000 |
| 20250406_193437|batch698 | excel_track | DroneTarget | BirdTarget | low_altitude_suspect | low_altitude_suspect | low_altitude_suspect |  |  | 0.0000 |
| 20250406_193437|batch770 | excel_track | DroneTarget | BirdTarget | low_altitude_suspect | low_altitude_suspect | low_altitude_suspect |  |  | 0.0000 |
| cq08|track|276 | cq08_track | DroneTarget | ClutterTarget | low_altitude_suspect | low_altitude_suspect | low_altitude_suspect |  |  | 0.0000 |
| cq08|track|335 | cq08_track | BirdTarget | DroneTarget | low_altitude_suspect | low_altitude_suspect | low_altitude_suspect |  |  | 0.0000 |
| cq08|track|343 | cq08_track | BirdTarget | UnknownTarget | low_altitude_suspect | low_altitude_suspect | low_altitude_suspect |  |  | 0.0000 |
| cq08|track|640 | cq08_track | BirdTarget | DroneTarget | low_altitude_suspect | low_altitude_suspect | low_altitude_suspect |  |  | 0.0000 |
| cq08|track|790 | cq08_track | BirdTarget | UnknownTarget | low_altitude_suspect | low_altitude_suspect | low_altitude_suspect |  |  | 0.0000 |
| cq08|track|827 | cq08_track | BirdTarget | DroneTarget | low_altitude_suspect | low_altitude_suspect | low_altitude_suspect |  |  | 0.0000 |
