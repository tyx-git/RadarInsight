# 实验报告：expert1_v2_tr_rd_full

## 总体指标
- accuracy: 0.8613
- macro_precision: 0.7484
- macro_recall: 0.8024
- macro_f1: 0.7706
- route_accuracy: 1.0000
- loss: 0.6895
- skipped_non_finite_batches: 0

## 设备信息
- requested_device: cuda
- resolved_device: cuda
- cuda_available: True
- cuda_device_name: NVIDIA GeForce RTX 4060 Laptop GPU

## 训练产物
- best_checkpoint: artifacts\20260512-124930\expert1_v2_tr_rd_full\best_model.pt
- history_file: artifacts\20260512-124930\expert1_v2_tr_rd_full\training_history.json
- manifest_file: artifacts\20260512-124930\expert1_v2_tr_rd_full\training_split_manifest.json
- evaluation_report: artifacts\20260512-124930\expert1_v2_tr_rd_full\evaluation_report.json

## 各类别指标
| 类别 | support | precision | recall | f1 |
|---|---:|---:|---:|---:|
| DroneTarget | 151 | 0.9493 | 0.8675 | 0.9066 |
| BirdTarget | 80 | 0.7614 | 0.8375 | 0.7976 |
| BalloonTarget | 54 | 0.9815 | 0.9815 | 0.9815 |
| ClutterTarget | 7 | 0.5000 | 0.7143 | 0.5882 |
| UnknownTarget | 18 | 0.5500 | 0.6111 | 0.5789 |

## 主要混淆对
| 真实类别 | 预测类别 | 次数 |
|---|---|---:|
| DroneTarget | BirdTarget | 17 |
| BirdTarget | DroneTarget | 5 |
| BirdTarget | UnknownTarget | 5 |
| BirdTarget | ClutterTarget | 3 |
| UnknownTarget | BirdTarget | 3 |
| DroneTarget | UnknownTarget | 2 |
| ClutterTarget | UnknownTarget | 2 |
| UnknownTarget | DroneTarget | 2 |
| UnknownTarget | ClutterTarget | 2 |
| DroneTarget | BalloonTarget | 1 |

## 路由表现
| 路由 | support | correct | recall |
|---|---:|---:|---:|
| low_altitude_suspect | 310 | 310 | 1.0000 |

## Expert1 V2 分支
| 分支 | accuracy | macro_f1 |
|---|---:|---:|
| final | 0.8613 | 0.7706 |
| trajectory | 0.7774 | 0.6993 |
| TR+RD joint | 0.8387 | 0.7513 |
| RD evidence | 0.6613 | 0.5779 |
- joint_on_rd: accuracy=0.8987, macro_f1=0.7964, minus_traj_macro_f1=0.0863
- rd_evidence_on_rd: accuracy=0.7046, macro_f1=0.6627
| RD evidence 贡献指标 | 值 |
|---|---:|
| rd_rescue_count | 21 |
| rd_harm_count | 41 |
| rd_rescue_rate_on_rd | 0.0886 |
| rd_harm_rate_on_rd | 0.1730 |
| rd_net_gain_count | -20 |
| rd_net_gain_rate_on_rd | -0.0844 |
| rd_rescue_unadopted_count | 6 |
| rd_harm_adopted_count | 3 |
| known_confidence_mean | 0.7867 |
| residual_score_mean | 0.3215 |
| residual_score_on_residual_mean | 0.8446 |
| traj_residual_accuracy | 0.9226 |
| rd_residual_accuracy_on_rd | 0.4895 |
| rd_unknown_clutter_detail_accuracy_on_residual_rd | 0.8636 |
| rd_clutter_morphology_accuracy_on_residual_rd | 0.6818 |
| rd_unknown_clutter_detail_clutter_recall | 0.8571 |
| rd_clutter_morphology_clutter_recall | 0.5714 |

| bounded RD residual 权重 | on-RD 平均权重 |
|---|---:|
| DroneBird | 0.2038 |
| UnknownClutter | 0.2038 |
| Balloon | 0.2038 |

| RD hard pair | support | traj acc | rd acc | final acc | rd rescue | rd harm | rd net | final net |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DroneTarget__vs__BirdTarget | 162 | 0.7963 | 0.7160 | 0.9012 | 13 | 26 | -13 | 17 |
| UnknownTarget__vs__ClutterTarget | 22 | 0.5909 | 0.5455 | 0.6818 | 4 | 5 | -1 | 2 |
| BirdTarget__vs__ClutterTarget | 63 | 0.7778 | 0.6825 | 0.7937 | 7 | 13 | -6 | 1 |

## 按来源统计
- source_balanced_accuracy: 0.7151
- min_source_accuracy: 0.5000
- source_balanced_macro_f1: 0.5228
- min_source_macro_f1: 0.2000
| 来源 | count | final acc | final F1 | traj F1 | joint F1 | joint-on-RD F1 | RD-evidence-on-RD F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| cq08_track | 233 | 0.9056 | 0.8007 | 0.7123 | 0.8007 | 0.8007 | 0.6641 |
| excel_track | 73 | 0.7397 | 0.5677 | 0.5677 | 0.3845 | 0.0000 | 0.0000 |
| raw_tr_rd | 4 | 0.5000 | 0.2000 | 0.1333 | 0.2000 | 0.2000 | 0.1333 |

## 来源 x 类别指标
| 来源 | 类别 | support | predicted | precision | recall | f1 |
|---|---|---:|---:|---:|---:|---:|

## 典型错分样本（前20条）
| sample_id | source | 真实类别 | 预测类别 | 真实路由 | 初始路由 | 最终路由 | 回退路径 | reject reason | low reject score |
|---|---|---|---|---|---|---|---|---|---:|
| 20230929161757|Bird|12|6527 | raw_tr_rd | BirdTarget | DroneTarget | low_altitude_suspect | low_altitude_suspect | low_altitude_suspect |  |  | 0.0000 |
| 20231114155403|Bird|01|8175 | raw_tr_rd | BirdTarget | DroneTarget | low_altitude_suspect | low_altitude_suspect | low_altitude_suspect |  |  | 0.0000 |
| 20241130_213057|batch508 | excel_track | DroneTarget | BirdTarget | low_altitude_suspect | low_altitude_suspect | low_altitude_suspect |  |  | 0.0000 |
| 20241130_213057|batch703 | excel_track | DroneTarget | UnknownTarget | low_altitude_suspect | low_altitude_suspect | low_altitude_suspect |  |  | 0.0000 |
| 20250327_191432|batch625 | excel_track | UnknownTarget | DroneTarget | low_altitude_suspect | low_altitude_suspect | low_altitude_suspect |  |  | 0.0000 |
| 20250328_210801|batch210 | excel_track | DroneTarget | UnknownTarget | low_altitude_suspect | low_altitude_suspect | low_altitude_suspect |  |  | 0.0000 |
| 20250331_113300|batch197 | excel_track | DroneTarget | BirdTarget | low_altitude_suspect | low_altitude_suspect | low_altitude_suspect |  |  | 0.0000 |
| 20250331_113300|batch428 | excel_track | DroneTarget | BirdTarget | low_altitude_suspect | low_altitude_suspect | low_altitude_suspect |  |  | 0.0000 |
| 20250331_113300|batch431 | excel_track | DroneTarget | BirdTarget | low_altitude_suspect | low_altitude_suspect | low_altitude_suspect |  |  | 0.0000 |
| 20250331_113300|batch826 | excel_track | DroneTarget | BirdTarget | low_altitude_suspect | low_altitude_suspect | low_altitude_suspect |  |  | 0.0000 |
| 20250401_091500|batch243 | excel_track | DroneTarget | BirdTarget | low_altitude_suspect | low_altitude_suspect | low_altitude_suspect |  |  | 0.0000 |
| 20250401_091500|batch338 | excel_track | DroneTarget | BirdTarget | low_altitude_suspect | low_altitude_suspect | low_altitude_suspect |  |  | 0.0000 |
| 20250401_091500|batch476 | excel_track | DroneTarget | BirdTarget | low_altitude_suspect | low_altitude_suspect | low_altitude_suspect |  |  | 0.0000 |
| 20250401_091500|batch615 | excel_track | DroneTarget | BirdTarget | low_altitude_suspect | low_altitude_suspect | low_altitude_suspect |  |  | 0.0000 |
| 20250401_091500|batch74 | excel_track | DroneTarget | BirdTarget | low_altitude_suspect | low_altitude_suspect | low_altitude_suspect |  |  | 0.0000 |
| 20250403_190653|batch691 | excel_track | UnknownTarget | DroneTarget | low_altitude_suspect | low_altitude_suspect | low_altitude_suspect |  |  | 0.0000 |
| 20250403_190653|batch718 | excel_track | BirdTarget | DroneTarget | low_altitude_suspect | low_altitude_suspect | low_altitude_suspect |  |  | 0.0000 |
| 20250403_190653|batch746 | excel_track | BirdTarget | DroneTarget | low_altitude_suspect | low_altitude_suspect | low_altitude_suspect |  |  | 0.0000 |
| 20250406_193437|batch698 | excel_track | DroneTarget | BirdTarget | low_altitude_suspect | low_altitude_suspect | low_altitude_suspect |  |  | 0.0000 |
| 20250406_193437|batch701 | excel_track | DroneTarget | BirdTarget | low_altitude_suspect | low_altitude_suspect | low_altitude_suspect |  |  | 0.0000 |
