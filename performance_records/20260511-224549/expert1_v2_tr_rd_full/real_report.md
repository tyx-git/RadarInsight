# 实验报告：expert1_v2_tr_rd_full

## 总体指标
- accuracy: 0.8710
- macro_precision: 0.7750
- macro_recall: 0.8387
- macro_f1: 0.7971
- route_accuracy: 1.0000
- loss: 0.6469
- skipped_non_finite_batches: 0

## 设备信息
- requested_device: cuda
- resolved_device: cuda
- cuda_available: True
- cuda_device_name: NVIDIA GeForce RTX 4060 Laptop GPU

## 训练产物
- best_checkpoint: artifacts\20260511-224549\expert1_v2_tr_rd_full\fallback_candidate_model.pt
- history_file: artifacts\20260511-224549\expert1_v2_tr_rd_full\training_history.json
- manifest_file: artifacts\20260511-224549\expert1_v2_tr_rd_full\training_split_manifest.json
- evaluation_report: artifacts\20260511-224549\expert1_v2_tr_rd_full\evaluation_report.json

## 各类别指标
| 类别 | support | precision | recall | f1 |
|---|---:|---:|---:|---:|
| DroneTarget | 151 | 0.9315 | 0.9007 | 0.9158 |
| BirdTarget | 80 | 0.8077 | 0.7875 | 0.7975 |
| BalloonTarget | 54 | 0.9298 | 0.9815 | 0.9550 |
| ClutterTarget | 7 | 0.5000 | 0.8571 | 0.6316 |
| UnknownTarget | 18 | 0.7059 | 0.6667 | 0.6857 |

## 主要混淆对
| 真实类别 | 预测类别 | 次数 |
|---|---|---:|
| DroneTarget | BirdTarget | 12 |
| BirdTarget | DroneTarget | 8 |
| BirdTarget | ClutterTarget | 4 |
| BirdTarget | UnknownTarget | 4 |
| DroneTarget | BalloonTarget | 2 |
| UnknownTarget | DroneTarget | 2 |
| UnknownTarget | BirdTarget | 2 |
| UnknownTarget | ClutterTarget | 2 |
| DroneTarget | UnknownTarget | 1 |
| BirdTarget | BalloonTarget | 1 |

## 路由表现
| 路由 | support | correct | recall |
|---|---:|---:|---:|
| low_altitude_suspect | 310 | 310 | 1.0000 |

## Expert1 RD 分支
| 分支 | accuracy | macro_f1 |
|---|---:|---:|
| final | 0.8710 | 0.7971 |
| trajectory | 0.8161 | 0.7614 |
| rd | 0.6613 | 0.5772 |
- rd_on_rd: accuracy=0.6920, macro_f1=0.6431
| RD 贡献指标 | 值 |
|---|---:|
| rd_rescue_count | 12 |
| rd_harm_count | 45 |
| rd_rescue_rate_on_rd | 0.0506 |
| rd_harm_rate_on_rd | 0.1899 |
| rd_net_gain_count | -33 |
| rd_net_gain_rate_on_rd | -0.1392 |
| rd_rescue_unadopted_count | 3 |
| rd_harm_adopted_count | 2 |
| known_confidence_mean | 0.7939 |
| residual_score_mean | 0.0829 |
| residual_score_on_residual_mean | 0.4304 |
| traj_residual_accuracy | 0.9258 |
| rd_residual_accuracy_on_rd | 0.9072 |
| rd_unknown_clutter_detail_accuracy_on_residual_rd | 0.8182 |
| rd_clutter_morphology_accuracy_on_residual_rd | 0.7727 |
| rd_unknown_clutter_detail_clutter_recall | 0.5714 |
| rd_clutter_morphology_clutter_recall | 0.7143 |

| RD 局部接管 gate | on-RD 平均权重 |
|---|---:|
| DroneBird | 0.0004 |
| UnknownClutter | 0.0004 |
| Balloon | 0.0004 |

| RD hard pair | support | traj acc | rd acc | final acc | rd rescue | rd harm | rd net | final net |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DroneTarget__vs__BirdTarget | 162 | 0.7901 | 0.6728 | 0.8951 | 10 | 29 | -19 | 17 |
| UnknownTarget__vs__ClutterTarget | 22 | 0.7727 | 0.7727 | 0.7727 | 2 | 2 | 0 | 0 |
| BirdTarget__vs__ClutterTarget | 63 | 0.7460 | 0.6508 | 0.7778 | 5 | 11 | -6 | 2 |

## 按来源统计
- source_balanced_accuracy: 0.6438
- min_source_accuracy: 0.2500
- source_balanced_macro_f1: 0.4990
- min_source_macro_f1: 0.0800
| 来源 | count | accuracy | macro_f1 | macro_recall |
|---|---:|---:|---:|---:|
| cq08_track | 233 | 0.9142 | 0.8287 | 0.8671 |
| excel_track | 73 | 0.7671 | 0.5881 | 0.5844 |
| raw_tr_rd | 4 | 0.2500 | 0.0800 | 0.0667 |

## 来源 x 类别指标
| 来源 | 类别 | support | predicted | precision | recall | f1 |
|---|---|---:|---:|---:|---:|---:|

## 典型错分样本（前20条）
| sample_id | source | 真实类别 | 预测类别 | 真实路由 | 初始路由 | 最终路由 | 回退路径 | reject reason | low reject score |
|---|---|---|---|---|---|---|---|---|---:|
| 20230824104942|Rotary drone|04|1971 | raw_tr_rd | DroneTarget | BirdTarget | low_altitude_suspect | low_altitude_suspect | low_altitude_suspect |  |  | 0.0000 |
| 20230929161757|Bird|12|6527 | raw_tr_rd | BirdTarget | DroneTarget | low_altitude_suspect | low_altitude_suspect | low_altitude_suspect |  |  | 0.0000 |
| 20231014161536|Bird|13|15 | raw_tr_rd | BirdTarget | DroneTarget | low_altitude_suspect | low_altitude_suspect | low_altitude_suspect |  |  | 0.0000 |
| 20250327_191432|batch625 | excel_track | UnknownTarget | DroneTarget | low_altitude_suspect | low_altitude_suspect | low_altitude_suspect |  |  | 0.0000 |
| 20250328_210801|batch210 | excel_track | DroneTarget | UnknownTarget | low_altitude_suspect | low_altitude_suspect | low_altitude_suspect |  |  | 0.0000 |
| 20250331_113300|batch197 | excel_track | DroneTarget | BirdTarget | low_altitude_suspect | low_altitude_suspect | low_altitude_suspect |  |  | 0.0000 |
| 20250331_113300|batch431 | excel_track | DroneTarget | BirdTarget | low_altitude_suspect | low_altitude_suspect | low_altitude_suspect |  |  | 0.0000 |
| 20250331_113300|batch514 | excel_track | BirdTarget | DroneTarget | low_altitude_suspect | low_altitude_suspect | low_altitude_suspect |  |  | 0.0000 |
| 20250401_091500|batch243 | excel_track | DroneTarget | BirdTarget | low_altitude_suspect | low_altitude_suspect | low_altitude_suspect |  |  | 0.0000 |
| 20250401_091500|batch338 | excel_track | DroneTarget | BirdTarget | low_altitude_suspect | low_altitude_suspect | low_altitude_suspect |  |  | 0.0000 |
| 20250401_091500|batch476 | excel_track | DroneTarget | BirdTarget | low_altitude_suspect | low_altitude_suspect | low_altitude_suspect |  |  | 0.0000 |
| 20250401_091500|batch58 | excel_track | BirdTarget | DroneTarget | low_altitude_suspect | low_altitude_suspect | low_altitude_suspect |  |  | 0.0000 |
| 20250401_091500|batch615 | excel_track | DroneTarget | BirdTarget | low_altitude_suspect | low_altitude_suspect | low_altitude_suspect |  |  | 0.0000 |
| 20250401_091500|batch74 | excel_track | DroneTarget | BirdTarget | low_altitude_suspect | low_altitude_suspect | low_altitude_suspect |  |  | 0.0000 |
| 20250403_190653|batch691 | excel_track | UnknownTarget | DroneTarget | low_altitude_suspect | low_altitude_suspect | low_altitude_suspect |  |  | 0.0000 |
| 20250403_190653|batch718 | excel_track | BirdTarget | DroneTarget | low_altitude_suspect | low_altitude_suspect | low_altitude_suspect |  |  | 0.0000 |
| 20250403_190653|batch746 | excel_track | BirdTarget | DroneTarget | low_altitude_suspect | low_altitude_suspect | low_altitude_suspect |  |  | 0.0000 |
| 20250406_193437|batch481 | excel_track | DroneTarget | BirdTarget | low_altitude_suspect | low_altitude_suspect | low_altitude_suspect |  |  | 0.0000 |
| 20250406_193437|batch698 | excel_track | DroneTarget | BirdTarget | low_altitude_suspect | low_altitude_suspect | low_altitude_suspect |  |  | 0.0000 |
| 20250406_193437|batch770 | excel_track | DroneTarget | BirdTarget | low_altitude_suspect | low_altitude_suspect | low_altitude_suspect |  |  | 0.0000 |
