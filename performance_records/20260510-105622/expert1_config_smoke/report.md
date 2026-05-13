# 实验报告：expert1_config_smoke

## 总体指标
- accuracy: 0.8548
- macro_precision: 0.7074
- macro_recall: 0.7889
- macro_f1: 0.7327
- route_accuracy: 1.0000
- loss: 4.0780
- skipped_non_finite_batches: 0

## 设备信息
- requested_device: cuda
- resolved_device: cuda
- cuda_available: True
- cuda_device_name: NVIDIA GeForce RTX 4060 Laptop GPU

## 训练产物
- best_checkpoint: artifacts\20260510-105622\expert1_config_smoke\fallback_candidate_model.pt
- history_file: artifacts\20260510-105622\expert1_config_smoke\training_history.json
- manifest_file: artifacts\20260510-105622\expert1_config_smoke\training_split_manifest.json
- evaluation_report: artifacts\20260510-105622\expert1_config_smoke\evaluation_report.json

## 各类别指标
| 类别 | support | precision | recall | f1 |
|---|---:|---:|---:|---:|
| DroneTarget | 151 | 0.9510 | 0.9007 | 0.9252 |
| BirdTarget | 80 | 0.8333 | 0.7500 | 0.7895 |
| BalloonTarget | 54 | 0.9623 | 0.9444 | 0.9533 |
| ClutterTarget | 7 | 0.3077 | 0.5714 | 0.4000 |
| UnknownTarget | 18 | 0.4828 | 0.7778 | 0.5957 |

## 主要混淆对
| 真实类别 | 预测类别 | 次数 |
|---|---|---:|
| DroneTarget | BirdTarget | 10 |
| BirdTarget | UnknownTarget | 8 |
| BirdTarget | DroneTarget | 6 |
| BirdTarget | ClutterTarget | 5 |
| DroneTarget | UnknownTarget | 4 |
| ClutterTarget | UnknownTarget | 3 |
| BalloonTarget | ClutterTarget | 2 |
| UnknownTarget | ClutterTarget | 2 |
| DroneTarget | BalloonTarget | 1 |
| BirdTarget | BalloonTarget | 1 |

## 路由表现
| 路由 | support | correct | recall |
|---|---:|---:|---:|
| low_altitude_suspect | 310 | 310 | 1.0000 |

## Expert1 RD 分支
| 分支 | accuracy | macro_f1 |
|---|---:|---:|
| final | 0.8548 | 0.7327 |
| trajectory | 0.8581 | 0.7486 |
| rd | 0.6645 | 0.5547 |
- rd_on_rd: accuracy=0.7173, macro_f1=0.6174
| RD 贡献指标 | 值 |
|---|---:|
| rd_rescue_count | 7 |
| rd_harm_count | 47 |
| rd_rescue_rate_on_rd | 0.0295 |
| rd_harm_rate_on_rd | 0.1983 |
| rd_net_gain_count | -40 |
| rd_net_gain_rate_on_rd | -0.1688 |
| rd_rescue_unadopted_count | 7 |
| rd_harm_adopted_count | 3 |
| known_confidence_mean | 0.7731 |
| residual_score_mean | 0.0975 |
| residual_score_on_residual_mean | 0.4644 |
| traj_residual_accuracy | 0.9129 |
| rd_residual_accuracy_on_rd | 0.9072 |
| rd_unknown_clutter_detail_accuracy_on_residual_rd | 0.7273 |
| rd_clutter_morphology_accuracy_on_residual_rd | 0.6818 |
| rd_unknown_clutter_detail_clutter_recall | 0.5714 |
| rd_clutter_morphology_clutter_recall | 0.4286 |

| RD 局部接管 gate | on-RD 平均权重 |
|---|---:|
| DroneBird | 0.1307 |
| UnknownClutter | 0.0785 |
| Balloon | 0.1069 |

| RD hard pair | support | traj acc | rd acc | final acc | rd rescue | rd harm | rd net | final net |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DroneTarget__vs__BirdTarget | 162 | 0.8827 | 0.7222 | 0.8827 | 5 | 31 | -26 | 0 |
| UnknownTarget__vs__ClutterTarget | 22 | 0.7727 | 0.6364 | 0.7273 | 1 | 4 | -3 | -1 |
| BirdTarget__vs__ClutterTarget | 63 | 0.7143 | 0.6984 | 0.6984 | 4 | 5 | -1 | -1 |

## 按来源统计
- source_balanced_accuracy: 0.6366
- min_source_accuracy: 0.2500
- source_balanced_macro_f1: 0.4788
- min_source_macro_f1: 0.0800
| 来源 | count | accuracy | macro_f1 | macro_recall |
|---|---:|---:|---:|---:|
| cq08_track | 233 | 0.8927 | 0.7615 | 0.8063 |
| excel_track | 73 | 0.7671 | 0.5948 | 0.6467 |
| raw_tr_rd | 4 | 0.2500 | 0.0800 | 0.0667 |

## 来源 x 类别指标
| 来源 | 类别 | support | predicted | precision | recall | f1 |
|---|---|---:|---:|---:|---:|---:|

## 典型错分样本（前20条）
| sample_id | source | 真实类别 | 预测类别 | 真实路由 | 初始路由 | 最终路由 | 回退路径 | reject reason | low reject score |
|---|---|---|---|---|---|---|---|---|---:|
| 20230824104942|Rotary drone|04|1971 | raw_tr_rd | DroneTarget | BirdTarget | low_altitude_suspect | low_altitude_suspect | low_altitude_suspect |  |  | 0.0000 |
| 20231014161536|Bird|13|15 | raw_tr_rd | BirdTarget | DroneTarget | low_altitude_suspect | low_altitude_suspect | low_altitude_suspect |  |  | 0.0000 |
| 20231114155403|Bird|01|8175 | raw_tr_rd | BirdTarget | DroneTarget | low_altitude_suspect | low_altitude_suspect | low_altitude_suspect |  |  | 0.0000 |
| 20241128_191432|batch694 | excel_track | DroneTarget | UnknownTarget | low_altitude_suspect | low_altitude_suspect | low_altitude_suspect |  |  | 0.0000 |
| 20241130_213057|batch508 | excel_track | DroneTarget | UnknownTarget | low_altitude_suspect | low_altitude_suspect | low_altitude_suspect |  |  | 0.0000 |
| 20241130_213057|batch703 | excel_track | DroneTarget | UnknownTarget | low_altitude_suspect | low_altitude_suspect | low_altitude_suspect |  |  | 0.0000 |
| 20250327_191432|batch625 | excel_track | UnknownTarget | DroneTarget | low_altitude_suspect | low_altitude_suspect | low_altitude_suspect |  |  | 0.0000 |
| 20250328_210801|batch210 | excel_track | DroneTarget | UnknownTarget | low_altitude_suspect | low_altitude_suspect | low_altitude_suspect |  |  | 0.0000 |
| 20250331_113300|batch197 | excel_track | DroneTarget | BirdTarget | low_altitude_suspect | low_altitude_suspect | low_altitude_suspect |  |  | 0.0000 |
| 20250331_113300|batch431 | excel_track | DroneTarget | BirdTarget | low_altitude_suspect | low_altitude_suspect | low_altitude_suspect |  |  | 0.0000 |
| 20250331_113300|batch514 | excel_track | BirdTarget | DroneTarget | low_altitude_suspect | low_altitude_suspect | low_altitude_suspect |  |  | 0.0000 |
| 20250401_091500|batch243 | excel_track | DroneTarget | BirdTarget | low_altitude_suspect | low_altitude_suspect | low_altitude_suspect |  |  | 0.0000 |
| 20250401_091500|batch476 | excel_track | DroneTarget | BirdTarget | low_altitude_suspect | low_altitude_suspect | low_altitude_suspect |  |  | 0.0000 |
| 20250401_091500|batch58 | excel_track | BirdTarget | DroneTarget | low_altitude_suspect | low_altitude_suspect | low_altitude_suspect |  |  | 0.0000 |
| 20250401_091500|batch615 | excel_track | DroneTarget | BirdTarget | low_altitude_suspect | low_altitude_suspect | low_altitude_suspect |  |  | 0.0000 |
| 20250403_190653|batch718 | excel_track | BirdTarget | UnknownTarget | low_altitude_suspect | low_altitude_suspect | low_altitude_suspect |  |  | 0.0000 |
| 20250403_190653|batch746 | excel_track | BirdTarget | DroneTarget | low_altitude_suspect | low_altitude_suspect | low_altitude_suspect |  |  | 0.0000 |
| 20250406_193437|batch481 | excel_track | DroneTarget | BirdTarget | low_altitude_suspect | low_altitude_suspect | low_altitude_suspect |  |  | 0.0000 |
| 20250406_193437|batch698 | excel_track | DroneTarget | BirdTarget | low_altitude_suspect | low_altitude_suspect | low_altitude_suspect |  |  | 0.0000 |
| 20250406_193437|batch770 | excel_track | DroneTarget | BirdTarget | low_altitude_suspect | low_altitude_suspect | low_altitude_suspect |  |  | 0.0000 |
