# RADAR-LN3 V21f：V21e + 谱形时间一致性探索性 OOF 协议

## 定位

V21f 是 KnownGate 层的后验探索性联合实验：在已经完成的 V21e 因果航迹一致性基础上，追加 V21a 的六维 inter-frame 谱形时间一致性特征，用来检验它是否能救回 V21e 的困难 Bird。该实验不是新的产品级联、拒识校准或部署实验。

V21f 的特征是在查看 V21e 同一开发域 OOF 错误后提出，因此存在 post-selection bias / look-elsewhere 风险。即使五折结果为正，也只能决定是否设计新的独立确认实验，不能直接证明泛化收益。

## 研究问题与预注册方向

主要问题：在固定 V21e 课程一致性输入、数据分组、损失、训练预算和随机种子后，追加谱形时间一致性，是否在外折 parent-final 预测中产生增量收益，尤其是 V21e 已知的困难 Bird？

谱形特征来自当前事件及其因果可用的相邻 RD 帧，不读取标签、未来事件、parent-final 统计或路径标识。六个值为：

1. `current_event_adjacent_rd_velocity_profile_cosine_mean`
2. `current_event_adjacent_rd_velocity_profile_cosine_minimum`
3. `current_event_adjacent_rd_peak_velocity_drift_mean_fraction`
4. `current_event_adjacent_rd_peak_velocity_drift_max_fraction`
5. `current_event_zero_velocity_energy_fraction_std`
6. `current_event_zero_velocity_energy_fraction_minimum`

V21f 的七维输入顺序为：

```text
V21e causal_prefix_course_circular_incoherence
+ V21a six inter-frame spectral-consistency values
```

特征缓存只保存 `sample_ids / values / valid`；sample ID 和 valid mask 是对齐与缺失语义，不是模型输入捷径。

## 配对设计

Control 不重新训练，直接复用同一五折中的 V21e Treatment OOF 预测。每个 V21f Treatment fold：

- 使用与 V21e 完全相同的 `provenance_component_id` 五折分配；
- 使用相同的 fold-train normalization、class weight、seed=43、dataloader seed、batch、loss 和 20 epoch 固定预算；
- 递归重建对应 V21e Treatment 的随机初态；
- 复制所有已有参数和 buffer，仅把新增六列 Fusion 权重精确初始化为零；
- 从头训练全部参数，不加载 V21e 已训练 checkpoint 微调；
- 最后一个固定 epoch 才对 held-out fold 做一次预测，不在外折选择 epoch。

因此，配对比较的基准是 V21e Treatment，而不是 V21 baseline。V21f 初态 hash 必须等于对应 V21e Treatment prediction 的 `initial_model_state_sha256`；初始 logits 必须逐元素相等。

## 数据范围与封存边界

- 已知 train parent：986；provenance component：975；五折 parent 预测单位。
- 训练标签仅为 `DroneTarget / BirdTarget / BalloonTarget / ClutterTarget`。
- `UnknownTarget` 不进入训练、特征归一化、OOF prediction 或阈值拟合。
- validation、test payload、正式拒识、stable、P95 均保持 sealed / 未执行。
- Clutter 的外折 support 每折约 6–7 个，汇总报告必须同时给出 support；低 support 下的点估计不能解释为稳定总体性能。

## 困难 Bird 后验归因

V21e OOF 的固定归因人口为：14 个错误 Bird、225 个正确 Bird、33 个 Clutter。该人口只用于提出和描述 V21f 假设，不用于重新拟合阈值或修改标签。当前归因报告要求六个谱形值在 14 个困难 Bird 上全部有效，并报告：

- Clutter 对困难 Bird 的预设方向 AUC；
- Clutter 对正确 Bird 的同方向 AUC；
- 困难 Bird 对正确 Bird 的方向无关 AUC；
- 各组支持数与分位数。

这些 AUC 是机制线索，不是无偏效果量；最终判断以 V21f 外折 parent 配对结果为准。

## 冻结决策规则

增量信号需要同时满足：

- parent Macro-F1 严格增加 `> 0.02`；
- Bird net rescue `> 0`；
- Drone、Balloon、Clutter 的 net rescue 均不为负；
- KnownTarget Recall 与 KnownClutter Recall 不下降；
- provenance component 分层 paired bootstrap 的 Macro-F1 95% 下界 `> 0`。

绝对护栏仍要求 Treatment 每类 Precision、Recall、F1 最低值达到 `0.90`。绝对护栏失败时，结果只能记录为探索性失败，不能进入阈值校准或产品动作。

即便上述条件通过，V21f 也只允许晋级新的独立确认；以下动作仍禁止：

- 使用 V21f OOF 拟合 Unknown / Clutter 拒识阈值；
- 读取 test 或宣称 test 泛化；
- 把 KnownGate 二分类结果写成五类产品性能；
- 直接进入 stable、完整级联或部署。

## 执行入口

准备、契约 smoke 和静态隔离检查：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File F:\Project\RadarInsight\src-v3\scripts\v21f_known_gate_oof_pipeline.ps1 -Stage prepare
```

五折 Treatment 长训练及最终配对汇总：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File F:\Project\RadarInsight\src-v3\scripts\v21f_known_gate_oof_pipeline.ps1 -Stage all
```

脚本按 fold 串行执行，固定 20 epoch；中断后可再次执行同一命令，脚本只从当前 fold artifact 的合法 `last_model.pt` 恢复。任一契约、hash、训练摘要或预测检查失败都会停止。

## 产物状态字段

V21f readiness、fold prediction 和 paired report 必须明确记录：

```text
Unknown training exposure = 0
validation access = 0
test payload access = 0
threshold fitting = 未执行
formal rejection = 未执行
stable = 未执行
P95 = 未执行
deployment_eligible = false
```

V21f OOF fold checkpoint 是评估支持产物，不是部署 checkpoint。

## 当前已知限制

谱形困难 Bird 归因来自同一 CQ08 域、同一开发错误分析流程；五折仍不能替代跨日期、跨环境、跨传感器的独立 holdout。Clutter support 很小，任何 Clutter 召回变化都必须附带 parent support 和组件 bootstrap，不得仅凭单折或单个样本做物理结论。
