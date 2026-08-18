# V21i：V21e + 因果 d_range 稳定性嵌套 OOF

## 定位

V21i 是 KnownGate 层的单因素探索性实验。Control 为封存的 V21e OOF Treatment，Treatment 只追加 V21d 的因果前缀特征 `causal_prefix_mtd_minus_tr_range_sample_std_m_over_75m`，检验它是否能补充 Clutter/Bird 边界。

V21d train-only parent 诊断的 Clutter/Bird 方向无关 AUC 约为 `0.6471`，属于偏弱但可检验的候选；V21d 既有 development 硬动作没有变化，所以 V21i 必须按 V21e 外折增量重新判断，不能把先验 AUC当作收益。

## 特征与因果语义

该特征只累计当前事件以前已经观察到的有效 `MTD detected range - TR filtered range` 残差，样本标准差按冻结的 V21d prefix 公式计算；不足两个有效观测为 invalid，缺失事件不添加未来信息。V21i 输入顺序为：

```text
V21e causal_prefix_course_circular_incoherence
+ causal_prefix_mtd_minus_tr_range_sample_std_m_over_75m
```

## 配对协议与门槛

五折 assignment、fold-train normalization、class weight、seed=43、batch、loss、20 epoch 和固定最后 epoch 预测全部复用 V21e OOF。仅新增一列 Fusion 权重且初始为零；全参数从头训练。

主要边界为 `BirdTarget / ClutterTarget`；Drone、Balloon 为安全标签。冻结验收要求 parent Macro-F1 严格提升 `>0.02`、Bird/Clutter 主要标签合计净救回为正且各自不受损、KnownTarget/Clutter Recall 不下降、component bootstrap 下界大于零，并满足每类 P/R/F1 `>=0.90` 的绝对护栏。

Clutter support 很小，必须同时输出 parent support、分层 bootstrap 和错误样本方向；弱软概率改善不能替代硬动作门槛。

## 封存边界与命令

UnknownTarget、validation/test payload、阈值拟合、正式拒识、完整级联、stable、P95 和部署均未执行。通过也只允许进入新的独立确认。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File F:\Project\RadarInsight\src-v3\scripts\v21i_known_gate_oof_pipeline.ps1 -Stage all
```
