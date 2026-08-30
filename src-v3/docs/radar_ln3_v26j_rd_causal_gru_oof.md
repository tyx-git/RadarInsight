# RADAR-LN3 V26j：事件内 RD 因果 GRU 聚合

实验版本：V26j。

直接基础实验：V26e L1 fixed epoch-32 Known-only 五折 OOF。

变量：仅将 `model.rd_frame_aggregation` 从 `masked_mean` 改为
`causal_gru`。GRU 只在单个 RD 事件的帧序列内更新；它不会跨 TR 事件、
parent、batch 或 fold 传递状态。

固定：CQ08 V16 index、V21e provenance-grouped 五折、seed=43、batch=2、
32 epoch、loss、课程圆形离散度特征、dt 语义、自研 CfC-style 外层主干、
V26c L2 epoch-15、Known-only 外层交叉拟合 L3 协议。

UnknownTarget 不进入 L1 训练、归一化、checkpoint 选择或 L3 阈值拟合。
混合 Known/Unknown 验证与测试只在五个 L1 checkpoint、L2 checkpoint 和
L3 阈值全数冻结之后执行。`PENDING_ABSTAIN` 不计为 Unknown 拒识成功。

## 防止语义偏移的实现约束

- V26j 不能以 V26 的 `masked_mean` 模型构建后再覆写 `model.config`。
  该做法不会创建 `GRUCell`，属于配置已变而真实主干未变。
- 每一折 runtime、checkpoint、prediction、summary 和混合审计都会验证：
  `config=causal_gru`、`rd_encoder.aggregation_mode=causal_gru`、
  `rd_encoder.temporal` 为 `GRUCell` 且其参数实际存在于 `state_dict`。
- 除 `rd_encoder.temporal.*` 外，V26j 从 V26e 等价 masked-mean 初始化中复制
  全部形状兼容的共享参数。因此不是随机初始化差异或特征改动实验。
- RD padding mask 为 false 时 GRU state 保持不变；烟雾测试会验证：改变 padded
  帧不改变输出、扰动后续有效帧会改变输出、零 RD 帧输出有限且为零。

## 执行顺序

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  F:\Project\RadarInsight\src-v3\scripts\v26j_known_gate_rd_causal_gru_oof_pipeline.ps1 `
  -Stage prepare -ProjectRoot F:\Project\RadarInsight

powershell -NoProfile -ExecutionPolicy Bypass -File `
  F:\Project\RadarInsight\src-v3\scripts\v26j_known_gate_rd_causal_gru_oof_pipeline.ps1 `
  -Stage smoke -ProjectRoot F:\Project\RadarInsight

# 五折全量训练；手动启动
powershell -NoProfile -ExecutionPolicy Bypass -File `
  F:\Project\RadarInsight\src-v3\scripts\v26j_known_gate_rd_causal_gru_oof_pipeline.ps1 `
  -Stage train -ProjectRoot F:\Project\RadarInsight

# 训练完成后：Known-only L3 calibration，再进行冻结 mixed L1->L2->L3 audit
powershell -NoProfile -ExecutionPolicy Bypass -File `
  F:\Project\RadarInsight\src-v3\scripts\v26j_known_gate_rd_causal_gru_oof_pipeline.ps1 `
  -Stage calibrate -ProjectRoot F:\Project\RadarInsight

powershell -NoProfile -ExecutionPolicy Bypass -File `
  F:\Project\RadarInsight\src-v3\scripts\v26j_known_gate_rd_causal_gru_oof_pipeline.ps1 `
  -Stage audit -ProjectRoot F:\Project\RadarInsight
```

`all` 只按上述次序编排，并不会改变单因素定义；由于其中包含五折 32 epoch
训练，应由人工显式启动。
