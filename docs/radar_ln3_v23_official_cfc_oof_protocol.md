# RADAR-LN3 V23 官方 CfC OOF 协议

## 目标

V23 检验项目原有的 `CfCLiquidCausalEncoder` 是否应被官方 Neural Circuit Policies (`ncps`) 的标准 `ncps.torch.CfC` 替代。该实验仅覆盖 KnownGate 的 known-only、capture/content provenance-component grouped 五折 OOF；不产生完整产品、五分类、Unknown 拒识、L3、stable、P95 或 test 结论。

## 两个独立对照

| 实验 | 冻结 Control | Treatment | 唯一结构变量 |
| --- | --- | --- | --- |
| V23a | V21e OOF Treatment，`batch_weight_sum` | 官方 CfC + 相同 V21e loss | 项目自定义 CfC-inspired encoder → 官方 `ncps.torch.CfC` |
| V23b | V22 OOF Treatment，`fixed_parent_count` | 官方 CfC + 相同 V22 loss | 项目自定义 CfC-inspired encoder → 官方 `ncps.torch.CfC` |

V22 仍是历史已执行的 fixed-parent-denominator loss 基线；它不会被改写为官方 CfC。否则 V23b 会同时改变 loss 与 encoder，失去单因素解释。

## 官方实现与固定规格

- distribution：`ncps==1.0.1`
- 官方源码：`https://github.com/mlech26l/ncps.git`
- 固定 commit：`695ec22d7d1831f2ced67346ef7d2f08c525bd47`
- 实际类：`ncps.torch.CfC`，其内部状态方程由官方 `ncps.torch.cfc_cell.CfCCell` 持有；Radar-LN3 不重写该方程。
- `mode=default`、`activation=lecun_tanh`、`mixed_memory=false`、`return_sequences=true`、`batch_first=true`、`backbone_layers=1`、`backbone_dropout=0`。
- `hidden_dim=48`、`backbone_units=48`。历史自定义时序模块参数量为 14,112，官方 CfC 为 14,064，差 48 参数；不使用官方默认的 128 宽 backbone，避免容量因素混入。

## 时间与在线契约

官方 CfC 接收真实非负 wall-clock `dt`：first event 和同秒观测为 `0`，positive delta 保留实测秒数并仅上截断至既有 120 秒。历史 `floor/bounded_observation` 增益不注入官方 CfC，这是 encoder 替换的内在差异，必须如实报告。

上游 `ncps==1.0.1` 的 batch time-span wrapper 对 `[B]` time-span 有广播限制。V23 按 parent 调用未修改的官方 `CfC.forward()`，并只传入真实有效长度；padding 永不进入 recurrence。Smoke 必须验证 online 单点与 offline 序列 logits 精确一致。

## 冻结条件

- V16 index、V21e 五折 assignment、fold-train normalization、course cache 和 feature 顺序保持不变。
- 已知标签仍为 Drone/Bird/Balloon→KnownTarget、Clutter→KnownClutter；Unknown 不是训练类别。
- batch size、seed 43、每 epoch parent 顺序、AdamW、学习率、clip、20 epoch、prefix weight、fixed-final-epoch held-out protocol 保持来源实验的值。
- 每个 V23 epoch 的 parent-order SHA-256 必须等于对应冻结 Control epoch；架构不同，禁止声称初始参数或 logits 相同。
- Unknown、validation、test、阈值拟合、正式拒识、完整级联、stable 与 P95 均为“未执行”。

## 通过门槛

必须同时满足：parent Macro-F1 严格超过 Control `+2pp`、KnownTarget 与 Clutter Recall 不下降、Drone/Bird/Balloon 方向净救回不为负、eligible-prefix 的 Macro-F1/两类 Recall 不下降、capture/content component bootstrap 的 Macro-F1 差值 95% 下界大于 0。Clutter 每折仅 6–7 parent，报告必须保留 support 与 bootstrap，不得把单折波动写成跨源泛化结论。

不满足时冻结为 `hypothesis_not_supported` 或低于采纳门槛的诊断结果，不进入 L3、校准、test 或部署。
