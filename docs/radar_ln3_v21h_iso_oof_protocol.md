# V21h：V21e + 目标行能量占比 iso 嵌套 OOF

## 定位

V21h 是 KnownGate 层的单因素探索性实验。Control 为已经封存的 V21e OOF Treatment，Treatment 只在其一维因果航迹一致性后追加一维 `target_row_linear_energy_fraction_mean`，用于检验 iso 是否仍能补充 Clutter/Drone 边界。

V21c 独立实验的 train-only parent 方向无关 Clutter/Drone AUC 约为 `0.7771`，但其 development 硬动作没有变化。因此 V21h 不是重复宣称 V21c 成功，而是检验该证据是否能在 V21e 新表征下提供增量。

## 配对协议

- 五折分组完全复用 V21e 的 `provenance_component_id` assignment；986 个 known train parent，975 个 component。
- V21e Treatment OOF prediction 作为冻结 Control，不重新训练 Control。
- Treatment 递归重建 V21e Treatment 初态，复制已有 course 列，仅新增 iso Fusion 列精确置零。
- 全部参数从头训练 20 epoch；不加载 V21e checkpoint 微调，不用 held-out fold 选 epoch。
- 只在最后固定 epoch 对 held-out fold 预测一次。

## 主要验收

主要边界为 `DroneTarget / ClutterTarget`；Bird、Balloon 作为安全标签。冻结门槛为 parent Macro-F1 严格提升 `>0.02`、主要标签净救回为正且各标签不受损、KnownTarget/Clutter Recall 不下降、component bootstrap Macro-F1 95% 下界大于零，同时 Treatment 每类 P/R/F1 最低值达到 `0.90`。

Clutter support 每折约 6–7 个，汇总必须报告 parent support 和 component bootstrap；不能用单点 Clutter Recall 做放行结论。

## 封存边界

UnknownTarget 不进入训练、归一化或 OOF；validation/test payload、阈值拟合、正式拒识、完整级联、stable、P95 和部署均保持未执行。即使通过，也只能进入新的独立确认，不得直接上线。

## 执行命令

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File F:\Project\RadarInsight\src-v3\scripts\v21h_known_gate_oof_pipeline.ps1 -Stage all
```

中断后重复同一命令即可从当前 fold 的合法 `last_model.pt` 恢复。五折完成后脚本自动生成 paired report。
