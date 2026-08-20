# RADAR-LN3 V22b：RD 时间残差后置读出单因素实验

## [KnownGate]

### [Direct:验证 V22a 负收益是否来自 residual 注入液态状态]

[]1. Control 固定为 V21e seed43 五折 OOF Treatment predictions：复用同一 provenance-component fold assignment、每折 train-only normalization、初始状态、parent 顺序、20 epoch、loss 与 fixed-final-epoch 规则；不重复训练 Control。

[]2. Treatment 使用与 V22a 完全相同的当前事件 RD 时间残差：`last_active_RD_frame_embedding - first_active_RD_frame_embedding`；缺失或单帧 RD 时严格为零，不读取未来事件。

[]3. 唯一结构变量是注入位置。V22a 在 recurrent fusion 前注入；V22b 不让 residual 进入 CfC/liquid state，而在每个事件的 KnownGate logits 处添加零初始化、无偏置、24 参数的反对称校正 `[-0.5·wᵀr, +0.5·wᵀr]`。

[]4. V22b 每折必须从与对应 V21e fold Treatment 相同的初始状态全参数训练 20 epoch。共享 TR/RD/CfC/KnownGate 参数与新增 24 参数均参与训练；冻结主干、从 final checkpoint 继续训练或用 held-out 选 epoch 均不属于本实验。

[]5. V21e seed43 development 已为 232/232，无法提供正向增量空间，因此 V22b 正式验收只使用986个known train parent的五折OOF。采用门槛：OOF parent-final Macro-F1严格提升超过2pp；KnownTarget与Clutter安全、Drone/Bird/Balloon Recall、eligible-prefix安全同时通过，并报告975个provenance component配对bootstrap。

[]6. 诊断解释：若 V22b 正向而 V22a 负向，说明 residual 有独立判别信息但 recurrent 注入有害；若 V22b 中性或负向，则放弃 first-last embedding residual，不再为其做多 seed 或 OOF。

[]7. Unknown 不作为训练类别；本实验只覆盖 KnownGate 局部任务。OOF、test、完整 L1/L2/L3、正式拒识、stable 与 P95 均保持“未执行”。
