# V26c L1/L2 Known-only OOF L3 calibration

实验版本：V26c。

变量：不改变已冻结 V26 或 V18r 的全量模型。V26c 仅复用 V21e 已封存的
capture/content provenance-component 五折 **parent assignment**，并为每一折
分别从头训练：

- L1 KnownGate：只用该折的 Known 四类训练 parent、折内 RD normalization、
  label-free causal-course cache 切片；固定使用第 14 epoch 快照；
- L2 DBB specialist：只用该折的 Drone/Bird/Balloon 训练 parent、折内 RD
  normalization、固定的 V18r RD-physical4 + stable8 特征协议；固定使用第
  15 epoch 快照。

Clutter 不进入 L2 的训练或 normalization；Unknown 不进入 L1/L2 的训练、
normalization、采样、特征选择、checkpoint 选择或 L3 阈值拟合。V18r 的
stable8 特征协议是 V26c 前已经冻结的输入协议，不在 OOF 中根据 held-out
标签重选。

验证：每一个 Known train parent 在 L1 中恰好作为 held-out parent 一次；每个
Drone/Bird/Balloon train parent 在 L2 中恰好作为 held-out parent 一次；任何
capture/content provenance component 不跨 train/held-out。每折所有模型均用
折内训练数据重新初始化和训练，不能复用 V26、V18r 或 V21e 的模型状态、
normalization、cache 数值或预测。

L3 后续只允许使用这些 Known parent-final OOF 分数：对外折 `f` 的三个接受
阈值只可由其余四折 OOF parent-final 记录拟合。该阈值原样用于外折，随后
才可冻结 registry 并进行混合 Unknown 审计。Pending/Abstain 永远不能算作
Unknown 拒识成功。

边界：V26c 的折模型只供 OOF 校准，不是部署 checkpoint；prepare/训练阶段
不读取 validation/test，也不产生 Unknown 拒识、L3 放行、稳定性、P95 或
部署结论。
