# RADAR-LN3 V21e 多seed配对确认协议

状态：训练前冻结。该阶段只确认V21e因果航向圆周不一致度的seed43 parent-final正向信号能否跨优化随机性复现，不修改特征公式、数据、loss、epoch、checkpoint选择或KnownGate类别。

## 1. 固定实验臂

- seed集合固定为`42/43/44`。三套V21 Baseline均已完成全参数20 epoch训练，作为不可变同seedControl复用，不重新选Control。
- seed43 V21e Treatment已完成全参数20 epoch训练并通过单seed门槛，只读复用。
- 新训练仅为seed42和seed44 V21e Treatment。二者各自从对应Baseline的同seed随机初态复制全部共享参数，仅将新增一列Fusion精确置零，再从头训练全部参数20 epoch；禁止加载已训练Baseline或seed43 Treatment微调。
- 每个seed的Control/Treatment必须具有相同DataLoader seed、逐epochparent顺序、loss、class weight和checkpoint排序。不同seed的初始化必须互异。

## 2. 不变合同

- 复用V16 capture/content grouped index、V21 known-only normalization、V21e一维train/validation cache及原实验合同，不重建raw、MAT或RD数据。
- UnknownTarget继续不进入KnownGate类别、训练、归一化、checkpoint选择或本轮指标。
- test、正式拒识、完整级联、stable和P95均不执行；原development已用于V21a-e特征选择，仍不是无偏确认集。

## 3. 冻结决策

parent-final主信号确认要求同时满足：

1. 至少2/3个seed的Treatment−Baseline parent Macro-F1严格大于`0.02`；
2. 三seedMacro-F1差值中位数严格大于`0.02`；
3. 至少2/3个seed的四标签合计parent net rescue为正、各标签net rescue均非负，并且KnownTarget与Clutter Recall不下降；
4. 至少2/3个seed的四标签分层paired parent bootstrap Macro-F1差值95%下界大于0。

满足上述条件只允许进入train parent-grouped OOF。它不直接允许采用checkpoint、执行test或声明open-set收益。

eligible-prefix另设独立在线安全诊断：至少2/3个seed且三seed中位数均需保持Macro-F1、KnownTarget Recall和Clutter Recall不下降。该诊断不篡改parent-final主结论，但失败时禁止把V21e称为在线KnownGate改进，必须先处理pending/stable与早期动作语义。

## 4. 统计边界

三个seed是同一批232个development parent上的优化重复，不是三个独立数据样本，不计算三seedp值。Clutter每个seed仍只有9个parent；parent bootstrap只量化当前parent集合内的不确定性，不能覆盖环境、session、未标注物理子型或新采集分布。V21e来自查看既有development错误后的事后发现，即使多seed通过，仍需train parent-grouped OOF和新的独立holdout。
