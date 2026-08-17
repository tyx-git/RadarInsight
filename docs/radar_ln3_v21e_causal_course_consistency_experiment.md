# RADAR-LN3 V21e 因果航迹一致性单因素实验

状态：训练前冻结。V21e只检验显式的长期航向圆周一致性，能否改善V21 KnownGate的已知目标/Clutter局部边界。UnknownTarget不进入类别、loss、normalization、checkpoint选择或本轮指标。

## 1. 唯一实验变量

Treatment只新增一维`causal_prefix_course_circular_incoherence`：

`1 - |mean(exp(i × course_deg))|`

每个事件只使用当前及历史course。CQ08首事件的speed/course常为初始化零，因此首事件无条件排除，即使其值非零也不参与统计；第二事件及以后合法的真实`0°`必须正常参与。有效非首点course少于3个时输出`0/invalid`；缺失事件不增加观测，但已达到最小support后可以保持既有因果前缀值。值域天然为`[0,1]`，不再调clip或变换。

Baseline已经使用逐点course的sin/cos及因果course变化量。V21e因此检验的是显式长前缀圆周离散度是否提供主干之外的残差信息，并不宣称首次把航向提供给模型。不得同时加入course step、径向符号翻转、V21a–d证据、额外mask或其他超参。

## 2. 配对协议

- 复用V16 index/split、V21 known-only normalization与seed43 development Baseline。
- Treatment从相同seed43随机初态构建，复制全部共享参数，只将新增Fusion列精确置零；不得加载Baseline best checkpoint微调。
- 两臂训练核心、20 epoch、DataLoader parent顺序、loss、class weight与checkpoint排序完全一致；全参数从头训练。
- test不读取、不训练、不评估。OOF、完整级联、正式拒识、stable、P95和部署均未执行。

## 3. 先验与偏倚披露

严格公式的train-only parent-final复算显示，Clutter相对全部KnownTarget的高值AUC约`0.982003`；相对Drone/Bird/Balloon分别约`0.970843/0.986941/0.999363`。这些数值只证明单变量总体分布有方向，不证明它对现有主干错误边界有互补收益。

该候选来自查看V21a–d固定development错例后的事后分析，并且此前已在同一validation上完成多次特征比较，存在明确look-elsewhere风险。因此本轮development结果不是无偏确认。即使通过，也只允许进入multi-seed、train parent-grouped OOF和新独立holdout，不得用于部署或Unknown拒识声明。

## 4. 冻结评估

- 主单位：development validation parent-final KnownGate Macro-F1。
- 采纳门槛：Treatment−Baseline Macro-F1严格大于`0.02`；Drone、Bird、Balloon、Clutter四个原始标签各自net rescue不得为负，四类合计net rescue必须为正；Known Target Recall与Clutter Recall均不得下降。
- 同时报告parent/window/eligible-prefix变化、NLL/Brier、20个same-epoch轨迹、5000次按四原始标签分层的paired parent bootstrap，以及Clutter support=`9`。
- 必须执行固定Treatment checkpoint的直接贡献诊断：保持所有训练权重不变，只将输入的V21e特征置零并重新评估。它用于区分新列的直接推理贡献与全参数训练路径变化，不替代Baseline对照，也不参与checkpoint选择。
- development validation不能用于调整公式、最小support、阈值、符号或epoch。中间epoch优于Baseline不能替代own-best冻结裁决。

## 5. 结论边界

V21e的KnownGate二分类argmax仍不是最终产品动作。`UNKNOWN_REJECTED`与`PENDING_ABSTAIN`必须等待L3 train parent-grouped OOF评分器和质量规则；本实验不输出Unknown概率，也不评估正式open-set拒识。
