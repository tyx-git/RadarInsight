结论：可以接受已知类小幅退让，但不能继续提高同一个全局 knownness 阈值。V27 下一阶段应改成“风险受约束的开放集优化”：冻结 V27a 分类端点，重点提高 Bird-route Unknown 的表征可分性，再由动态 L3 寻找 Pareto 工作点。

当前证据
- V27a-r3 仍是最可靠的分类基座：215/223，Accuracy 96.41%，Macro-F1 96.14%，相对 baseline 净纠正两个 Bird。
- V27b-r6 在 development 的 99% known-recall 点，把 Unknown rejection 从 28.7% 提到 41.7%，即 +13.0pp；但在独立 calibration 上没有复现，seed42 反而从 13.3% 降至 6.7%。
- calibration 只有 15 个 eligible Unknown，每个样本就代表 6.67pp，当前不足以证明“大幅提升”。
- Unknown 高度集中于 Bird 路线：development 有 35/38、calibration 有 13/15 被 L2 预测为 Bird。因此仅给 Drone/Balloon 使用更严格阈值，收益上限很低；核心仍是 Bird-vs-Unknown 表征。

- V27c-r21/r22/r23 已基本排除当前 TR motion 作为分类输入的价值；motion coverage 更适合作为难度/质量分层变量。
- V27c seed43 的 216/223 是重训随机性结果。现有 logits ensemble 不能保留它额外修正的 Bird，因此不建议直接换基座。

审查问题处理状态
1. 已修复并由 endpoint materialization 验证：L1 继续使用冻结 V27 epoch 14，L2 已绑定冻结 V27a Main epoch 20；已知 parent 的 baseline/V27a replay 最大误差为 2.38e-6，小于 3e-6。当前完整三层 control 不再悄悄回退到 V27 baseline L2 epoch 11。
2. 已修复并由 feature cache/smoke 验证：L1 knownness population 为 DBB、Clutter 和 Unknown，低质量 parent 不进入 fit；L2 knownness population 仅为 DBB 与 Unknown，Clutter 仅诊断。r1 smoke 中 L1 Clutter fit eligible=42，L2 Clutter fit eligible=0，Unknown knownness population=168。
3. 已修复并由 r1 训练验证：knownness checkpoint 选择使用 development 上逐类 known-recall safety floor 和 Unknown rejection，而不是普通 AUROC；quality head 仍记录 development AUROC。L1 knownness 选中 epoch 2（AUROC 0.896356），L2 knownness 选中 epoch 16（AUROC 0.933173）。
4. 已修复并由 r1 calibration 验证：L3 阈值搜索加入逐类 known recall、正确已知新增拒识总预算、每类正确已知拒识预算，以及 Clutter recall floor；当前 calibration pool 上 L1/L2 selection 均不可行。最低 conformal p 阈值 0.1 下，L1 已知目标 route recall 的最高最小值约 0.904545，L2 三类 route recall 的最高最小值约 0.909091，均低于 0.98，因此 registry 正确保持 unavailable，没有活动阈值。
5. 已修复并由 r1 evaluation 验证：L3 open-set 统计以联合质量 eligibility 为分母，并分别报告 known route recall、正确/错误已知接受、Unknown 严格拒识、Unknown 错误接受和 Pending。未校准时全量 fail-closed 为 PENDING_ABSTAIN，Unknown Pending 不计作拒识成功。
6. r1 有意不启用 raw-scale 和 stable8 增强。Liquid V2 的原始尺度信息保留问题是 r2 的唯一表征变量，不能把 r1 control 当作该问题的解决。
7. 已按当前协议处理：确定性质量规则承担硬 Pending，learned quality 只训练并作为诊断/advisory；L3 动作不由 learned quality 回流。
8. 已按当前协议隔离：head fitting 使用 representation_fit，checkpoint selection 使用 representation_development，动态校准使用 calibration role；但阈值与 conformal 参考当前仍在同一 calibration pool 选择，因此 registry 明确标记 internal_exploratory，不具备正式 coverage/deployment 资格。
9. 已修复并由 GPU 训练和 calibration 复核设备边界：head checkpoint 以 CPU 加载后，校准前显式将 head 移到目标 device；scorer 的协方差单位阵也跟随 projection device。r1 训练、calibration 和 evaluation 均已完成。

10. 已修复并通过 smoke/重建复核的工程边界：feature cache schema 已升为 v2 以包含 baseline sequence logits；cache canonical-path 校验不再被局部变量覆盖；校准 p-value 候选在观测值低于 alpha 时仍保留 alpha cutoff；endpoint/representation 显式校验 parent label 顺序和 L1/L2 calibration length 顺序；配置 expected-hash 键集合必须与声明源文件完全一致；V27d 日志同时写入 artifact-local 文件和 stderr。

推荐实验顺序
实验       唯一主要变量                                  目的
━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 V27d-r1    接入冻结 V27a logits并修正人口、mask、指标    建立可信的完整三层 control；本轮不宣称模型提升
─────────  ────────────────────────────────────────────  ─────────────────────────────────────────────────
V27d-r2    增加显式 raw-scale 轨迹统计                   验证 Liquid V2 保留的尺度是否改善 Unknown 排序
─────────  ────────────────────────────────────────────  ─────────────────────────────────────────────────
V27d-r3    在 r2 上加入 stable8 因果稳定性               复核 r6 的尾部正信号能否跨 calibration 保持
─────────  ────────────────────────────────────────────  ─────────────────────────────────────────────────
V27d-r4    32维低维空间及多原型类条件 nonconformity      专门解决 Bird-like Unknown 与多模态已知流形重叠
─────────  ────────────────────────────────────────────  ─────────────────────────────────────────────────
V27d-r5    类条件/质量分层动态校准和完整 L1→L2→L3        在严格风险预算下选择最终 Pareto 工作点
V27d-r2 建议显式加入、但不改变 Liquid 状态方程：
- log ||r_t||
- 末次事件占累计更新比例，用于识别“单个高置信事件拉动终态”
V27d-r4 不应只做一个 Bird-vs-Unknown 二分类器。建议使用：
- 全局 knownness；
- predicted-class 条件 nonconformity；
- Bird-route 专门的多原型距离；
- class-conditional conformal set。
原型数、余弦、kNN、收缩协方差 Mahalanobis应在同一32维投影上消融，不能同时改变空间和评分公式。建议的“略微妥协”预算
在当前 calibration 的218个 eligible known上，先预注册：
  - KnownTarget DBB route recall ≥98%，即最多4个已知不进入DBB动作；
  - 原本分类正确的210个样本最多新增拒绝2个；
  - 每个类别最多新增损失1个正确 parent；
  - Clutter recall ≥8/9，即88.89%；
  - 低质量 Pending不计作Unknown拒识；
  - Unknown rejection相对 matched control至少提升15pp，并在 calibration和development同方向；
  - 数据量充足后，paired bootstrap置信区间下界必须大于0。
第一阶段可以把绝对 Unknown rejection ≥50%作为里程碑；达到60%以上且正确已知损失≤1%才算真正的大幅提升。当前15个 calibration Unknown无法可靠支持该结论，建议扩充至至少100个独立Unknown，最好200个以上，并另留最终 audit；交叉拟合只能提高数据利用率，不能替代新增独立样本。
