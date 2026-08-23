# V24c strict same-second recurrent-state-skip diagnostic

实验版本：V24c。

变量：在完成的 V24 KnownGate L1 基线之上，仅将
`liquid_invalid_dt_update_mode` 从 `bounded_observation` 改为
`strict_skip`。当冻结的 `tr_dt_reason` 为
`zero_delta_at_second_precision` 时，该事件的融合输入会照常构造，但
递归状态在 recurrent LayerNorm 之后恢复为进入该步的精确状态。首事件
仍以 alpha=1 初始化，正 dt 仍使用原连续时间更新，padding 不更新。

验证：同秒观测的递归累积是否携带可泛化的因果证据，还是类别/采样捷径。
V24c 复用 V24 的 V21e causal-course train/validation cache，训练 loss、
模型宽度、特征、随机种子、parent 顺序、checkpoint 规则与 20 epoch
均冻结；V24 已完成的 checkpoint 仅作为开发配对报告参照，绝不作为
V24c 训练初始化或微调来源。

预注册审计：报告 parent-final、eligible-prefix、parent-balanced
eligible-prefix、window、Clutter support=9 的 Wilson 区间、按 parent 的
bootstrap、同秒 follow-up/最大同秒序位/轨迹长度/capture 分层，以及逐
parent 改变。Clutter 在训练和验证均没有同秒 follow-up，这一事实仅作
描述性混杂审计，不作为训练 readiness 或模型输入。

边界：本实验仅是 KnownGate L1 开发诊断；OOF、test、Unknown、L3、完整
级联、正式拒识、stable 与 P95 均不执行，也不得据此作部署或产品动作声明。
