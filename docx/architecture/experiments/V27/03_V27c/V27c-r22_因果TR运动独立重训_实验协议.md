# V27c-r22：因果 TR motion 的独立 L2 端到端重训

## 问题

V27c-r21 只训练了一个冻结 V27a Main 端点外的末端 motion residual。三个
seed 都安全，但没有改变任何一个 validation parent 的预测。这个结果只能排除
“小容量、末端 logit 修正”这一实现，不足以否定 `causal-tr-motion-v1` 的价值。

R22 因此检验更强、但仍可归因的问题：motion 数值若在每个事件进入 Liquid V2
更新之前，能否在重训的 L2 表征中改善 Drone/Bird/Balloon 分类？

## 固定边界

- L1 固定为 V27 epoch-14：只绑定其 hash，不加载、不前向、不优化；
- L2 从 V27 epoch-11 分类器和 V27a Main epoch-20 stable8 证据分支初始化；
- 保留 V27a 已验证的可靠 stable8 全帧路径，不能回退为只使用压缩 summary；
- 只读取已知 DBB 的 train/validation；Unknown、Clutter、knownness、quality、
  calibration、L3 与 test 均不参与；
- 训练损失继续是固定 parent 分母的 class-balanced final CE；不使用验证集选择
  checkpoint，固定训练 20 epoch。

## 两个匹配 arm

| Arm | 训练项 | 唯一输入差异 |
|---|---|---|
| `control` | L2 base、V27a Main stable8 分支、同容量 motion adapter | motion adapter 接收全零数值 |
| `motion_treatment` | 与 control 完全相同 | adapter 接收经一次 validity gating 的 9 维因果 motion 数值 |

motion adapter 是无 bias 的 `9 -> 16 -> 48` SiLU 残差。其输出层零初始化，故两臂
在训练前精确同态；adapter 输出被加到原始 fusion event 后、Liquid V2 递归之前。
因此它能影响整条后续 raw-memory 因果轨迹，而不是仅修改 parent-final logits。

`tr_motion_valid_mask` 仅在 dataset sidecar materialization 时用于把无效数值置零；
它不会进入模型、checkpoint 或学习特征。可靠 stable8 的 frame mask 是已有微多普勒
分支的必要帧 padding 合同，与 motion validity mask 不是同一种输入。

## 评价

每个 seed 的报告同时包含：

- V27a Main 冻结参考、匹配 control、motion treatment，以及 treatment 的全零
  motion 反事实；
- parent-final 与 parent-balanced eligible-prefix 指标、逐 parent paired transition、
  四个既有 `Drone -> Bird` 高置信错误审计；
- 分层 paired bootstrap；
- length、活动 token support、分量 support 三种不读取 label 的 motion 置换；
- 已知端点安全和 motion 增益分别判断。任何结果均不构成 Unknown 拒识、L3 或部署结论。

## 结果

训练和评估尚未开始。结果将在三 seed 全部完成后追加到本文件。
