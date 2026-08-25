# V25e：共享模块 dominant-gradient conflict projection

## 问题

V25c 的 V25a 精确回放显示，KnownTarget 与 ClutterTarget 混合 batch 在
`tr_encoder`、`fusion` 和 `liquid` 中持续出现负梯度余弦。由于 Clutter
梯度范数通常更大，历史 batch loss 可能让两个任务在共享表示层互相抵消。

## 唯一变量

V25e 的 control 是已完成的 V25a seed 43。Treatment 只在严格的
`[KnownTarget, ClutterTarget]` 两 parent batch 中修改共享模块梯度：若两个
parent 梯度点积为负，保留较小范数梯度，并把较大范数梯度投影到较小范数
梯度的正交补空间。投影模块仅为 `tr_encoder`、`fusion`、`liquid`。

`rd_encoder`、`classifier`、纯类 batch、loss、类别权重、特征、dt、采样、
初始化、优化器、clip、checkpoint 规则、数据顺序和 20 epoch 均冻结。


\[
g_d' = g_d -
\frac{\langle g_d,g_s\rangle}{\lVert g_s\rVert_2^2}g_s,
\qquad \langle g_d,g_s\rangle < 0
\]

其中 `d` 是较大范数的一侧，`s` 是较小范数的一侧。若点积非负或较小
梯度为零，则不投影。

## 机制审计

训练器先执行原始 V25a fused loss 的 `backward(retain_graph=True)`，再用
显式 parent-term proxy 计算投影增量，并只把增量加入三个目标共享模块的
`.grad`。因此 RD encoder 和 classifier 的历史梯度必须逐 tensor 保持不变。
训练摘要只保存匿名 batch/module 计数与有限标量，不保存 parent、track 或
capture 标识。

## 预注册开发门槛

1. parent-final Clutter recall 为 9/9、Clutter precision ≥ 0.90、KnownTarget
   recall 为 223/223。
2. parent-balanced eligible-prefix 的 Clutter precision/recall 不下降，
   Clutter F1 至少提升 2pp，KnownTarget recall 和 macro-F1 不下降。
3. 机制遥测证明总更新 9860 个，其中 638 个混合 batch、9222 个纯类 batch；
   RD/classifier 梯度精确不变，且三个目标共享模块均发生有效投影。

Clutter validation support 只有 9 个，报告必须同时给出 support 和 parent
bootstrap。该实验不执行 Unknown、OOF、test、L3、cascade、rejection、stable
或 P95，因此通过不代表产品放行。
