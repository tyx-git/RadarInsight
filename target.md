# 阶段性目标：两层标签体系与低空双主干回退机制

更新时间：2026-05-08

## 当前阶段定位

当前项目进入架构再对齐阶段。目标不是继续堆叠零散补丁，而是把标签、专家一低空双主干、System 回退机制重新约束到一个清晰方向：

```text
两层标签训练
轨迹粗筛与专家路由
Expert1 低空轨迹/RD 双主干
Unknown/reject 驱动的回退再分析
最终可输出大类，也可保留细类诊断
```

## 一、标签体系优化

### 设计原则

标签只保留两层：

```text
coarse_label：最终业务大类
fine_label：训练用细类
```

不保留第三层原始标注提示字段，不新增 `source_subtype_hint`、`UnknownDroneSubtype` 等冗余概念。

### 细类裁定原则

细类必须来自可靠标注、数据说明、文件名或明确字段，不能凭经验臆造。

当前明确可用的 CQ-08 标签：

```text
标签 1：小型旋翼无人机
标签 2：轻型旋翼无人机
标签 3：鸟类
标签 4：空飘球
标签 5：杂波
标签 6：其它未识别目标
```

其中标签 1 和标签 2 不继续拆成更细类别，统一作为旋翼无人机细类。

### CQ-08 映射方案

```text
label_id=1 -> coarse_label=DroneTarget,   fine_label=RotorDroneTarget
label_id=2 -> coarse_label=DroneTarget,   fine_label=RotorDroneTarget
label_id=3 -> coarse_label=BirdTarget,    fine_label=BirdTarget
label_id=4 -> coarse_label=BalloonTarget, fine_label=BalloonTarget
label_id=5 -> coarse_label=ClutterTarget, fine_label=ClutterTarget
label_id=6 -> coarse_label=UnknownTarget, fine_label=UnknownTarget
```

### Aircraft / Helicopter 裁定

飞机和直升机是否拆细类，必须等数据源中存在可靠字段或标注后再做。没有可靠证据前：

```text
coarse_label=AircraftTarget
fine_label=AircraftTarget
```

不使用猜测式 `HelicopterTarget` 标签。

### 输出策略

训练期可以使用 `fine_label` 提高表征学习质量；评估和业务输出必须同时支持：

```text
fine 输出：用于诊断、专家内部指标、后续调参
coarse 输出：用于最终业务分类
```

默认对外输出 `coarse_label`。

## 二、Expert1 低空模式约束

Expert1 只保留两种正式推理模式：

```text
dual_main：默认模式，轨迹主干和 RD 主干同时作为低空主证据。
deep_reroute：更高计算复杂度的纠错路径，用于高混淆、reject、Unknown 或回退再分析。
```

`track_only` 不再作为 Expert1 正式推理模式，只作为报告指标和未来快速部署诊断能力保留。

### RD 定位

低空目标识别中，RD 不应只是轨迹低置信时的辅助分支。训练期和默认低空推理期应采用：

```text
有 RD 的低空样本：
  trajectory branch 常开
  RD branch 常开
  fusion/adoption 学习两种证据如何裁决
```

后续优化重点从“是否激活 RD”转为：

```text
RD 作为第二主干如何参与裁决
RD 在 Drone/Bird、Unknown/Clutter 等局部混淆对上何时接管
deep_reroute 何时启动更强 RD 分析
```

### deep_reroute 能力要求

`deep_reroute` 必须明显强于 `dual_main`，不能只是同一套 logits 的后处理。至少应包含：

```text
更高分辨率 RD patch
更多 RD 帧
局部 pair correction
Unknown/Clutter residual 判断
更强的专家头
更强的辅助头
```

第一轮 Expert1 判为 `UnknownTarget` 时，不直接作为最终输出，而是作为 reject signal 触发 `deep_reroute`。只有 `deep_reroute` 后仍判为 `UnknownTarget`，才输出最终 Unknown。

## 三、Expert1 指标口径

Expert1 独立性能必须保留两个口径：

```text
expert1_routed_performance：
  只统计 Router 实际分配给 Expert1 的样本，反映真实运行表现。

expert1_true_low_altitude_performance：
  统计所有真实低空类样本，反映 Expert1 能力上限和 Router 漏分风险。
```

提交 git 时，Expert1 相关指标优先写入 `expert1_routed_performance`，同时保留 true-low-altitude 诊断指标。

## 四、回退机制讨论目标

回退机制需要重新设计，不再满足于一次前向后的 logits 后处理。

已定稿原则：

```text
1. Expert1 第一轮 Unknown 是 reject signal，不是立即最终输出。
2. Expert1 deep_reroute 后仍 Unknown，才输出最终 UnknownTarget。
3. Expert1 deep_reroute 使用更高分辨率 RD patch、更多 RD 帧、局部 pair correction、Unknown/Clutter residual、更强专家头和辅助头。
4. 回退最多两轮，超过后兜底 UnknownTarget。
5. 训练期不启用动态回退控制流。
6. 训练期必须训练 deep_reroute 所需分支。
7. 测试报告必须同时输出“不启用回退”和“启用回退”的对比。
```

回退机制只用于推理、测试和部署路径。训练过程中不启用动态跨专家回退控制流，训练重点仍然是：

```text
精心训练各个专家
训练 Expert1 dual_main 与 deep_reroute 所需分支
训练 Expert2/Expert4 的 accept/reject 能力
训练 System 的专家仲裁链路
```

### 推理流程

```text
Pass 1:
  Router 选择目标专家。
  Expert1 使用 dual_main；其它专家使用常规路径。

Reject:
  专家输出 Unknown、低 accept score、高混淆对、或证据冲突。

Pass 2:
  如果 reject reason 属于低空内部混淆，则回到 Expert1 deep_reroute。
  如果 reject reason 表明专家族不匹配，则重新路由到相邻专家。

Fallback:
  最多两轮回退后仍无法裁决，输出 UnknownTarget。
```

### 训练与评估边界

训练期不做动态回退控制流，避免不可微路由和训练链路复杂化。训练时应全量训练 `dual_main` 与 `deep_reroute` 所需的 RD 主干、局部纠错头、Unknown/Clutter residual 头和辅助头。

测试与报告阶段必须同时评估：

```text
without_reroute：
  只使用第一轮 Router + 专家常规输出。

with_reroute：
  启用 reject/deep_reroute/跨专家回退。
```

报告至少包含：

```text
reroute_trigger_rate
reroute_recovery_count
reroute_recovery_rate
reroute_harm_count
reroute_harm_rate
reroute_fallback_unknown_count
without_reroute_accuracy
with_reroute_accuracy
without_reroute_macro_f1
with_reroute_macro_f1
per_class_before_after_recall
```

## 五、跨专家回退设计

跨专家回退采用有方向的 reject graph，不做无目标的随机尝试。

### 基本原则

```text
Router 只给第一候选专家。
专家自己输出 accept / reject 信号。
reject 后根据 reject_reason 选择下一个专家。
最多两次回退。
全部 reject 后输出 UnknownTarget。
```

Router 在回退链路中只能作为 prior，不能作为 veto。被回退选中的专家不能继续被原始 Router gate 压死。

### 专家 accept / reject 信号

Expert1 低空：

```text
accept:
  dual_main 或 deep_reroute 后，低空已知类置信足够。
  或最终 Unknown residual 成立。

reject:
  第一轮 Unknown 先进入 Expert1 deep_reroute，不立刻跨专家。
  deep 后仍不像低空类，且 Unknown residual 也不强，则跨专家回退。
  trajectory/RD 都低置信，且物理特征不像低空，则跨专家回退。
```

Expert2 常规空中：

```text
accept:
  aircraft_similarity >= threshold。
  aircraft margin 足够。
  route / 物理特征没有强烈反对。

reject:
  aircraft_similarity 低。
  aircraft margin 低。
  更像低空慢速目标或地面目标。
```

Expert3 高空/高速：

```text
accept:
  只接受硬物理规则命中样本。

reject:
  不作为普通回退专家。
  没有高空/高速规则命中时，不参与候选竞争。
```

Expert4 地面：

```text
accept:
  groundness 高。
  People/Car/Ship top1 margin 足够。

reject:
  groundness 低。
  地面三类 margin 低。
  物理特征明显不贴地。
```

### Reject graph

```text
Expert1 reject:
  先进入 Expert1 deep_reroute。
  deep 后仍 reject:
    aircraft evidence 高 -> Expert2
    groundness 高 -> Expert4
    否则 -> UnknownTarget

Expert2 reject:
  low-altitude evidence 高或有 RD -> Expert1 deep_reroute
  groundness 高 -> Expert4
  否则 -> UnknownTarget

Expert4 reject:
  airborne / low-altitude evidence 高 -> Expert1 dual_main 或 deep_reroute
  aircraft evidence 高 -> Expert2
  否则 -> UnknownTarget

Expert3:
  硬规则命中直接接受。
  不命中则不作为回退目标。
```

### 两轮回退定义

```text
Pass 0：Router 初始专家。
Pass 1：第一次回退。
Pass 2：第二次回退。
```

最多三次专家裁决。超过后直接输出 `UnknownTarget`。

### Unknown 语义

Expert1 的 Unknown 必须区分两种：

```text
Pass 0 Unknown:
  reject signal，优先进入 Expert1 deep_reroute。

deep_reroute 后 Unknown:
  如果 residual / unknown score 强，则接受为最终 UnknownTarget。
  如果 residual 也弱，则允许跨专家回退。
```

### 回退后的仲裁

跨专家回退后，不继续使用原始 Router gate 压制被选中的专家。被选中的专家使用 calibrated candidate score：

```text
final_score =
  expert_accept_score
+ small_route_prior
+ class_bias_or_prior_correction
- risk_penalty
```

Router 只能小幅加分，不能否决高置信专家。

### 跨专家回退报告字段

测试报告必须区分：

```text
without_cross_reroute
with_cross_reroute
```

并记录：

```text
initial_expert
final_expert
reject_reason
reroute_path
reroute_count
helped
harmed
unchanged
fallback_to_unknown
per_class_before_after_recall
```

## 六、Reject 细节设计

Reject 不能只依赖一个全局置信度阈值，应采用分层裁决。

### 强制回退

以下情况必须触发回退：

```text
Pass 1 预测 UnknownTarget
top1/top2 属于高风险混淆对且 margin 小
trajectory 与 RD 预测冲突，且 RD 置信度达到可用线
Expert2 判 Aircraft 但 aircraft_similarity 低于阈值
Expert4 判 Ground 但 groundness 低
```

Expert1 第一批高风险混淆对：

```text
DroneTarget <-> BirdTarget
UnknownTarget <-> ClutterTarget
BirdTarget <-> ClutterTarget
DroneTarget <-> UnknownTarget
```

### 条件回退

以下情况需要结合风险和代价判断：

```text
top1 confidence 中等，但 route prior 很低
source 属于弱源 raw_tr_rd，且预测落在低空混淆对
RD 与轨迹一致但两者 margin 都小
专家输出与物理规则明显冲突
```

### 禁止回退

以下情况直接接受，避免过度回退：

```text
Expert1 trajectory 和 RD 一致，且 top1 margin 高
Expert2 aircraft_similarity 明显高于阈值，且 aircraft margin 高
Expert4 groundness 高，且 ground top1 margin 高
Expert3 由硬物理规则命中
```

### Expert1 reject score

Expert1 应统一计算 reject score，而不是散落多个不可追踪的判断：

```text
reject_score =
  unknown_bonus
+ confusion_pair_bonus
+ evidence_conflict_bonus
+ low_margin_bonus
+ weak_source_bonus
- agreement_confidence_credit
```

触发规则：

```text
reject_score >= threshold -> deep_reroute
```

每个组成项都必须进入报告，便于判断回退原因。

### Expert1 deep_reroute 初始触发规则

初始规则先保守设置，后续通过验证集调参：

```text
预测 UnknownTarget：必进
Drone/Bird margin < 0.20：进
Unknown/Clutter margin < 0.25：进
Bird/Clutter margin < 0.20：进
traj_pred != rd_pred 且 rd_conf >= 0.45：进
raw_tr_rd 来源且 top1/top2 是高风险对：进
```

### deep_reroute 后的接受规则

`deep_reroute` 不能无条件覆盖第一轮结果，必须满足接受条件之一：

```text
deep_conf 高于 pass1_conf + delta
deep 解决了高风险混淆对
deep 把 Unknown 改成明确类且 residual score 支持
deep 仍为 Unknown，则最终 Unknown
```

如果 deep_reroute 低置信且没有解决混淆，应保留 Pass 1 或输出 Unknown，避免“越分析越错”。

### 回退样本报告字段

每个回退样本至少记录：

```text
reject_reason
reject_score
pass1_pred
deep_pred
final_pred
true_label
pass1_conf
deep_conf
traj_pred
rd_pred
confusion_pair
reroute_helped
reroute_harmed
```

## 七、工程约束

- 不写冗余代码和冗余函数。
- 不新增没有训练意义的标签字段。
- 不复制训练器逻辑。
- 不让细类体系破坏当前大类输出。
- 不把 RD 继续作为低空目标的弱辅助分支。
- 不在没有可靠标注时引入新细类。

## 八、当前最高优先级

```text
1. 讨论并定稿回退机制。
2. 最小化实现两层标签接口。
3. 将 Expert1 默认路径调整为 dual_main。
4. 将性能报告补齐 routed Expert1、true-low-altitude Expert1、RD 主干贡献、回退行为指标。
```
