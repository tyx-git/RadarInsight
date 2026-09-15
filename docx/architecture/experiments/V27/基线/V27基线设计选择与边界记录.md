# V27 baseline：设计选择、优势与边界决策记录

> 文档目的：记录截至当前讨论已经确定的 V27 baseline 设计选择，并明确每个选择的优势、证据边界和实现约束。  
> 范围：仅 `src-v3`，忽略 `src`。  
> 重要命名：`V27 baseline` 是完整三层系统的基线定义；`V27a` 才是第一个后续实验编号。当前 `src-v3/v27` 已具备 L1/L2 分类、独立 knownness/quality heads、动态 L3、日志和审计链路，但本轮 development safety floor 未通过，仍不能称为部署就绪。

## 0. 阅读规则

- “已确定”表示 V27 设计必须遵守。
- “条件启用”表示该机制可以进入 V27，但必须满足文中写明的因果、数据和安全条件。
- “候选/后续实验”表示不能未经单因素验证直接写入基线或产品动作。
- “优势”是设计层面的预期收益，不等于已经完成的性能或部署证明。

## 1. 系统身份与任务边界

### Q01：V27 baseline 到底代表什么？

问题：V27 是否只是 Liquid V2 的 L2 DBB 三分类实验？

选择：将 V27 baseline 定义为完整的 L1-L2-L3 系统，并在 baseline 本身建立 L1/L2 的独立 open-set 分支、quality/maturity 分支和 L3 合同。`V27a` 只能是该完整基线之后的受控消融编号，不能把当前无后缀 `V27` 的 L2 候选实现重新命名为 V27a。（已确定）

优势：实验编号、系统身份和因果归因不再混淆；可以分阶段训练和验证主干、独立 heads 与 L3，而不会把一个局部结果误报为完整系统结果。

边界：当前 smoke/overfit 通过只证明代码合同和小样本训练管线可运行，不证明 V27 baseline 已完成。

### Q02：V27 是否必须同时完善 L1、L2、L3？

问题：是否可以先只把 L2 做成最终 V27，L1 和 L3 继续沿用旧版本？

选择：V27 baseline 本身必须包含三个层次：

1. L1：KnownTarget / Clutter 路由与冲突检测，并具备独立的 open-set 能力。
2. L2：仅负责 Drone / Bird / Balloon 三分类，并具备独立的 open-set 能力。
3. L3：消费已校准的分类、knownness、质量和冲突证据，执行最终语义动作。（已确定）

L1 与 L2 使用同一套 Liquid V2 cell/wrapper 实现，但拥有独立参数、独立 recurrent state、独立任务特征和独立 checkpoint。

优势：保留层级任务的职责边界，避免 L1 的路由错误、L2 的子类错误和 L3 的拒识错误互相掩盖；同一主干接口又便于后续统一维护和做 matched comparison。

### Q03：Unknown 应该是什么？

问题：Unknown 是否应作为 L1/L2 的额外分类类别？

选择：Unknown 永远不是 L1 或 L2 的 classifier 类别，而是 L3 的拒识动作。L2 固定为 Drone/Bird/Balloon 三分类；Unknown 不参加 L1/L2 分类损失、runtime normalization、早停或分类 checkpoint selection。分类基座冻结后，Unknown 可以在隔离的 `representation_fit` 中作为独立 knownness 分支的负例，在 `representation_development` 中用于头部选择/排序诊断，并在独立 `calibration` pool 中用于动态 L3 校准；quality head 对 Unknown 只使用确定性质量/成熟度事实，不使用 Unknown 标签作为质量标签。（已确定）

优势：避免把异质开放集合强行压成一个普通类别；分类 posterior 不会被错误地解释为 knownness，Unknown 也不会污染类别边界、归一化统计和分类头。

证据：`src-v3/radar_ln3/open_set_architecture.py` 的当前契约，以及 V26k 对单点距离门控失败的诊断。

### Q04：L2 Liquid V2 调好后能否与旧 V1 L1 联合？

问题：V2 L2 是否必须等 V2 L1 完成后才能使用？

选择：技术上允许短期使用 `V1 L1 -> V2 L2 -> 重新校准的 L3` hybrid cascade，但这只是兼容性阶段，不是最终 V27。V1 和 V2 不共享 recurrent state、参数或 checkpoint；V2 主干改变后不得沿用旧 L3 阈值。最终 V27 应将 L1 也迁移到独立的 Liquid V2 实例。（已确定）

优势：可以先验证 L2 主干价值，降低一次性重建三层系统的风险，同时保留最终主干统一方向。

## 2. 历史基座与正向特征选择

### Q05：V27 是否应整体复制最新或最高分的历史实验？

问题：为什么不能直接以 V21e、V26 或 V18r 作为整个 V27 的基座？

选择：不按实验编号整体复制，而是按职责组合有证据的组件：

| V27 职责 | 采用的历史来源 | 具体处理 |
|---|---|---|
| L2 分类特征 | V18 stable8 | 使用 RD physical4 + broadband multiscale micro-Doppler stable8，作为 matched V1/V2 的共同特征锚点 |
| L1 正向特征 | V21e | 采用其因果圆形航向离散度，进入 L1；部署资格仍需重新验证 |
| 完整级联和 L3 协议参考 | V26c/V26e | 参考其层级、OOF/calibration 和审计合同，不无条件复制其模型或旧阈值 |
| Liquid 主干 | 自研 Liquid V2 | 采用 raw memory + pre-norm computation，不迁移官方 CfC 状态方程 |
| 负向边界 | V18r/V20r/V23g/V24c/V25/V26k | 记录失败原因，防止把失败机制默认打开 |

（已确定）

优势：同时保留 V21e 的有效信息、V18 stable8 的 L2 可比性和 V26 的系统审计经验；不会因为“最新”或“整体指标最高”而混入不同任务、不同数据协议或已失败的拒识机制。

### Q06：V21e 的圆形离散度是否进入 V27？

问题：已在 V21e 观察到正向收益的圆形特征是否应被丢弃，因为其 grouped OOF 尚未获得部署资格？

选择：采用 V21e 的 canonical feature：`causal_prefix_course_circular_incoherence`。公式为：

```text
1 - abs(mean(exp(i * radians(course_deg))))
```

它只使用当前及历史已观测的有效非首事件航向；首事件排除，至少需要三个有效非首事件，且必须保留 validity/support count。该特征进入 L1 分类输入，并可以作为 L1 knownness/quality 的显式旁路证据；不作为 L2 的 DBB 分类特征。（条件启用，基线应保留）

优势：圆形统计能表达航向轨迹的分散/一致程度，对 Clutter 与 DBB 的区分有物理可解释性；因果前缀形式还可以暴露“事件之间证据是否一致”，而非只依赖末端状态。

证据边界：V21e 多 seed 的 parent-final 方向为正，macro-F1 增益约为 `+0.09485`、`+0.13720`、`+0.16367`；但 capture/content-grouped OOF 的弱类 absolute guard 只有约 `0.6429`，低于 `0.9`，因此不能直接声称已具备部署资格。V21e 的正确结论是“保留并重新验证”，不是“无条件放行”。

### Q07：所谓“因果前缀不一致”是否是另一个新特征？

问题：用户所说的因果前缀不一致，是否应与 V21e 圆形离散度重复实现？

选择：先用代码和 schema 判断它是否就是 `causal_prefix_course_circular_incoherence` 的中文别名。若是同一统计量，只保留一个 canonical 字段，禁止重复拼接或重复计权；若是不同统计量，则必须新增独立、版本化、因果可审计的字段和单因素实验，不能把未经定义的特征悄悄并入 V27 baseline。（已确定的边界；独立新特征属于后续候选）

优势：避免特征重复、命名漂移和虚假的“两个正向特征”；同时保留在代码中真正缺失时补齐该信息的空间。

### Q08：哪些历史正向配置应在 V27 中默认启用？

问题：是否只保留特征，还是连同历史时间和损失机制一起全部打开？

选择：只默认启用有明确任务匹配且没有已知反例的部分：

- V18 stable8 + RD physical4：L2 分类默认特征。
- V21e causal course circular incoherence：L1 默认特征，带 validity/support 旁路。
- raw memory + pre-norm computation：V27 主干默认结构。
- parent-level fixed-parent-count 分类损失：默认训练语义。
- `bounded_observation` 时间策略：默认 active mode。
- 确定性的 RD/时间质量事实：允许作为输入或旁路事实，但不允许质量头输出回流。

其他历史机制必须按失败或证据等级处理，不能仅因曾经产生局部增益就默认开启。（已确定）

优势：把“正向证据”真正转化为基线能力，同时保留单因素可归因性；不会把局部指标改善误当成全系统晋级条件。

## 3. Liquid V2 主干与信息保留

### Q09：V27 应使用自研主干还是官方 CfC 主干？

问题：官方 CfC 是否应直接替换当前自研液态单元？

选择：保留自研 Liquid V2 作为唯一产品主干；只选择性迁移官方 CfC 的工程组织思想，不迁移其状态方程、mixed memory、wiring 或第二套产品后端。（已确定）

优势：保持当前项目已经验证过的 `tau/alpha/dt_reason` 语义和任务接口，同时获得清晰的 cell/wrapper/time-policy 分层、在线离线一致性和更强的 provenance 管理。

### Q10：Liquid V2 的原始记忆状态如何更新？

问题：为了稳定训练，是否继续在每个递归步骤后对状态做 LayerNorm？

选择：使用 raw memory + pre-norm computation：

\[
n_{t-1}=\operatorname{LN}_{compute}(r_{t-1})
\]
\[
(c_t,g_t)=F(x_t,n_{t-1})
\]
\[
m_t=g_t\odot c_t+(1-g_t)\odot r_{t-1}
\]
\[
r_t=(1-\alpha_t)\odot r_{t-1}+\alpha_t\odot m_t
\]

`r_t` 是跨事件持续保存的原始记忆，不执行逐事件后置 LayerNorm；`n_t` 只是计算临时量，不能回写为下一时刻记忆。（已确定）

优势：保留递归过程中原本会被 LayerNorm 压缩的全局尺度、更新幅度和长期轨迹信息；knownness 分支才有机会观察事件间尺度漂移、更新不匹配和证据稳定性。

### Q11：分类、knownness、质量头如何放置？

问题：是否让一个 LayerNorm 后的终态同时服务分类和密度/knownness 建模？

选择：从持续保存的 raw memory 与完整 raw trajectory 分叉不同任务：

- 分类头读取独立的 `LayerNorm(r_t)`，只用于分类读出。
- knownness 读取完整的 `r_1:T` raw causal trajectory、innovation、时间和质量信息。
- quality/maturity 使用独立输入、独立参数和独立监督。
- compute LayerNorm、classifier LayerNorm、knownness 内部的可选归一化互不混为同一模块。
- 任何读出归一化都不能写回 recurrent memory。（已确定）

优势：分类可以获得稳定的读出尺度，knownness 又不会被迫接受已经丢失全局尺度的单点终态；三种任务的语义和梯度边界清晰。

### Q12：轨迹中哪些量必须显式区分？

问题：是否可以把所有状态差分都叫作 innovation？

选择：至少分别记录：

- `candidate_innovation = c_t - r_{t-1}`
- `mixed_memory_innovation = m_t - r_{t-1}`
- `actual_update = r_t - r_{t-1}`

knownness 使用哪一种或哪几种必须写入 manifest；同时保存 `r_1:T`、gate、alpha、tau、dt_reason、zero ordinal、有效事件数和质量事实。（已确定）

优势：可以区分“候选内容变化”“门控后的变化”和“最终实际更新”，从而识别单个高置信事件拉动终态、事件间证据不一致和 dt/状态变化不匹配等开放集信号。

### Q13：L1/L2 是否共享 V2 状态？

问题：统一主干是否意味着 L1、L2 共用一套 hidden state 或参数？

选择：统一的是实现合同，不是状态和权重。L1、L2 各自实例化 LiquidV2Cell、wrapper、分类头、knownness 头和质量头；不把 V1 checkpoint 或 L1 recurrent state 直接作为 V2/L2 初态。（已确定）

优势：避免任务目标、样本路由和误差结构互相污染；又能确保 offline、chunked 和 online 使用同一种递归语义。

### Q14：官方 CfC 哪些内容可以迁移？

问题：参考官方实现时，哪些部分属于工程思想，哪些会改变研究变量？

选择：允许迁移：cell/wrapper/time-policy 职责分离、显式 initial state、同时返回 sequence/final state、padding 永不进入递归、offline/online 共用 cell、状态方程/时间策略/实现 hash 写入 checkpoint。禁止迁移：官方状态方程、mixed memory、wiring、第二套可切换产品主干。（已确定）

优势：得到可靠的状态传递和复现边界，同时不把官方 CfC 的动力学差异混入 V2 主干实验。

## 4. 时间、质量事实与归一化

### Q15：V27 active time policy 应该是什么？

问题：当前支持 `floor`、`bounded_observation`、`skip`、`strict_skip`、`density_weighted`，V27 是否默认使用 density-weighted？

选择：L1/L2 V27 baseline 的 active mode 使用 `bounded_observation`；继续保留 `tau`、`alpha`、`dt_reason`、`zero_ordinal` 和 strict-skip 的实现能力，但 `density_weighted`、`strict_skip`、`skip` 和 `floor` 只能作为后续单因素变量，不能在 baseline 中暗中并行生效。（已确定）

优势：对 zero/invalid dt 进行有限的观测更新，避免递归被异常时间值破坏；同时保持时间语义可替换、可审计。

历史边界：V24c 的 strict same-second skip 使 parent-final 和 eligible-prefix 指标退化；V18r/V20r/V23g 的 density-weighted 方向未通过完整采用门槛。当前 V27 配置和 manifest 的 active mode 已是 `bounded_observation`；`density_weighted` 仅作为 supported future mode 保留。

### Q16：确定性的 `rd_quality`/dt 质量事实是否允许进入 L1/L2 分类主干？

问题：确定性的 RD 可用性、dt validity/reason、有效事件数、SNR/距离等事实，是否只能给 quality head，不能给分类器？

选择：允许确定性事实作为 L1/L2 的显式输入或旁路特征，但禁止 learned quality/maturity 头的输出回流到 recurrent state、分类输入或 knownness 表示；质量事实不得被用来直接把样本标成 Unknown。（Recommended，已确定）

优势：增强缺失和观测质量变化下的鲁棒性，同时避免“质量头自我强化”的循环依赖和标签泄漏。L3 仍先依据确定性 eligibility，再依据 learned quality 和 calibrated knownness 做动作。

### Q17：V27 runtime normalization 应使用哪一批数据？

问题：能否继续复用包含五类标签的历史 V16 normalization statistics？

选择：V27 runtime normalization 只从 CQ08 DBB train parents（Drone/Bird/Balloon）拟合；Clutter、Unknown、validation、test、calibration、audit 都排除。历史 V16 五类 normalization 只能作为 V18 feature-generation provenance，不能作为 V27 runtime model input。L1/L2/knownness/quality 若人口不同，必须各自拥有命名清晰的 normalization contract。（已确定）

优势：维持 Unknown training exposure 为零，避免 Unknown/Clutter 通过统计量间接进入模型；每个统计量都能通过 parent、split 和 hash 复核。

## 5. Open-set 表征、评分与校准

### Q18：L2 分类概率和 knownness 是否应该完全拆分？

问题：是否可以把 max-softmax、classification margin 或最终分类 LayerNorm 终态直接当作 knownness？

选择：分类概率只回答“在三个 DBB 类中更像哪一类”；knownness 使用独立分支和独立目标，不能把 Unknown 作为第四类，也不能让分类置信度直接等于 knownness。分类 margin/entropy/energy 只能作为显式登记的旁路候选。（已确定）

优势：避免闭集相对分类置信度冒充开放集支持度，保留对“高置信错误已知”和“边缘已知”的独立建模能力。

### Q19：knownness 应该读取什么信息？

问题：是否使用一个 parent-final 单点状态就足够？

选择：knownness 读取完整因果轨迹 `r_1:T`，并结合三类 innovation、状态更新一致性、事件密度、`dt/dt_reason/zero_ordinal`、RD/微多普勒 validity、SNR/距离、有效帧数和显式质量事实。它必须能表达：事件间证据不一致、RD 反射模式不稳定、dt 与状态变化不匹配、只有一个事件像 DBB、末端被单个高置信事件拉入已知类。（已确定）

优势：表征与 Unknown 的真实生成差异匹配，而不是把所有信息压缩成一个已经失去历史尺度的终态点。

### Q20：32 维 projection、多原型和 Mahalanobis 是否直接锁定？

问题：是否应预先断言“32 维、多原型、Mahalanobis”就是正确答案？

选择：32 维 knownness projection、多原型余弦、train-parent kNN nonconformity、低维 Mahalanobis、energy 以及 prototype-distance + classification-margin 都保留为候选消融变量；先选择表征维度，再比较评分器，不能同时改变 projection、原型数、协方差和主干。Mahalanobis 只能作用于学习后的 knownness space，不能作用于递归 LayerNorm 后的 legacy parent-final state。（已确定）

优势：把“维度”“几何空间”和“评分公式”拆开验证，避免把某个漂亮的距离公式误认为开放集能力来源；同时保留 Mahalanobis 在合适空间内发挥作用的机会。

### Q21：是否取消所有硬门控？

问题：V26k 单一距离阈值失败后，是否应完全取消离散动作边界？

选择：取消 V26k 式单一距离阈值直接 gate，但不取消最终硬动作。硬边界集中在独立 calibration 后的 L3，由 calibrated knownness、quality、conformal prediction set、分类冲突和 eligibility 共同决定；固定 softmax 阈值不是最终方案。（已确定）

优势：先解决任务匹配的表征和排序，再做校准与动作；不会用一个对训练内距离拟合的阈值把边缘已知、错误已知和异质 Unknown 混成同一动作。

### Q22：固定 softmax 阈值和动态拟合如何取舍？

问题：为了效率，是否用固定 max-softmax 阈值完成 L1/L2 open-set？

选择：V27 的正式方案使用独立 open-set calibration pool 动态拟合分数方向、温度/校准参数、conformal quantile、类别/质量条件和最终动作边界。softmax 只能作为描述性 comparator，或后期单独的效率消融实验，不能作为当前正式 open-set 依据。（已确定）

优势：阈值适应当前任务、特征、主干和质量分层；模型变化后可以重新校准，不会把固定经验阈值误认为跨实验稳定的 knownness 语义。

## 6. Quality、Pending 与 L3 动作

### Q23：quality/maturity 是否独立建立？

问题：是否可以用低 softmax、低 knownness 或分类错误直接当作低质量标签？

选择：建立独立 quality/maturity head，并与 knownness、classification 分开训练。质量监督来自可审计的有效事件数、RD/微多普勒可用性、SNR/距离、dt 语义、缺失模式、成熟度或获准的 prefix 稳定性定义；Unknown 标签、分类错误和低 posterior 不能循环定义 quality。（已确定）

优势：能区分“还没有足够证据”与“证据足够但不像已知分布”，防止通过扩大 Pending 人为改善 Unknown 指标。

### Q24：L3 的决策顺序和 reason code 是什么？

问题：低质量、类内歧义、错误已知和真正 Unknown 是否可以都输出同一个拒识结果？

选择：固定以下顺序和细粒度原因：

1. 有效事件不足、RD/微多普勒质量不足或 dt 语义不可用：`PENDING_ABSTAIN / LOW_QUALITY_OR_IMMATURE`。
2. 质量足够但 knownness 不足，或 conformal prediction set 为空：`UNKNOWN_REJECTED / OOD_NONCONFORMING`。
3. knownness 足够且只有一个可靠 DBB 类：输出 `DroneTarget`、`BirdTarget` 或 `BalloonTarget`。
4. DBB 子类证据冲突：记录 `KNOWN_CLASS_AMBIGUOUS`；流式场景可 Pending，终态质量足够时可 Unknown，不能回退为 Clutter。
5. L1 接受 Clutter：输出 `ClutterTarget`。
6. L1 target/clutter 冲突：按现有契约进入 Pending，不能回退为 Clutter。（已确定）

优势：产品动作与原因可审计，低质量样本不会被计为 Unknown 拒识成功；可以分别统计 OOD、类内冲突、错误已知和观测不足。

## 7. 训练、损失与实验顺序

### Q25：四/五个任务是否一次性端到端联合训练？

问题：总损失是否直接写成一个公式后一次性优化？

选择：总损失可以表示为：

\[
L=L_{\mathrm{cls-final}}+\lambda_pL_{\mathrm{cls-prefix}}+\lambda_kL_{\mathrm{knownness}}+\lambda_qL_{\mathrm{quality}}+\lambda_mL_{\mathrm{metric}}
\]

但不允许一开始强行端到端联合。推荐顺序是：

1. 先用 matched V1/V2 训练和验证 L2 分类主干。
2. 冻结已知分类主干和分类头，单独训练 knownness projection、原型/评分器和 quality/maturity head。
3. 先检查 KnownTarget Recall、eligible-prefix 和已知分类安全是否保持或提升。
4. 只有在必要时才解冻最后少量明确列名的层，以更低学习率进行联合微调。
5. 重新测量 classification/knownness/quality/metric 的梯度 cosine、norm 和冲突频率；只有新任务实测存在有害冲突时，才使用交替优化或非对称 PCGrad。（已确定）

优势：先保护已知分类和表征排序，再逐步增加任务；可区分真实梯度冲突、损失尺度失衡和数据噪声，降低 V25 类似冲突再次破坏主任务的风险。

边界：V25 只证明旧 L1 Known/Clutter 共享梯度存在严重冲突，不能直接证明新 classification/knownness/quality/metric 四任务必然冲突。

### Q26：已知分类损失采用哪种归约？

问题：是否按当前 batch 中出现的类别重新归一化 class weight？

选择：继续使用 parent-level class-balanced 的 `fixed_parent_count` 归约和 train-only parent class weights；禁止隐式 `batch_weight_sum`，禁止让 batch 类别组成改变目标函数尺度。prefix loss 若启用，也必须有独立的 parent-level 归约和预注册权重。（已确定）

优势：稀有类不会因为某个 batch 缺少其他类别而被重新缩小，训练目标在不同采样批次之间保持一致，checkpoint 比较更可解释。

历史边界：V22 证明固定分母实现了正确的目标语义，但该历史单因素本身没有带来足够性能收益；因此采用它是为了目标正确和可复现，不应宣称它已经被性能晋级。

### Q27：V27 是否立即做 OOF 跨源验证？

问题：是否在 baseline 阶段同时进行 OOF、跨源和完整部署审计？

选择：暂不进行 OOF 跨源验证，先完成 baseline 的 matched 训练和实现验证。仍须严格区分 train、validation、representation-development、calibration 和 final audit；test 与最终 audit 在前置阶段保持封存。（已确定）

优势：先隔离主干和任务设计变量，降低计算成本与归因复杂度；不会把尚未完成的 baseline 与跨源泛化结论混在一起。

边界：不做 OOF 跨源不等于允许数据泄漏；parent/capture/content 边界、hash 和角色隔离仍是硬约束。

### Q28：独立 open-set calibration pool 如何使用？

问题：Unknown 是否可以直接加入训练，或用 validation/test 反向选择阈值？

选择：建立独立的 open-set calibration pool，用于 dynamic score calibration、prototype/covariance/kNN support 的获准选择和 L3 阈值拟合；它不参与分类主干训练、normalization、early stopping 或 final audit。Unknown 不进入分类训练；若使用 Unknown 监督 knownness，必须另设隔离的 development exposure pool，并与 calibration/audit 不重叠。（已确定）

优势：同时满足严格 open-set 的无训练污染要求和动态拟合需要；阈值、协方差、原型、support、temperature、conformal quantile 都可以追溯到明确的数据角色。

## 8. 工程合同与复现要求

### Q29：如何保证 offline、chunked、online 语义一致？

问题：是否只验证离线完整序列，在线推理另写一套逻辑？

选择：offline、chunked 和 online 必须调用同一个 Liquid V2 cell。continuation 除 raw state 外，还要携带 previous timestamp、zero-delta ordinal、有效事件数和 maturity 元数据；padding 永不进入递归，strict-skip 事件必须真正绕过 cell。（已确定）

优势：避免训练/部署状态方程分叉，能正确表达事件密度和时间语义，降低线上 prefix 与离线 final 不一致的风险。

### Q30：配置、路径和 registry 如何绑定？

问题：是否可以让不同脚本自行解析布尔值、拼接路径和读取 registry？

选择：统一入口和配置合同：

- 所有布尔值使用同一个严格 parser，拒绝把任意非空字符串当作 `true`。
- 所有配置路径先相对 project root 解析，并 fail-closed 验证仍在允许根目录内。
- registry、split、feature、normalization、equation、time-policy、loss 和 action schema 都写入 canonical hash。
- checkpoint 绑定 task/layer、experiment/arm、model contract、实现 hash、状态方程 hash、time-policy hash、feature/normalization/split hash、loss reduction、class weights、seed 和 continuation schema。
- V1→V2、L1→L2、control→treatment 默认严格拒绝；禁止 `strict=False`、partial load 或同名 tensor 静默迁移。
- 以一个统一 CLI 驱动 prepare、smoke、overfit、train、evaluate、calibrate、audit，清理职责重复的历史脚本。（已确定）

优势：让错误配置、路径越界、registry 漂移和错误 checkpoint 在入口处失败；实验产物可以被独立复核，不依赖文件名猜测。

## 9. 历史机制的启用、保留和排除表

| 机制/配置 | V27 处理 | 原因与优势 |
|---|---|---|
| V18 RD physical4 + stable8 | 默认启用到 L2 | 最稳定的 L2 特征锚点，保持 V1/V2 matched 可比 |
| V21e causal course circular incoherence | 默认进入 L1；knownness/quality 作为旁路候选 | 已有多 seed 正向方向，能表达航向轨迹离散；部署前仍需重新验证 |
| `bounded_observation` | L1/L2 active | 保留有效/异常 dt 的有限观测更新，避免 strict skip 的退化 |
| `density_weighted` | 保留实现，baseline 禁止默认启用 | V18r/V20r/V23g 未通过完整采用门槛，避免将局部增益误当整体收益 |
| `strict_skip` | 保留实现，baseline 禁止默认启用 | V24c 对 parent/prefix 有负向结果 |
| fixed-parent-count loss | 默认启用 | 目标归约正确、与 batch composition 解耦；性能收益仍需独立验证 |
| V25 gradient projection | 不默认启用 | 旧任务的投影实验未通过安全门；新任务需重新测量冲突后决定 |
| V26g availability-only hard quality gate | 不迁移为质量头 | 早期 prefix 行为变差；只复用确定性 availability facts |
| V26k final-state Mahalanobis hard gate | 排除 | 已知召回约降至 0.91，仍接受约 86%–89% 的 target-route Unknown |
| V24a RD peak drift | 排除 | 只有局部近阈值修正，未证明最终收益 |
| V21h ISO / V21i d-range | 排除默认基线 | 历史上未形成可采纳正向证据 |
| 官方 CfC 状态方程/mixed memory | 禁止迁移 | 会改变 V27 唯一主干变量并引入第二套产品动力学 |
| 固定 softmax 阈值 | 不作为正式 L3 | 不能适应主干/质量/来源变化；只可作为后期效率 comparator |

## 10. 当前 `src-v3/v27` 的实现状态

以下是当前 V27 baseline 已核验的实现合同，不是新的实验选择：

1. L1/L2 已是独立 Liquid V2 task model，拥有独立参数、状态、分类头、knownness head 和 quality head。
2. 配置已声明 L1/L2 open-set capability、独立 heads 和 L3；分类基座与 heads 已训练，L3 registry 已动态拟合，同时保留未拟合 registry 的 fail-closed 路径。
3. L1/L2 active mode 已统一为 `bounded_observation`；`density_weighted` 只作为 future ablation capability 保留。
4. L1 数据路径已携带 V21e circular feature 的 value、validity 与 causal support；L2 使用固定 RD physical4 + stable8。
5. V27 专用 L3 registry/resolver 已绑定独立 knownness/quality/calibration 语义和细粒度 reason code。
6. L2 使用 DBB train-only runtime normalization；L1 使用 known-only train normalization；Unknown 未进入分类训练、runtime normalization 或分类 checkpoint selection。
7. 系统 manifest、task checkpoint、representation/head score cache、calibration registry 和 replay/audit report 均有 hash 绑定。
8. contract smoke 和 open-set audit 已通过；OOF、跨源、test 和部署结论仍未执行或未获资格。development replay 的 KnownTarget route recall 为 `0.9401`，低于 `0.95` 安全线，故当前 `deployment_eligible=false`。

## 11. V27 baseline 的最小完成条件

- [x] L1、L2 都有独立 Liquid V2 实例、open-set 分支、quality/maturity 分支和独立 recurrent state。
- [x] L2 保持严格三分类；Unknown 不进入 classifier，作为 L3 拒识动作，并可在冻结基座后的隔离 knownness/calibration 角色中使用。
- [x] V21e circular feature 的 causal/value/validity/support contract 已进入 L1 数据路径。
- [x] Liquid V2 raw memory、pre-norm computation、独立 classifier LayerNorm、raw trajectory trace 已通过合同测试。
- [x] L1/L2 active time mode 明确为 `bounded_observation`；其他 mode 不得隐式生效。
- [x] DBB train-only runtime normalization、feature provenance normalization 和各任务 normalization 角色已分离并 hash 绑定。
- [x] knownness、quality、classification 的训练和参数边界已分开；calibration pool 上的 KnownTarget safety check 先于 L3 gate。
- [x] 独立 calibration pool、动态校准流程、L3 动作顺序和 reason code 已实现；固定 softmax/单距离阈值未成为正式 gate。
- [x] offline/chunked/online、padding、strict-skip、initial state、path boundary、registry hash 和 checkpoint fail-closed 测试通过。
- [ ] development replay 的 KnownTarget route recall 达到 `0.95`，并满足部署安全条件；当前为 `0.9401`，所以 `deployment_eligible=false`。

## 12. 总结性结论

V27 的核心选择不是“找一个历史上分数最高的整体模型”，而是：

```text
V18 stable8 L2 特征
        + V21e causal circular feature 的 L1 正向信息
        + 自研 Liquid V2 raw memory / pre-norm computation
        + 独立 classification / knownness / quality 任务头
        + 完整 raw causal trajectory
        + 独立动态 calibration 的 L3
        = 可审计的三层 V27 baseline
```

Open-set 能力的优先级固定为：任务匹配的表征 → knownness 排序 → 独立校准 → L3 硬动作边界。LayerNorm、32 维、原型数、Mahalanobis、energy、margin 和梯度冲突优化器都必须接受消融和证伪，不能在没有对应证据时被当作答案。
