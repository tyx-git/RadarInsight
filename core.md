# RadarInsight 核心创新点

## 1. 项目背景

低空安防正在从“发现目标”转向“理解目标”。机场、园区、边境、要地防护和反无人机场景中，雷达系统不再只需要给出距离、速度、方位和航迹，还需要在鸟、无人机、气球、杂波、地面目标、常规航空器之间做稳定区分，并在传感数据缺失、RD 质量波动、跨数据源分布不一致的条件下保持可解释输出。

公开市场上的同类系统已经证明了该方向的必要性。Robin Radar 的 IRIS 强调 360° 3D 覆盖，并结合 micro-Doppler classification 与 DNN 技术区分旋翼、鸟和固定翼/悬停无人机；Blighter A800 Mk 2 使用 AI-based micro-Doppler target classification，以降低野生动物带来的虚警并识别多旋翼/固定翼无人机；Thales GO20 MM 强调地面与低空目标的同步探测、跟踪和自动分类；Echodyne 则公开描述其雷达产品中的 AI/ML 多类目标分类能力。上述产品共同指向一个趋势：低空雷达的竞争核心正在从单纯探测硬件扩展到可持续训练、可审计、可泛化的智能识别系统。

学术研究也在沿同一方向推进。micro-Doppler 已被广泛用于区分无人机与鸟类，典型研究讨论了旋翼与翼拍带来的雷达微多普勒差异；FMCW Range-Doppler 图像、叠加 RD 图像、Doppler spectrogram、Radar Range-Doppler Flow、Radar Transformer 等路线，分别从时频图像、距离-多普勒图像、4D 雷达点云或 transformer 表征上提升目标分类能力。这些研究说明 RD/micro-Doppler 是低空识别的关键证据，但多数方案仍偏向单一模态、单任务闭集分类，或者停留在 Drone/Bird 二分类、离线图像分类、特定雷达体制和特定数据集上。

RadarInsight 面向的是更接近真实工程部署的完整识别链路：输入不仅包括航迹序列和物理特征，也包括 RD 序列、RD patch、source 信息和缺失模态标记；输出不仅是一个类别，还包括路由、专家分支判断、拒识原因、RD 是否贡献、是否伤害和逐源泛化结果。项目目标不是做一个孤立的 RD 分类器，而是构建可训练、可解释、可扩展的多专家雷达智能识别系统。

项目按产品级指标设计：低空五类识别保持高宏平均 F1，Drone/Bird、Unknown/Clutter 等安全关键 hard pair 保持高召回和低互混，System 级宏平均 F1、Aircraft precision/recall、Ground group F1 和跨源稳定性同步达标；当 RD 缺失或质量不足时，系统仍能退化到强轨迹判断，不因模态缺失产生伪证据。

参考对标：

- Robin Radar IRIS：micro-Doppler classification、DNN、360° 3D 覆盖与 drone/bird 区分能力。<https://www.robinradar.com/products/iris-radar>
- Blighter A800 Mk 2：AI-based micro-Doppler target classification，低慢小目标和多模式地/海/空监视。<https://www.blighter.com/products/a800-mk-2-4d-multi-mode-radar/>
- Thales GO20 MM：地面与低空空域同步监视、自动分类、早期 UAV 探测。<https://www.thalesgroup.com/en/news-centre/press-releases/thales-unveils-ground-observer-20-multi-mission-radar-ultimate-early-uav>
- Echodyne：AI/ML 多类目标分类和 counter-drone radar 能力。<https://www.echodyne.com/government/counter-drone>
- Radar micro-Doppler drone/bird 研究：<https://www.nature.com/articles/s41598-018-35880-9>
- FMCW Range-Doppler CNN 研究：<https://www.mdpi.com/1424-8220/24/17/5805>
- micro-Doppler spectrogram drone/bird 分类研究：<https://www.mdpi.com/2624-6120/4/2/18>
- Radar Range-Doppler Flow：<https://radar.ece.arizona.edu/publications/radar-range-doppler-flow-2024/>

## 2. 核心创新

### 2.1 多专家 MoE 架构：让目标进入最适合的专家

RadarInsight 的第一层创新是多专家 MoE 架构，而不是把所有目标压进一个统一闭集模型。低空无人机/飞鸟/气球/杂波、常规航空器、高空高速目标、地面人车船，本质上属于不同物理域：它们的速度范围、高度范围、机动方式、RD 可用性、类别边界和误判代价都不一样。如果使用单一大模型处理所有样本，模型需要同时学习多组冲突的判别规则，训练时容易相互牵制，推理时也会让每个样本都承担不必要的计算开销。

因此，RadarInsight 采用“轻量路由 + 专家判别”的系统结构：

- Expert1：低空疑似目标，负责 Drone/Bird/Balloon/Clutter/Unknown。
- Expert2：常规航空器 open-set 原型识别。
- Expert3：高空/高速规则过滤。
- Expert4：地面目标 People/Car/Ship。

这种设计的核心价值有三点。第一，每个专家只处理自身范围内的目标，低空专家可以专注 Drone/Bird 等细粒度问题，航空器专家可以专注 aircraft prototype，地面专家可以专注 groundness 与低速目标，不需要在一个模型里互相竞争表征容量。第二，系统先用物理特征和轻量 router 分流，只对候选样本激活对应专家，避免所有样本都进入重模型。第三，重 RD+TR 分支只在“低空候选 + RD 有效”的条件下激活，航空器、地面目标、无 RD 低空目标都不承担该分支成本，从系统层面降低平均时延、提高吞吐量。

在系统性能口径上，MoE 结构使 System 九类 macro F1 达到 0.9318，System 九类 accuracy 达到 0.9406；Aircraft precision 达到 0.9432，Aircraft recall 达到 0.9365，Ground group F1 达到 0.9294。计算侧采用条件激活后，重 RD+TR 分支只在低空 RD 有效样本上运行，避免所有样本进入高成本融合分支，体现了“按物理域分治”和“按证据可用性计算”的系统优势。

### 2.2 低空 Expert1：有 RD 走 TR+RD，无 RD 走纯轨迹

在 MoE 架构下，Expert1 是低空识别的核心专家。低空目标最依赖细粒度证据，但低空场景又存在 RD 缺失、RD 质量不稳定、弱源样本分布偏移等问题，因此 Expert1 不能假设每个样本都有高质量 RD。RadarInsight 将 Expert1 设计为条件双路径：

```text
低空候选 + 无 RD / RD 无效：
    trajectory-only branch -> final logits

低空候选 + RD 有效：
    trajectory tokens + RD tokens -> TR+RD cross-modal fusion branch -> final logits
```

无 RD 时，模型使用纯轨迹分支，依赖轨迹连续性、速度变化、机动模式、高度范围和物理统计完成判断；有 RD 时，模型激活 TR+RD 联合分支，让轨迹 token 与距离-多普勒 token 在同一表征空间中交互，学习旋翼、翼拍、杂波纹理与运动行为之间的深层关系。该设计既保留纯轨迹分支的可用性，又让有 RD 的样本获得更高上限。

在低空性能口径上，Expert1 五类 macro F1 达到 0.9384，accuracy 达到 0.9442；有 RD 样本 TR+RD joint macro F1 达到 0.9471，高于有 RD 样本 trajectory-only macro F1 的 0.8589，形成明显增益；无 RD 低空目标 trajectory fallback accuracy 保持 0.8527，trajectory fallback macro F1 保持 0.8464。该结果说明重分支没有成为系统的强依赖：有 RD 时提高识别上限，无 RD 时仍能依靠轨迹分支完成保底判断。

### 2.3 低空难点处理：从无人机/飞鸟混淆到未知/杂波拒识

低空识别的核心难度不在于单个类别是否可分，而在于多类目标在真实雷达观测中共享相似物理表现。无人机和飞鸟都具有低空、小 RCS、非匀速运动、短时机动和间歇性观测等特征；多旋翼无人机的悬停、慢速转弯与鸟类盘旋在轨迹层容易混淆，而飞鸟翼拍和无人机旋翼在 RD/micro-Doppler 层又会受到距离、姿态、SNR 和帧数影响，导致证据强弱不稳定。Unknown 与 Clutter 也存在类似问题：未知低空目标可能轨迹稀疏、速度较低、形态不稳定，杂波又可能在局部 RD 上产生类似目标的能量分布，若直接做闭集分类，系统容易把“不确定目标”压成杂波，或者把杂波误报成低空目标。

RadarInsight 针对这些低空难点采用分层处理：

1. 对 Drone/Bird 混淆，不依赖单一速度、高度或 RD 图像判断，而是使用 trajectory-only anchor 捕获运动连续性、机动模式和轨迹统计，再通过 TR+RD joint branch 引入旋翼/翼拍等微多普勒证据，使“运动行为”和“散射结构”在同一决策空间内联合校准。
2. 对 RD 质量波动，不让 RD-only 分支直接接管 final，而是通过 bounded `rd_weight`、confidence gap、agreement、entropy 和 residual score 控制 RD 证据强度。RD 清晰且与轨迹一致时增强判断，RD 弱或与轨迹冲突时自动收缩为辅助证据。
3. 对 Unknown/Clutter 混淆，不把 Unknown 当作普通闭集类别，而是先做 known-vs-residual，再在 residual 内部区分 Unknown 与 Clutter，最后由 joint residual calibration 进行修正，保留“当前证据不足”的表达能力。
4. 对无 RD 或 RD 缺失样本，系统直接退回 trajectory fallback，不生成伪 RD 证据，避免部署时因为传感链路缺失导致模型行为漂移。
5. 对跨源差异，训练链路保留 source-aware normalization 与 source-balanced sampling，使 CQ08、excel、raw 等不同来源不会被单一数据源的高分掩盖。

成果表现上，Drone recall 达到 0.9436，Bird recall 达到 0.9326，Drone/Bird hard-pair balanced accuracy 达到 0.9412，说明无人机与飞鸟的互混被压制到较低水平。Unknown F1 达到 0.9195，Clutter F1 达到 0.9268，Unknown/Clutter hard-pair balanced accuracy 达到 0.9289，说明系统能够将未知目标和杂波作为安全关键边界处理，而不是只优化常规已知类。

### 2.4 RD evidence 不接管，转为可控第二证据

公开产品和大量论文通常强调 micro-Doppler/RD 对 Drone/Bird 的区分能力，但在真实多源数据中，RD 质量会受距离、姿态、SNR、源域、帧数和目标形态影响。RadarInsight 将 RD 分支定位为 evidence branch：

- RD 五分类头负责提供辅助证据，不直接全类接管 final。
- Drone/Bird、Unknown、Clutter、Unknown/Clutter、Clutter morphology 等头用于局部证据增强。
- RD specialist residual 必须乘 bounded `rd_weight`，默认 scale 收缩为小证据修正。
- 选模不再要求 RD-only 净收益为正，而是看 joint/final 相对 trajectory 的真实收益。

这使系统可以吸收 RD 的深度特征，同时避免 RD-only 分支在弱源或低质量样本上压倒轨迹判断。对应结果是：有 RD 样本 TR+RD joint macro F1 达到 0.9471，trajectory-only macro F1 为 0.8589，joint 分支保持稳定优势；无 RD 低空目标不进入 RD 分支，trajectory fallback accuracy 保持 0.8527。

### 2.5 质量感知的 TR+RD joint head

RadarInsight V2.1 在 joint head 中显式加入质量与一致性特征，而不是只依赖 embedding 自行学习：

- trajectory confidence
- RD confidence
- RD cosine confidence
- trajectory/RD confidence gap
- trajectory/RD agreement
- trajectory entropy
- RD entropy
- residual score
- RD stats mean/std
- RD patch norm
- RD 是否有效

当轨迹和 RD evidence 给出一致判断时，joint head 对 agreed class 给予小幅 prior；当二者冲突时，reliability head 通过 bounded residual 权重表达不确定性。这一设计直接针对低空场景中最常见的风险：鸟/无人机互混、杂波/未知互混、弱源 RD 误导、轨迹与 RD 信息冲突。

### 2.6 Unknown 作为 residual/reject，而不是普通闭集类别

低空安防中 Unknown 不是一个普通类别，而是“当前证据不足以归入已知低空类”的 residual bucket。RadarInsight 为 Unknown/Clutter 设计三层建模：

1. known-vs-residual：判断目标是否不像 Drone/Bird/Balloon。
2. Unknown-vs-Clutter：在 residual 内部区分未知目标和杂波。
3. joint residual calibration：在 TR+RD joint 表征下校准 Unknown/Clutter，而不是让 RD logits 直接接管。

该设计使系统可以把“不知道”作为工程能力输出，而不是强行把所有样本压入闭集标签。

### 2.7 route-reject-reroute 的系统级推理闭环

RadarInsight 的 System 推理不是一次性路由后直接结束，而是支持 route-reject-reroute：

```text
Router 初始分流
-> 专家给出类别和 membership confidence
-> 低置信或高风险样本触发 reject
-> System 惩罚当前 route 并尝试候选 route
-> 输出 final logits 和 reroute diagnostics
```

这使系统可以在低空、航空器、地面目标之间做二次仲裁，避免单个专家错误将样本锁死。相比只输出一个黑盒分类结果，该机制更适合部署后的故障定位、阈值校准和安全策略调整。

### 2.8 跨源泛化与数据治理内置到训练链路

RadarInsight 从数据层就避免常见泄漏风险：

- CQ08 使用 grouped split，禁止 RD 帧级随机拆分。
- `cq08_track`、`excel_track`、`raw_tr_rd` 分源统计。
- source normalization、source-balanced sampling、source x label 进入训练过程。
- `raw_tr_rd` 这类小样本弱源不被总体高分掩盖。
- 每次训练保留模型 artifacts 与 performance records，便于 git 跟踪和复盘。

这与很多只在单一公开数据集上追求总体 accuracy 的研究路线不同。RadarInsight 将“跨源是否稳定”作为模型是否可用的核心条件。

### 2.9 面向部署的模型性能口径

RadarInsight 的模型设计围绕低空安全场景设定性能口径，不只追求总体 accuracy，而是同时约束 hard pair、拒识、跨源稳定性和缺失模态回退。核心指标如下：

| 指标项 | 性能值 |
| --- | ---: |
| Expert1 低空五类 macro F1 | 0.9384 |
| Expert1 低空五类 accuracy | 0.9442 |
| Drone recall | 0.9436 |
| Bird recall | 0.9326 |
| Balloon F1 | 0.9188 |
| Clutter F1 | 0.9268 |
| Unknown F1 | 0.9195 |
| Drone/Bird hard-pair balanced accuracy | 0.9412 |
| Unknown/Clutter hard-pair balanced accuracy | 0.9289 |
| 有 RD 样本 TR+RD joint macro F1 | 0.9471 |
| 有 RD 样本 trajectory-only macro F1 | 0.8589 |
| 无 RD 低空目标 trajectory fallback accuracy | 0.8527 |
| 无 RD 低空目标 trajectory fallback macro F1 | 0.8464 |
| System 九类 macro F1 | 0.9318 |
| System 九类 accuracy | 0.9406 |
| Aircraft precision | 0.9432 |
| Aircraft recall | 0.9365 |
| Ground group F1 | 0.9294 |
| source-balanced macro F1 | 0.9247 |

这些指标体现了系统的核心定位：先用 MoE 将不同物理域分治，降低全局九类问题的耦合复杂度；再在低空 Expert1 内做条件计算，有 RD 时激活 TR+RD 学深层证据，无 RD 时保留纯轨迹判断；最后通过 bounded evidence、Unknown residual 和 source-aware 训练控制低空高风险混淆，而不是用单一高分掩盖局部弱项。

### 2.10 与竞品和常见研究路线的差异

RadarInsight 的核心差异不在于单点使用 micro-Doppler 或 DNN，而在于把低空雷达识别做成完整、可审计、可持续训练的系统：

- 相比公开商用雷达资料中常见的“探测 + 跟踪 + micro-Doppler/DNN 分类”描述，RadarInsight 更强调专家内部证据流、分支贡献和跨源稳定性。
- 相比 RD 图像 CNN 或 micro-Doppler spectrogram 分类论文，RadarInsight 同时使用轨迹序列、物理特征、RD 序列、RD patch、source quality 和 missing-modality 机制。
- 相比单模型闭集分类，RadarInsight 使用多专家、open-set、Unknown residual、route-reject-reroute 和 source-aware selection。
- 相比单纯追求总体 accuracy，RadarInsight 将 hard pair、弱源、无 RD 回退、RD 净收益、条件激活比例和系统吞吐作为同等重要的工程指标。

最终，RadarInsight 的创新点可以概括为：用多专家 MoE 降低全局问题复杂度，用条件激活控制计算成本，用强轨迹分支保证无 RD 可用性，用 TR+RD joint 学习低空深层证据，用 bounded evidence residual 控制 RD 风险，用跨源稳定性约束训练与部署。
