# RadarInsight Target Roadmap

## 1. Mission

The project now formally shifts from a clip-centric optimization loop to a session-centric recognition system.
The highest-priority objective is no longer to squeeze small gains from clip-level decision rules, but to build a stable session-level training and inference stack that can scale to the future paired dataset expansion.

The long-term target remains high-accuracy radar target recognition under real deployment constraints:

- high classification accuracy
- stable generalization across hard sessions
- support for real-time or near-real-time inference
- smooth scaling from the current paired dataset to the future approximately 4000 paired sessions

## 2. Current Ground Truth

The project has already established several important facts.

- The current strongest validated reference before the session-level transition is `artifacts/202604141227-12-4-cuda`.
- The current bottlenecks are not uniformly distributed across all classes.
- The dominant hard confusion pairs are:
  - `ShipTarget -> DroneTarget`
  - `BirdTarget -> ShipTarget`
- Errors naturally cluster by session and by long suspicious spans, not just by isolated clips.
- Many aggressive clip-level fixes were tested and rejected as unstable defaults, including broad hard sampling, heuristic conditional gating, generic confusion handlers, and several pairwise correction layers.
- A true session-level training backbone is a more structurally correct direction than continuing to overfit clip-level patches.

## 3. Strategic Shift

From this point forward, the project should be optimized around the following first-principles assumption:

**The radar target label is fundamentally a session property, while clips are only partial evidence.**

This means:

- the training unit should be a session bag, not a single clip
- the main prediction target should be a session label
- clip-level outputs should become supporting evidence rather than the final decision authority
- hard confusion handling should operate on session evidence and pair-specific evidence, not broad global biasing

## 4. Mainline Architecture Direction

### 4.1 Core Model Direction

The mainline architecture should continue evolving toward this shape:

1. clip encoder stage
   - track encoder
   - RD encoder
   - clip-level fusion encoder
   - clip embedding output

2. session bag stage
   - one session is represented as a bag of clip embeddings
   - masked attention pooling or set-based aggregation produces a session embedding
   - session classifier predicts the final class

3. session specialist stage
   - session-level pairwise experts for the hardest confusion pairs
   - session-level router or reliability features only as supporting evidence

4. deployment stage
   - support full-session inference
   - support prefix-session inference for real-time use
   - support confidence-driven early decision

### 4.2 What Should No Longer Be the Mainline

The following directions should not remain the default optimization route:

- broad hard-example oversampling
- global logit bias as a permanent fix
- heuristic feature-threshold conditional correction
- clip-only confusion handlers with strong residual intervention
- training strategies that improve a single confusion pair while destabilizing the overall manifold

These branches may remain as experimental tools, but not as the default route.

## 5. Immediate Engineering Priorities

### Priority 1: Stabilize Session-Level Training

The session-level mainline has been wired in, but it is not yet stable enough. The next engineering work should focus on stability before chasing additional peak accuracy.

Required actions:

- reduce training variance caused by random bag sampling
- add deterministic or semi-deterministic bag construction for early experiments
- add gradient accumulation to simulate a larger effective batch without exceeding 8 GB GPU memory
- standardize bag length handling and masked attention normalization
- expose session attention weights for debugging and error replay
- ensure evaluation, calibration, and diagnosis fully support session-bag checkpoints

Success criteria:

- repeated training runs should not fluctuate wildly across large ranges
- session-level metrics should be reproducible within a reasonably narrow band
- session-level checkpoints must be evaluable through the same tooling as clip-level checkpoints

### Priority 2: Upgrade the Session Aggregator

A future target aggregator branch should explicitly include recurrent sequence models as first-class options, not only set/attention pooling.

- target future variants:
  - `BiGRU + attention pooling`
  - `BiLSTM + attention pooling`
  - hybrid `recurrent encoder + lightweight transformer refinement`

Rationale:

- ordered clip sequences carry real temporal information
- small-to-medium radar session datasets may favor the stronger inductive bias of recurrent encoders
- recurrent session encoders may be more stable than pure attention encoders before the large-scale paired dataset arrives
- this branch should be treated as an explicit future mainline candidate, not a one-off experiment

The current attention-pooling bag model is only the first stable version. The next upgrade path should be:

1. masked attention pooling baseline
2. multi-head session attention pooling
3. lightweight set encoder or Set Transformer block
4. optional temporal bag encoder for ordered clip sequences

The guiding principle is to let the model learn which clips are trustworthy without forcing every clip to carry the session label equally.

### Priority 3: Make Session-Level Specialists Real

The main classifier should not be the only session-level head. The auxiliary heads should also be lifted from clip-only supervision to session-level supervision.

Needed session-level specialists:

- `ShipTarget vs DroneTarget` session expert
- `BirdTarget vs ShipTarget` session expert
- session-level surface or airborne routing cue if it remains useful

The correct pattern is:

- keep clip-level specialists as evidence generators
- aggregate specialist evidence at the session level
- train session-level specialist heads on top of pooled features

### Priority 4: Prepare for Streaming and Real-Time Inference

Session-level training does not need to imply full-session latency at deployment time.

The intended real-time strategy should be:

- train on full or partial session bags
- infer on prefix bags in streaming mode
- maintain an updated session embedding as new clips arrive
- emit early predictions when confidence is high enough

Required deployment features:

- prefix-session inference
- sliding-window bag inference
- confidence thresholding
- optional early exit

This keeps session-level semantics while staying compatible with real-time constraints.

## 6. Data Strategy for the Future 4000 Paired Sessions

The planned paired dataset expansion is likely the single biggest opportunity for real performance growth.

If the project reaches approximately 4000 paired sessions with matched track and RD data, the following become much more viable:

- stronger session-level encoders
- pair-specific metric learning
- contrastive session embedding objectives
- more reliable specialist heads
- domain generalization and shift-robust training
- more aggressive attention-based bag models

The future data pipeline should be designed now for that scale.

Required data-pipeline principles:

- lazy loading rather than eager loading
- session index manifests stored separately from heavy tensors
- RD data read on demand
- track data read on demand
- caching only metadata and compact manifests in memory
- support for partial sharding and future distributed training if needed

The future dataset format must preserve:

- stable `session_id`
- stable mapping between track data and RD data
- exact label alignment
- timestamp or ordering needed for prefix or streaming inference

## 6.0 Data Processing Boundary

The repository must maintain a hard boundary between data processing and model training.

- `pro/` is the only project area intended for raw-source parsing, source normalization, QC, slicing, and dataset preparation.
- `data/` is the training-consumption area and should contain only standardized session data copied from processed outputs.
- Raw-source-specific parsing code should live under `pro/`, not under the core training package.
- Training code should assume `data/` already contains standardized sessions and should not depend on source-specific preprocessing details.

## 6.1 Immediate New-Source Ingestion Plan

The newly introduced `CQ-08` source establishes a second-generation ingestion requirement.

Current confirmed facts about `CQ-08`:

- it currently contributes `1550` paired data groups, not `4260`
- every group contains:
  - raw radar echo data (`.dat`)
  - point-track text data
  - filtered track text data
- the three folders are one-to-one aligned by track identifier and label
- raw echo data is pulse-compressed complex data, not just a derived image product

This source should not be force-fitted into the legacy four-class route.
Instead, it should become the first standardized raw-source ingestion branch for the future larger session-level system.

## 6.2 Unified Label Mapping Direction

The label system must now be treated as a scalable unified taxonomy rather than a fixed four-class taxonomy.

For the `CQ-08` source, the currently confirmed mapping is:

- `1` -> `DroneTarget`
- `2` -> `DroneTarget`
- `3` -> `BirdTarget`
- `4` -> `BalloonTarget`
- `5` -> `ClutterTarget`
- `6` -> `OtherTarget`

This mapping should be encoded explicitly in the ingestion pipeline and in dataset manifests.

## 6.3 Optional-Modality Principle

The legacy dataset and the new `CQ-08` source do not expose identical modalities.

- legacy sessions provide track and RD, but no point-track modality
- `CQ-08` provides track, raw echo, and point-track data

Therefore the correct unification rule is not to force all sources to have the same modalities.
Instead, the system should use a unified session schema with optional modalities.

Every standardized session should support:

- `track_sequence`
- `rd_tensor`
- `point_sequence` or `point_features`
- `has_track`
- `has_rd`
- `has_points`

This means point-track data should be treated as an optional modality, not a mandatory requirement.

## 6.4 Role of Point-Track Data

Point-track data is useful and should be retained, but not as the primary mandatory backbone for all sources.

Its near-term role should be:

- auxiliary evidence branch
- session quality signal
- RD crop and alignment support
- reliability or ambiguity cue

Its long-term role may evolve into:

- point-set or point-sequence encoder
- session-level multi-modal branch

The first implementation should stay conservative:

- use point-track data as optional aggregated features
- use `has_points` masks for missing-modality handling
- do not make point-track data mandatory for training legacy sessions

## 6.5 Standardized Storage Policy

The project should preserve three logical data layers:

1. raw source layer
   - keep original `.dat` files as the immutable source of truth

2. parser or processing layer
   - parse raw echo, point-track, and filtered track files into normalized intermediate structures

3. training layer
   - store standardized session-aligned training data in a session container format such as `h5`

The training layer should not depend on thousands of loose intermediate arrays.
The recommended persistent training format remains session-organized container storage.

## 6.6 Immediate Engineering Work For `CQ-08`

- a staged denoise and alignment layer should be planned for real-world raw echo sources
  - denoise should remain configurable and source-aware
  - alignment should explicitly handle track-to-RD synchronization and target-centered crop refinement
  - these should be treated as preprocessing-layer capabilities under `pro/`, not as ad-hoc fixes inside the training loop

The next ingestion milestone for the new source should include:

- a Python rewrite of the MATLAB raw-echo parser
- a deterministic loader for GBK-encoded point-track and track text files
- a source-specific manifest builder
- a standardized session export path from:
  - raw echo
  - point-track data
  - filtered track data
  - mapped label name

The end result should be a source-normalized session manifest and session container output that can be consumed by the session-level training mainline.

## 6.7 Kaggle DopplerNet-Derived Session Source

A second auxiliary source is available under `pro/Kaggle/`, accompanied by the paper `DopplerNet: a convolutional neural network for recognising targets in real scenarios using a persistent range-Doppler radar`.

Current confirmed properties:

- `Cars/` and `People/` contain session-like folders
- each folder contains ordered CSV files
- each CSV is a fixed-size numeric array with shape approximately `11 x 61`
- this source appears to be RD-only and does not provide track data

Integration decision:

- this source is usable
- it should be treated as an RD-only auxiliary source
- its labels can be kept as:
  - `CarTarget`
  - `PeopleTarget`
- it should not be forced into a track+RD paired-source assumption

Engineering rule:

- the unified session schema must support optional track modality
- Kaggle sessions should be ingested as:
  - `has_rd = true`
  - `has_track = false`
  - `has_points = false`

Training strategy:

- this source should be used only after the session-level pipeline fully supports optional-modality training
- it is most appropriate for multi-source session training, auxiliary pretraining, or RD-specialist training rather than for the old paired-only route

## 7. Generalization Hypothesis

The project should proceed under the following working hypothesis:

**Session-level training is more likely than clip-level training to generalize under real target variability, provided enough paired sessions are available.**

Reasoning:

- the label belongs to the session, not to every individual clip equally
- hardest errors already cluster by session and by continuous spans
- clip-level training overweights local noise and accidental short-term patterns
- session-level training can learn persistent motion and RD evidence across time

This hypothesis still needs large-scale validation once the expanded paired dataset arrives.

## 8. Short-Term Experimental Protocol

Until the large dataset arrives, all session-level work should follow a strict protocol.

### Stage A: Stability First

- fixed seed
- fixed bag construction policy
- conservative bag size
- conservative bag batch size
- gradient accumulation if needed

### Stage B: Baseline Session Mainline

- main session classification loss
- light clip auxiliary loss only if it improves stability
- no aggressive confusion handler by default

### Stage C: Session Pairwise Specialists

- add session-level `Ship/Drone` expert first
- only after that is stable, add session-level `Bird/Ship` expert

### Stage D: Real-Time Session Inference

- prefix-bag evaluation
- confidence curve analysis
- early-exit experiments

## 9. Explicit Near-Term Roadmap

### Phase 1: Make Session Training Stable

- finish debugging the current session bag pipeline
- make evaluation fully compatible with session checkpoints
- reduce run-to-run variance
- document stable training settings for 8 GB GPU

### Phase 2: Make Session Training Useful

- add session-level specialist heads
- lift the most useful auxiliary heads from clip supervision to session supervision
- add attention inspection and session replay tools
- compare full-session and prefix-session behavior

### Phase 3: Prepare for Scale

- redesign data loading for thousands of sessions
- make manifests first-class training inputs
- support large-scale lazy session loading

### Phase 4: Push for Accuracy

- only after stability and scale are ready
- test stronger session encoders
- test metric learning on session embeddings
- test pair-specific reranking or specialist calibration on session outputs

## 10. Current Non-Negotiable Rules

- `debug.md` must be updated for every major structural change and every failed experiment with diagnostic value.
- `arch.md` must be updated whenever the architecture materially changes.
- New defaults should only be promoted after repeatable improvement, not after a single lucky run.
- The session-level route is now the official long-term direction.
- Clip-level-only optimization is no longer the strategic mainline.

## 11. Final Working Decision

The project now commits to the following long-term engineering decision:

**Build a scalable session-level radar recognition system first, then use the future large paired dataset to unlock the next accuracy tier.**

In practice this means:

- optimize session-level learning stability now
- keep the current strongest checkpoint as the comparison anchor
- treat future data expansion as the enabler for the next major leap in accuracy
- stop relying on fragile clip-level fixes as the primary growth mechanism

## 12. 分阶段10类训练方案

### 12.1 目标

当前10类主线的首要目标不是一次性冲到最高分，而是先定位：

- 到底是哪一个数据来源在拖低整体表现
- 到底是哪一组来源组合会引发明显退化
- 到底是数据源本身难学，还是多源混合后才出现冲突

当前三类来源为：

- `LegacySession`
  - `AircraftTarget`
  - `BirdTarget`
  - `DroneTarget`
  - `ShipTarget`
- `CQ-08-MAT`
  - `BirdTarget`
  - `DroneTarget`
  - `BalloonTarget`
  - `ClutterTarget`
  - `OtherTarget`
- `Carrada`
  - `CarTarget`
  - `PeopleTarget`
  - `CyclistTarget`

### 12.2 固定评估协议

所有阶段都必须固定以下评估口径：

- `Legacy-only`
- `CQ-08-only`
- `Carrada-only`
- `overall unified`

每轮统一记录：

- `accuracy`
- `macro_f1`
- confusion matrix
- 关键错误链

关键错误链至少包括：

- `PeopleTarget -> CyclistTarget`
- `CyclistTarget -> PeopleTarget`
- `BirdTarget -> OtherTarget`
- `ClutterTarget -> BirdTarget`
- `ClutterTarget -> OtherTarget`

### 12.3 阶段划分

#### Phase A：单源基线

先分别训练并评估：

1. `Legacy-only`
2. `CQ-08-only`
3. `Carrada-only`

目的：

- 判断每个来源单独是否可学
- 排除“来源本身就学不动”的情况

#### Phase B：两两组合

在单源基线后，依次训练并评估：

1. `Legacy + CQ-08`
2. `Legacy + Carrada`
3. `CQ-08 + Carrada`

目的：

- 判断是哪一对来源一混就出问题
- 判断 source shift 和标签空间冲突主要来自哪一侧

#### Phase C：完整10类联合训练

只有在前两阶段明确了主要冲突来源后，才进入完整 `10-class` 联合训练。

当前统一10类为：

- `AircraftTarget`
- `BirdTarget`
- `DroneTarget`
- `ShipTarget`
- `BalloonTarget`
- `ClutterTarget`
- `OtherTarget`
- `CarTarget`
- `PeopleTarget`
- `CyclistTarget`

### 12.4 当前推荐顺序

现阶段推荐顺序如下：

1. `Legacy-only`
2. `CQ-08-only`
3. `Carrada-only`
4. `Legacy + CQ-08`
5. `Legacy + Carrada`
6. `CQ-08 + Carrada`
7. `All 10`

这套顺序的目标不是最省时间，而是最快定位：

- `CQ-08` 是否是主要扰动源
- `Carrada` 是否是主要扰动源
- 是哪一对来源最不兼容

### 12.5 训练原则

所有阶段统一遵守：

- 继续使用 `session-only` 主线
- `clip_loss_weight = 0`
- 默认关闭不稳定的专项 `session auxiliary heads`
- 保留：
  - `source-aware split`
  - `group-aware split`
  - `source-aware + class-aware sampler`
- 不随意更换主干和评估协议，避免阶段之间不可比较

### 12.6 判断标准

只有当某个新增来源满足以下条件，才允许继续进入下一阶段：

- 原来源主指标不发生明显崩坏
- 新来源指标显著高于随机水平
- 关键混淆链没有出现灾难性恶化

建议观察规则：

- 若加入某来源后，原来源 `macro_f1` 大幅下降，则该来源为高风险扰动源
- 若某来源单独可学，但一混合即退化，则主要问题在 cross-source conflict
- 若某来源单独都学不稳，则先处理该来源内部问题，不进入更大联合训练
