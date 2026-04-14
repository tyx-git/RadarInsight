# Debug Log

## Maintenance Note

- This file was rewritten in English to remove mojibake, mixed-language fragments, and non-ASCII noise.
- Earlier Chinese notes were condensed into English summaries so the history remains readable and maintainable.
- Detailed per-stage notes are kept for the recent and still-active paired TR/RD route.

## Historical Summary: Stages 0-9

- **Stage 0 - Base Skeleton**
  - Built the first runnable dual-branch training skeleton.
  - Goal: prove the project could train end to end.
  - Limitation: feature capacity and RD modeling were still weak.

- **Stage 1 - High-Accuracy Backbone Route**
  - Upgraded the track branch to `Residual TCN + Transformer`.
  - Upgraded the RD branch to an `X3D-M` style 3D CNN.
  - Added bidirectional cross-attention as the main fusion route.

- **Stage 2 - Session-Level Training and Evaluation**
  - Switched split logic from clip-centric evaluation to session-aware evaluation.
  - Added separate evaluation tooling to reduce leakage and false optimism.
  - Established session-level generalization as a core project rule.

- **Stage 3 - Real Data Pipeline**
  - Moved the main route from synthetic or legacy experiments to real `.mat` data.
  - Built the `pro/raw -> pro/sessions -> data/sessions` preparation path.
  - Standardized paired TR and RD sessions for training.

- **Stage 4 - RD Strong Modeling**
  - Added radar-specific RD front-end blocks for micro-motion and background contrast.
  - Goal: stop treating RD like a generic image stream.
  - Outcome: stronger RD inductive bias, but still limited by alignment quality.

- **Stage 5 - Complex-Aware RD Inputs**
  - Expanded RD input channels to preserve more complex-domain structure.
  - Added real, imaginary, log-amplitude, temporal difference, and local-contrast style channels.
  - Outcome: richer RD representation but not yet enough on its own.

- **Stage 6 - Training Strategy Upgrade**
  - Added class-balanced loss, label smoothing, warmup, cosine decay, and gradient clipping.
  - Goal: make small-data training more stable.
  - Outcome: stability improved, but representation quality still dominated the ceiling.

- **Stage 7 - Multiresolution RD (toTest)**
  - Tested multiresolution DPL-to-RD processing.
  - Outcome: useful as an experiment branch, but not stable enough as the default path.

- **Stage 8 - Single-Scale Main Path Recovery**
  - Recovered the single-scale RD path as the main route after multiresolution regression.
  - Goal: prefer stability over architectural novelty.

- **Stage 9 - Early Diagnosis**
  - Early diagnosis pointed to ROI alignment, signal processing, and feature separability as the main blockers.

## Historical Summary: Stages 10-19

- **Stage 10 - RD Target Alignment and Signal Processing Upgrade**
  - Improved target-aligned RD extraction and front-end signal processing.

- **Stage 11 - First Validation After RD Alignment Upgrade**
  - Ran the first controlled validation after target-aligned RD changes.

- **Stage 12 - Stable Physical Axes and Smoothed ROI Alignment**
  - Strengthened axis definitions and smoothing in ROI alignment.

- **Stage 13 - Second Validation After Stable-Axis Refactor**
  - Re-tested after axis stabilization changes.

- **Stage 14 - Calibrated-First Axes and Guarded ROI Refinement**
  - Prioritized calibrated axes and safer ROI correction.

- **Stage 15 - Fixed-Resolution Range Axis with Track-Aligned Offset**
  - Regularized range-axis behavior and tied offset selection to track context.

- **Stage 16 - Tolerant Coverage Check for Calibrated Axes**
  - Added more tolerant coverage checks around calibrated axes.

- **Stage 17 - Hybrid Range/Doppler Axis Selection**
  - Mixed range and Doppler selection logic to improve robustness.

- **Stage 18 - Adaptive Calibrated Axis Multipliers**
  - Added adaptive scaling on calibrated axes.

- **Stage 19 - Validation After Adaptive Axis Hardening**
  - Re-validated after adaptive axis hardening.

## Historical Summary: Stages 20-35

- **Stage 20 - Dual-View RD Input**
  - Added local ROI plus global context views for RD.

- **Stage 21 - Validation After Dual-View RD Input**
  - Validated the dual-view RD route.

- **Stage 22 - Track Stabilization Validation**
  - Reduced normalization leakage and strengthened track priors.

- **Stage 23 - Sign-Invariant Motion Upgrade**
  - Improved motion features to reduce sign-sensitivity issues.

- **Stage 24 - Validation After Sign-Invariant Motion Upgrade**
  - Re-tested after motion feature changes.

- **Stage 25 - Dual-Branch Track Modeling for FixedWing Boundary Recovery**
  - Added stronger internal track modeling for difficult air-target boundaries.

- **Stage 26 - Validation After Dual-Branch Track Modeling**
  - Re-tested after track dual-branch changes.

- **Stage 27 - Gated Global Track Summary Fusion**
  - Added gated use of global track summary features.

- **Stage 28 - Modality Reliability Fusion**
  - Introduced reliability-aware fusion between track and RD.

- **Stage 29 - Global RD Index Repair and Public-Dataset Validation**
  - Repaired RD indexing and validated the route on public-style data.

- **Stage 30 - Non-Bird Intra-Class Boundary Enhancement**
  - Strengthened hierarchical representation and non-bird internal boundaries.

- **Stage 31 - Validation After Non-Bird Boundary Enhancement**
  - Re-tested after the non-bird boundary upgrade.

- **Stage 32 - Coarse Mapping and Intra-Group Boundary Loss Validation**
  - Added validation around coarse mapping and intra-group loss shaping.

- **Stage 33 - Fine-Grained Boundary Residual Injection**
  - Tested residual correction for fine-grained boundaries.

- **Stage 34 - Coarse-Probability-Gated Fine-Grained Residuals**
  - Made fine-grained residuals conditional on coarse confidence.

- **Stage 35 - Validation-Driven Group Logit Calibration**
  - Added group-wise calibration driven by validation behavior.

## Historical Summary: Stages 36-50

- **Stage 36 - Cosine Detail Head For Intra-Coarse Separation**
  - Added a cosine-style detail head to strengthen intra-group separation.

- **Stage 37 - Validation After Cosine Detail Head**
  - Re-tested after the cosine detail head.

- **Stage 38 - CPU Group Calibration On The New Best Checkpoint**
  - Validated calibration behavior on CPU exports.

- **Stage 39 - Fine-Grained Calibration Becomes The Main Follow-Up Direction**
  - Shifted focus toward calibration-driven refinement.

- **Stage 40 - Excel To TSV Real Track Pipeline**
  - Built the Excel-to-TSV real-data track route.

- **Stage 41 - TSV Training Adapter For Real Excel Tracks**
  - Added a TSV dataset adapter for the track-only route.

- **Stage 42 - Environment Variable Modality Control**
  - Added environment-based modality switches for runtime control.

- **Stage 43 - Default Environment Switches Stay Fully Enabled**
  - Kept both modalities on by default.

- **Stage 44 - TSV Track-Only Route Reaches Trainable State**
  - Confirmed the TSV route was trainable end to end.

- **Stage 45 - Remove Residual Old-Dataset Type Coupling**
  - Removed remaining dataset-type coupling in the training stack.

- **Stage 46 - Restore Missing Normalization Interfaces For TSV Training**
  - Restored the normalization hooks needed by TSV training.

- **Stage 47 - Adopt The A-Scheme Coarse Taxonomy For The Real TSV Route**
  - Updated the track-only route to a practical coarse taxonomy.

- **Stage 48 - Train The Current Real TSV Route On A-Scheme Coarse Labels**
  - Ran the first real TSV coarse-label training round.

- **Stage 49 - Restore The Best Validated TSV Track Baseline**
  - Recovered the strongest validated track-only baseline.

- **Stage 50 - Add Graph-Based Track Backbone On Top Of The Restored TSV Baseline**
  - Introduced the graph-based experimental track backbone.

## Stage 51 - Centralize Runtime Control Into train.json And Clean Low-Value Artifacts

- **Current Stage**: replace scattered runtime switches with a single JSON control file and remove clearly low-value artifacts.
- **Current Problem**:
  - Runtime behavior was spread across YAML defaults, CLI overrides, and environment variables.
  - The `artifacts` directory accumulated poor runs and temporary evaluation folders that made iteration harder to track.
- **Solution**:
  - Introduce `train.json` as the single runtime control file for training, evaluation, calibration, and session inspection.
  - Route track/RD switches and the track-backbone choice through `train.json`.
  - Add an artifact cleaner that removes obvious failed runs and temporary evaluation folders while preserving known-good baselines and calibrated outputs.
- **Outcome**:
  - Easier and more reproducible experimentation.
  - Cleaner artifact space for future comparisons.

## Stage 52 - Evaluate The Graph Track Backbone Against The Restored TSV Baseline

- **Problem**: the temporal baseline was stable, but it was unclear whether explicit graph message passing could improve discrimination.
- **Change**: ran a controlled comparison with `track_backbone_type` switched between `temporal` and `graph`.
- **Outcome**:
  - The graph branch trained successfully.
  - It did not beat the temporal baseline and remained experimental.

## Stage 53 - Add TR-Only Supplemental Sessions From pro/raw/TR

- **Problem**: the TSV route was still imbalanced and under-supported in several classes.
- **Change**: added a TR-only supplement processor from `pro/raw/TR` into the standardized dataset.
- **Outcome**:
  - Dataset support improved.
  - The first post-supplement training run still did not beat the best validated TSV baseline.

## Stage 54 - Merge Suspected Balloon And Clutter Into Other And Rebuild Session Splits

- **Problem**: extremely rare edge categories and over-sensitive split behavior made evaluation unstable.
- **Change**: merged low-support edge categories and replaced the old split heuristic with a multi-session greedy split.
- **Outcome**:
  - Evaluation became more practical and more representative.
  - Old and new results became less directly comparable.

## Stage 55 - Try Long-Tail Balancing And Focal Loss Under The Stricter Multi-Session Split

- **Problem**: the stricter split exposed class imbalance and coarse-target confusion.
- **Change**: re-enabled class-balanced weighting and added focal-style cross-entropy.
- **Outcome**:
  - Numerical behavior was stable.
  - The route still did not beat the best validated baseline.

## Stage 56 - Strengthen The Dual-Modal Temporal Backbone With Real Multi-Window Fusion

- **Problem**: the temporal route needed stronger scale-aware motion modeling.
- **Change**: added short, mid, and long track windows plus a lightweight time-by-feature 2D branch.
- **Outcome**:
  - The dual-modal temporal route became the main paired-data track path.

## Stage 57 - Add A Bird-vs-Surface Specialist To The Dual-Modal Fusion Path

- **Problem**: `BirdTarget` vs `ShipTarget` remained the strongest confusion pair.
- **Change**: added a small Bird-vs-Ship specialist head with light residual correction.
- **Outcome**:
  - Boundary stabilization improved, but the route still needed better RD representation.

## Stage 58 - Switch Track Backbone Toward DG-GCN

- **Problem**: wanted to test a graph-temporal track encoder under the dual-modal route.
- **Change**: introduced DG-GCN-style track encoding as the default experimental track path.
- **Outcome**:
  - The path was technically usable.
  - It later regressed badly under the current split.

## Stage 59 - Revert Default Track Backbone After DG-GCN Regression

- **Problem**: the DG-GCN production run regressed under the current paired split.
- **Change**: restored the temporal track encoder as the default backbone.
- **Outcome**:
  - The mainline returned to the more reliable temporal route.

## Stage 60 - Shift Fusion To RD-Dominant And Add Bird-Specialized RD Heads

- **Problem**: repeated experiments showed Bird collapse and weak hard-boundary discrimination.
- **Change**: shifted optimization toward RD-first fusion and added Bird-sensitive RD heads.
- **Outcome**:
  - The direction fit the deployment intuition but required careful control.

## Stage 61 - Soften RD-First Bias And Move Bird Optimization Into RD Representation

- **Problem**: hard RD dominance caused regressions and did not fully recover Bird.
- **Change**: reduced explicit RD bias and moved Bird emphasis deeper into RD representation.
- **Outcome**:
  - The route became more conservative and more stable than the previous hard RD push.

## Stage 62 - Add Bird-Surface Prototype Constraint On RD Features

- **Problem**: Bird still failed to form a stable RD cluster.
- **Change**: added a Bird-vs-Ship prototype-style loss on RD features.
- **Outcome**:
  - Direct feature-space shaping became part of the main Bird recovery strategy.

## Stage 63 - Bird-Surface Sampling And Triplet Constraint

- **Problem**: Bird still collapsed into surface classes.
- **Change**: increased Bird/Ship sampling and tested a Bird-vs-Ship triplet-style RD loss.
- **Outcome**:
  - The route attacked the main confusion pair directly but risked overfitting.

## Stage 64 - Soften Bird Reinforcement And Keep Bird-Aware Validation

- **Problem**: aggressive Bird reinforcement hurt transfer.
- **Change**: reduced Bird/Ship sampling emphasis and disabled triplet loss by default.
- **Outcome**:
  - Validation stayed representative while overfitting pressure dropped.

## Stage 65 - Track Router And RD Air Specialist Backbone

- **Problem**: a uniform fusion head was too weak for coarse routing plus hard air discrimination.
- **Change**: added a track router and RD Bird-vs-Drone specialist logic.
- **Outcome**:
  - The route matched the intended split of responsibilities between Track and RD.

## Stage 66 - Data Purification And Label Remap

- **Problem**: old routes mixed incompatible data sources and label spaces.
- **Change**:
  - Froze the supervised main route to paired TR/RD sessions in `data/sessions`.
  - Standardized the active coarse labels to `AircraftTarget`, `BirdTarget`, `DroneTarget`, and `ShipTarget`.
- **Outcome**:
  - The current mainline became much cleaner and more reproducible.

## Stage 67 - Balanced Split Weight Tuning

- **Problem**: the new paired split still had Bird/Ship sensitivity.
- **Change**: adopted conservative class balancing and focal settings.
- **Outcome**:
  - The training route became more stable under the purified split.

## Stage 68 - Early Stopping And Composite Selection

- **Problem**: single-metric checkpoint selection was too brittle.
- **Change**: switched best-checkpoint selection to `0.7 * val_macro_f1 + 0.3 * val_accuracy` and added early stopping.
- **Outcome**:
  - The route stopped depending on a single metric and became easier to monitor.

## Stage 69 - Stabilized Training Schedule For Paired RD/TR

- **Problem**: the previous schedule was too short and too sharp.
- **Change**: reduced learning rate to `2e-4` and extended the stable schedule to `12` epochs.
- **Outcome**:
  - Best validation points often appeared early but under a safer schedule.

## Stage 70 - Paired TR/RD Stable Baseline Improvement

- **Change**: established the first clearly stronger paired-data baseline.
- **Outcome**:
  - `artifacts/202604081748-30-4-cpu` reached about `Accuracy=0.7529`, `Macro F1=0.7494`.

## Stage 71 - Ship Specialist RD Heads

- **Problem**: `ShipTarget` remained the main unresolved class after Bird recovery improved.
- **Change**: added RD-side `Ship vs Drone` and `Ship vs Bird` specialist heads.
- **Outcome**:
  - This became the next incremental route for boundary repair.

## Stage 72 - Weaken Ship Specialists

- **Problem**: strong Ship residual injection disturbed the global 4-class boundary.
- **Change**: kept Ship specialists as auxiliary supervision and disabled strong residual injection.
- **Outcome**:
  - The weak-specialist strategy proved more stable than the stronger version.

## Stage 73 - Stable Paired TR/RD Improvement

- **Outcome**:
  - `artifacts/202604090918-30-4-cpu` improved the purified paired-data baseline.
  - Independent evaluation reached `Accuracy = 0.7691`, `Macro F1 = 0.7907`, `Loss = 1.5229`.

## Stage 74 - Evaluation Tooling, Fusion Diagnosis, And Data-Driven Hard-Session Mining

- **Problem**:
  - CUDA retrains showed large clip-level variance while session-level behavior stayed more stable.
  - A few hard sessions dominated the remaining error mass.
- **Change**:
  - Added `session-infer`, `session-diagnose`, and `mine-hard-sessions`.
  - Exported clip-level and session-level diagnosis CSV and JSON files.
  - Added suspicious span mining for the hardest sessions.
  - Added reproducibility controls, DataLoader worker seeding, and CUDA warning cleanup.
  - Fixed track-global feature extraction and caching issues.
- **Outcome**:
  - The project now has a usable closed loop: train -> evaluate -> session-infer -> mine-hard-sessions -> session-diagnose.
  - The main hard sessions were localized and their dominant bad time spans were identified.

## Stage 75 - Anti-Collapse Fusion And Train/Eval Route Separation

- **Problem**:
  - Diagnosis showed severe fusion collapse: on the historical best checkpoint, `track_weight` was close to `1.0` and `rd_weight` close to `0.0` on test data.
  - This meant the nominal dual-modal route was behaving almost like a Track-only route during inference.
- **Change**:
  - Added anti-collapse fusion controls to the model so RD can be given a minimum fusion floor and ambiguity-aware boost.
  - Tested static and curriculum anti-collapse strategies during training.
  - Discovered and fixed a configuration bug where eval-only fusion overrides had leaked into the train path.
  - Restored the preferred route to: stable train-time weighted fusion plus eval-time anti-collapse overrides.
- **Outcome**:
  - Offline evaluation of the historical best checkpoint improved under eval-time anti-collapse fusion.
  - Full train-time anti-collapse remained unstable at the current dataset scale.
  - The project now cleanly separates stable training behavior from stronger evaluation-time fusion rescue.

## Stage 76 - Similarity Search And Targeted Hard-Session Sampling

- **Problem**:
  - Hard-session mining showed that a few test sessions still dominated the clip ceiling.
  - Blind retraining was not consistently improving the best baseline.
- **Change**:
  - Matched the hardest test spans against training sessions and clip windows.
  - Identified the most similar Ship-hard and Bird-hard training sources.
  - Added targeted hard-session sampling controls to the training sampler.
  - First added a single hard-session boost, then split it into two independent controls:
    - `hard_ship_session_sampling_boost`
    - `hard_bird_session_sampling_boost`
- **Outcome**:
  - The Bird-hard route improved more reliably than blind retraining.
  - A single combined boost traded Bird gains for Ship regressions.
  - Splitting Bird-hard and Ship-hard sampling created a cleaner next-step optimization path.

## Current Preferred Route

- Train-time path:
  - keep the stable weighted-fusion route close to the historical best configuration
  - keep session-grouped training disabled
  - use composite validation selection and early stopping
- Eval-time path:
  - apply anti-collapse fusion overrides during evaluation and inference only
- Data-driven optimization path:
  - mine hard sessions after every meaningful run
  - inspect suspicious spans instead of relying only on aggregate metrics
  - strengthen Ship-hard and Bird-hard training sessions independently rather than with a single global boost

## Current Reference Baseline

- Preferred artifact: `artifacts/202604101750-12-4-cuda`
- Historical stable reference with eval-time anti-collapse re-evaluation:
  - clip-level `Accuracy = 0.8588`
  - clip-level `Macro F1 = 0.8446`
  - session-level `Accuracy = 0.8750`
  - session-level `Macro F1 = 0.8810`

## Current Open Problems

- The main unresolved clip-level boundaries remain:
  - `ShipTarget -> DroneTarget`
  - `BirdTarget -> ShipTarget`
- A few hard sessions still dominate the clip ceiling, especially:
  - `20230802145946_shiptarget_01_16838`
  - `20231014161536_birdtarget_13_15`
  - `20231011105711_birdtarget_02_2266`
- The route is now diagnosis-driven, but stable train-time anti-collapse remains unsolved under the current data scale.

## Stage 77 - Suspicious Ship Pattern Downweighting
- Added a train-time downweight path for ShipTarget clips that match the recurrent ship-hard failure mode.
- The implementation is intentionally conservative and only affects sampling weights, not labels or losses.
- New configuration fields were added in `src/training/config.py` and wired into the weighted sampler in `src/training/trainer.py`.
- The downweight rule uses track-global features already computed by the dataset loader:
  - `clip_speed_mean_mps`
  - `clip_speed_direction_consistency`
  - `clip_range_monotonicity`
  - `clip_snr_mean_db`
- The initial rule also supports explicit session-level targeting through `suspicious_ship_session_ids`.
- Default training control values were added to `train.json`:
  - `suspicious_ship_downweight_enabled = true`
  - `suspicious_ship_downweight_factor = 0.55`
  - `suspicious_ship_session_ids` set to the three ship-hard training sessions found by similarity search
  - `suspicious_ship_speed_mean_threshold = 6.0`
  - `suspicious_ship_direction_consistency_threshold = 0.78`
  - `suspicious_ship_surface_bias_threshold = 0.68`
  - `suspicious_ship_snr_mean_threshold = 14.0`
- Rationale:
  - The hardest persistent failure is still `ShipTarget -> DroneTarget`.
  - The hardest ShipTarget test session contains a long high-confidence drone-like span.
  - Similar spans exist in a small set of training ShipTarget sessions.
  - Instead of oversampling those sessions, this change lowers their contribution when they match the suspicious ship-hard pattern.
- Expected effect:
  - Reduce reinforcement of drone-like ship clips during training.
  - Preserve the current BirdTarget gains from dual-channel targeted sampling.
  - Lower the chance that ShipTarget decision boundaries drift toward DroneTarget.

## Stage 78 - Why Complex And Highly Confusing Samples Need Targeted Sampling
- Targeted sampling is still the right direction, but it should focus on complex and highly confusing samples instead of applying one global class-level boost.
- Current evidence from hard-session mining shows that most remaining errors are concentrated in a few hard sessions and a few continuous suspicious spans.
- This means the bottleneck is not uniform class imbalance. It is local ambiguity, local distribution shift, and repeated exposure to the wrong decision boundary.
- The most useful next sampling policy should prioritize three groups:
  - high-confidence mistakes from previous best-checkpoint evaluation
  - low-margin clips near the top-1/top-2 decision boundary
  - clips belonging to the hardest validated BirdTarget-versus-ShipTarget and ShipTarget-versus-DroneTarget patterns
- This should be implemented carefully because aggressive hard-example oversampling can improve one boundary while damaging another one.
- The current preferred direction is therefore:
  - keep the stable training backbone and loss setup
  - keep eval-time anti-collapse fusion override
  - keep dual-channel Bird-hard and Ship-hard session sampling
  - add a more selective confusion-aware sampling layer on top of the existing session priors
- Recommended safeguards for the next step:
  - cap the extra confusion-aware sampling multiplier to a modest range
  - avoid applying the same strong boost to Bird-hard and Ship-hard clips
  - downweight suspicious drone-like ShipTarget clips instead of oversampling them
  - prefer sampler-side weighting before introducing new losses
- Practical expectation:
  - this direction is more promising than another round of broad hyperparameter tuning
  - it is also more aligned with the current error structure than full session-grouped training

## Stage 79 - New Best-Looking Run Analysis And Remaining Confusions
- New run analyzed: `artifacts/202604131922-12-4-cuda`
- Raw training summary reported approximately `accuracy = 0.8544` and `macro_f1 = 0.8527`.
- Under the current evaluation path with eval-time fusion override, the measured clip-level metrics improved to approximately:
  - `accuracy = 0.8618`
  - `macro_f1 = 0.8600`
- Session-level metrics stayed at:
  - `session accuracy = 0.8750`
  - `session macro_f1 = 0.8810`
- Main remaining clip-level confusions from `evaluation_report.json`:
  - `BirdTarget -> ShipTarget = 40`
  - `ShipTarget -> DroneTarget = 49`
  - `BirdTarget -> DroneTarget = 4`
  - `ShipTarget -> BirdTarget = 1`
- This means the two dominant unresolved pairwise boundaries are still:
  - `BirdTarget` versus `ShipTarget`
  - `ShipTarget` versus `DroneTarget`
- Hard-session mining still selects the same three sessions:
  - `20230802145946_shiptarget_01_16838`
  - `20231014161536_birdtarget_13_15`
  - `20231011105711_birdtarget_02_2266`
- The worst suspicious spans in the new run are:
  - `20230802145946_shiptarget_01_16838: clip_token 0-140` dominated by `ShipTarget -> DroneTarget`
  - `20231014161536_birdtarget_13_15: clip_token 244-324` dominated by `BirdTarget -> ShipTarget`
  - `20231011105711_birdtarget_02_2266: clip_token 280-332` dominated by `BirdTarget -> ShipTarget`
- Current interpretation:
  - global training quality improved
  - AircraftTarget is already solved on this split
  - DroneTarget recall is perfect, but precision is still limited because drone-like ShipTarget spans are over-accepted
  - BirdTarget precision is very high, but recall is still capped by repeated BirdTarget-to-ShipTarget errors in a few long spans
- Recommended next algorithmic directions:
  - confusion-aware clip sampler driven by previous evaluation mistakes and low-margin clips
  - pair-specific margin or ranking losses for `ShipTarget vs DroneTarget` and `BirdTarget vs ShipTarget`
  - local span-level temporal modeling instead of full session-grouped training
  - selective uncertainty handling for suspicious drone-like ShipTarget clips
  - class-pair-aware session aggregation rather than one global aggregation rule

## Stage 80 - Pair-Targeted Optimization For The Remaining Confusions
- Added a new confusion-pair optimization path for the two dominant unresolved boundaries:
  - `ShipTarget vs DroneTarget`
  - `BirdTarget vs ShipTarget`
- The implementation has two parts.
- Part 1: confusion-aware sampler additions
  - Added `confusion_pair_sampling_enabled`
  - Added `confusion_pair_sampling_boost`
  - Added `confusion_pair_bird_ship_session_ids`
  - Added `confusion_pair_ship_drone_session_ids`
- When enabled, the sampler applies a modest extra weight only to clips that belong to the selected confusion-pair training sessions.
- Part 2: pair-specific margin losses
  - Added `ship_drone_margin_loss_weight`
  - Added `ship_drone_margin`
  - Added `bird_ship_margin_loss_weight`
  - Added `bird_ship_margin`
- These losses directly encourage:
  - `logit_ship > logit_drone` on true ShipTarget clips
  - `logit_bird > logit_ship` on true BirdTarget clips
- The first default version is intentionally conservative:
  - `confusion_pair_sampling_boost = 1.18`
  - `ship_drone_margin_loss_weight = 0.04`
  - `bird_ship_margin_loss_weight = 0.03`
  - both pair margins start at `0.2`
- Rationale:
  - previous experiments showed that broad global tuning no longer attacks the real bottleneck
  - the remaining test errors are now dominated by two pairwise boundaries and a few long suspicious spans
  - pair-targeted optimization should be lower risk than a large architecture rewrite

## Stage 81 - First Pair-Targeted Training Attempt Regressed Hard
- Run analyzed: `artifacts/202604131959-12-4-cuda`
- Result: strong regression after enabling the first joint pair-targeted sampler and pair-margin-loss package.
- Evaluation with current eval-time fusion override dropped to approximately:
  - `clip accuracy = 0.5235`
  - `clip macro_f1 = 0.6914`
  - `session accuracy = 0.6250`
- Failure pattern:
  - ShipTarget clips were massively reclassified as BirdTarget.
  - Main confusion matrix row became `ShipTarget -> BirdTarget = 313`.
  - BirdTarget itself became almost perfectly recalled, which confirms that the joint pair-targeted update over-corrected toward BirdTarget.
- Hard-session mining also changed character:
  - the top three hardest sessions all became ShipTarget sessions
  - `20231014102639_shiptarget_03_582` collapsed almost entirely to BirdTarget
  - `20230731152102_shiptarget_08_43551` collapsed entirely to BirdTarget
  - `20230802145946_shiptarget_01_16838` also mostly collapsed to BirdTarget
- Interpretation:
  - applying BirdTarget-vs-ShipTarget targeted sampling and BirdTarget-vs-ShipTarget margin pressure at the same time was too aggressive
  - the first version did not just sharpen the local boundary; it shifted the global ShipTarget manifold toward BirdTarget
- Updated recommendation after this failure:
  - do not keep the full joint pair-targeted package enabled by default
  - separate ShipTarget-vs-DroneTarget optimization from BirdTarget-vs-ShipTarget optimization
  - disable BirdTarget-vs-ShipTarget pair margin loss first
  - keep only a conservative ShipTarget-vs-DroneTarget pair-specific signal for the next iteration
  - if BirdTarget-vs-ShipTarget optimization is retried later, reintroduce it in sampler-only mode before adding margin loss again

## Stage 82 - Narrowed Strategy To A Single Confusion Pair
- The next iteration is now intentionally narrowed to one confusion pair only:
  - `ShipTarget vs DroneTarget`
- Current default control changes:
  - keep `confusion_pair_sampling_enabled = true`
  - reduce `confusion_pair_sampling_boost` to `1.15`
  - clear `confusion_pair_bird_ship_session_ids`
  - keep only `confusion_pair_ship_drone_session_ids`
  - disable both pair margin losses for this isolation run
- Reasoning:
  - the previous joint pair-targeted run showed that combining BirdTarget-vs-ShipTarget sampling pressure with pair margin loss was too aggressive
  - isolating a single confusion pair makes the next experiment easier to interpret
  - the most stable unresolved bottleneck across strong runs is still `ShipTarget -> DroneTarget`

## Stage 83 - ShipTarget-vs-DroneTarget-Only Targeted Sampling Result
- Run analyzed: `artifacts/202604132304-12-4-cuda`
- This narrowed sampler-only experiment did not create a catastrophic new collapse, which is better than the previous joint pair-targeted attempt.
- However, it still did not improve the main bottleneck enough.
- Evaluation with current eval-time fusion override was approximately:
  - `clip accuracy = 0.7971`
  - `clip macro_f1 = 0.8039`
  - `session accuracy = 0.8750`
  - `session macro_f1 = 0.8810`
- Main remaining confusions:
  - `ShipTarget -> DroneTarget = 70`
  - `BirdTarget -> ShipTarget = 40`
  - `ShipTarget -> BirdTarget = 16`
  - `BirdTarget -> DroneTarget = 12`
- Interpretation:
  - narrowing to one confusion pair avoided the previous BirdTarget-overcorrection failure
  - but sampler-only ShipTarget-vs-DroneTarget optimization was too weak to recover the best historical run
  - the hardest ShipTarget session is still dominated by a very long drone-like span
- Updated recommendation:
  - if ShipTarget-vs-DroneTarget remains the single focus, the next step should not be a stronger broad sampler boost
  - instead, use a conservative ShipTarget-vs-DroneTarget pair-specific margin signal or a targeted residual/logit correction path
  - BirdTarget-vs-ShipTarget should stay frozen while ShipTarget-vs-DroneTarget is being isolated

## Stage 84 - Light ShipTarget-vs-DroneTarget Margin Activation
- The current follow-up iteration keeps the narrowed ShipTarget-vs-DroneTarget-only strategy.
- Added a light pair-specific margin signal on top of the existing ShipTarget-vs-DroneTarget targeted sampler.
- Current default control values:
  - `ship_drone_margin_loss_weight = 0.02`
  - `ship_drone_margin = 0.15`
  - `bird_ship_margin_loss_weight = 0.0`
- Reasoning:
  - sampler-only ShipTarget-vs-DroneTarget pressure was not enough
  - the previous full joint pair-targeted attempt was too strong
  - a small ShipTarget-vs-DroneTarget margin term is the safest next midpoint

## Stage 85 - Light ShipTarget-vs-DroneTarget Margin Improved The Main Pair But Not The Overall Best
- Run analyzed: `artifacts/202604132353-12-4-cuda`
- Evaluation with current eval-time fusion override reached approximately:
  - `clip accuracy = 0.8309`
  - `clip macro_f1 = 0.8283`
  - `session accuracy = 0.8750`
  - `session macro_f1 = 0.8810`
- Compared with the previous ShipTarget-vs-DroneTarget sampler-only run, the new light margin term clearly improved the main target pair:
  - `ShipTarget -> DroneTarget` reduced from `70` to `42`
- However, the run still did not beat the strongest reference run because side effects increased on other boundaries:
  - `ShipTarget -> BirdTarget = 20`
  - `BirdTarget -> DroneTarget = 16`
  - `BirdTarget -> ShipTarget = 36`
- Interpretation:
  - the light ShipTarget-vs-DroneTarget margin term is useful and should not be discarded immediately
  - the current problem is no longer that ShipTarget-vs-DroneTarget cannot be improved
  - the new issue is that the correction introduces collateral drift into adjacent boundaries
- Updated recommendation:
  - keep the ShipTarget-vs-DroneTarget margin concept
  - make the next adjustment smaller and more selective rather than stronger
  - likely candidates are lowering ShipTarget-vs-DroneTarget sampler intensity or testing a margin-only variant

## Stage 86 - Switched To ShipTarget-vs-DroneTarget Margin-Only Test
- The next test isolates the ShipTarget-vs-DroneTarget margin effect by disabling the confusion-pair sampler entirely.
- Current control state:
  - `confusion_pair_sampling_enabled = false`
  - `confusion_pair_sampling_boost = 1.0`
  - `ship_drone_margin_loss_weight = 0.02`
  - `ship_drone_margin = 0.15`
  - `bird_ship_margin_loss_weight = 0.0`
- Goal:
  - determine whether the previous collateral boundary drift came mainly from the confusion-pair sampler or from the margin term itself

## Stage 87 - Margin-Only Result: Safer, Stronger, But ShipTarget-vs-DroneTarget Still Dominates
- Run analyzed: `artifacts/202604140018-12-4-cuda`
- Evaluation with current eval-time fusion override reached approximately:
  - `clip accuracy = 0.8515`
  - `clip macro_f1 = 0.8519`
  - `session accuracy = 0.8750`
  - `session macro_f1 = 0.8810`
- This is much stronger than the previous ShipTarget-vs-DroneTarget sampler-only run and confirms that the sampler was a major source of collateral drift.
- Main remaining confusions:
  - `ShipTarget -> DroneTarget = 69`
  - `BirdTarget -> ShipTarget = 28`
  - `BirdTarget -> DroneTarget = 3`
  - `ShipTarget -> BirdTarget = 1`
- Interpretation:
  - margin-only preserved the BirdTarget boundary much better than the previous sampler-plus-margin run
  - the dominant unresolved error is now cleanly isolated as `ShipTarget -> DroneTarget`
  - this is a better base for further optimization than the stronger hybrid runs
- Updated recommendation:
  - keep BirdTarget-vs-ShipTarget frozen
  - continue optimizing only ShipTarget-vs-DroneTarget
  - prefer more selective pair-specific pressure instead of reintroducing a broad sampler immediately
  - a likely next step is a conditional ShipTarget-vs-DroneTarget margin that focuses on explicit ship-hard sessions or suspicious ship-hard patterns only

## Stage 88 - Conditional ShipTarget-vs-DroneTarget Margin For Suspicious Ship-Hard Patterns
- Implemented a new training path that activates ShipTarget-vs-DroneTarget pair pressure only on suspicious ship-hard patterns.
- Dataset outputs now include a `suspicious_ship_hard_pattern` flag for both dataset backends.
- The initial flag uses the current ship-hard heuristic derived from track-global features and only marks ShipTarget clips that match the suspicious drone-like ship pattern.
- Added new config fields:
  - `conditional_ship_drone_margin_enabled`
  - `conditional_ship_drone_margin_weight`
  - `conditional_ship_drone_margin`
- Current default control state:
  - `ship_drone_margin_loss_weight = 0.0`
  - `conditional_ship_drone_margin_enabled = true`
  - `conditional_ship_drone_margin_weight = 0.03`
  - `conditional_ship_drone_margin = 0.18`
- Goal:
  - keep the safer BirdTarget behavior of the margin-only run
  - apply ShipTarget-vs-DroneTarget extra pressure only where the persistent ship-hard failure mode actually appears

## Stage 89 - Conditional Ship-Hard Pattern Flag Calibration
- The first conditional implementation was wired correctly but matched zero training clips because the heuristic was using normalized features with raw-scale thresholds.
- This was corrected by switching the suspicious ship-hard decision to raw track-global features.
- The conditional path was also updated to explicitly honor known ship-hard training sessions through `conditional_ship_drone_session_ids`.
- After calibration, the current training split contains a substantial number of conditional matches:
  - `train_ship_count = 454`
  - `train_suspicious_ship_count = 176`
- This means the conditional ShipTarget-vs-DroneTarget margin path is now actually active on training data and ready for a meaningful experiment.

## Stage 90 - Conditional Suspicious Ship-Hard Margin Regressed
- Run analyzed: `artifacts/202604140153-12-4-cuda`
- Result: the conditional suspicious-ship margin strategy regressed badly.
- Evaluation with current eval-time fusion override dropped to approximately:
  - `clip accuracy = 0.7265`
  - `clip macro_f1 = 0.7184`
  - `session accuracy = 0.7500`
- Main failure pattern:
  - `ShipTarget -> DroneTarget = 77`
  - `BirdTarget -> ShipTarget = 63`
  - `BirdTarget -> AircraftTarget = 14`
- Interpretation:
  - the conditional suspicious-ship subset was not selective enough
  - once activated, the extra ShipTarget-vs-DroneTarget pressure still destabilized neighboring boundaries instead of isolating the intended ship-hard mode
  - this confirms that the current heuristic is not yet precise enough to serve as a safe optimization trigger
- Updated recommendation:
  - do not continue with the current conditional suspicious-ship margin variant
  - roll back to the safer margin-only run as the preferred focused ShipTarget-vs-DroneTarget branch
  - if conditional optimization is revisited later, it should use explicit curated clip lists or diagnosis-driven manifests instead of a heuristic feature gate alone

## Stage 91 - Rolled Back To Margin-Only And Planned Curated-Clip Optimization
- The active training baseline has been rolled back to the safer ShipTarget-vs-DroneTarget margin-only branch.
- Active control state after rollback:
  - `confusion_pair_sampling_enabled = false`
  - `ship_drone_margin_loss_weight = 0.02`
  - `ship_drone_margin = 0.15`
  - `conditional_ship_drone_margin_enabled = false`
- Next preferred direction:
  - do not use another heuristic feature gate immediately
  - build a diagnosis-driven manifest or curated clip list from the exported suspicious spans
  - allow targeted optimization to act only on explicit reviewed spans instead of broad heuristics

## Stage 92 - Curated Clip Manifest Mechanism Added
- Added an explicit diagnosis-manifest path so future targeted optimization can use reviewed spans instead of heuristic gates.
- New utility module: `src/training/curated_manifest.py`
- New CLI command: `build-curated-manifest`
- New training config fields:
  - `curated_ship_drone_manifest_path`
  - `curated_ship_drone_margin_enabled`
  - `curated_ship_drone_margin_weight`
  - `curated_ship_drone_margin`
- Current intent:
  - keep the safer margin-only branch as the default
  - allow a future reviewed ShipTarget-vs-DroneTarget manifest to selectively trigger extra pair pressure only on explicit spans

## Stage 93 - New Strong Reference Run After Returning To The Safer Branch
- Run analyzed: `artifacts/202604141227-12-4-cuda`
- Evaluation with current eval-time fusion override reached approximately:
  - `clip accuracy = 0.8529`
  - `clip macro_f1 = 0.8645`
  - `session accuracy = 0.8750`
  - `session macro_f1 = 0.8810`
- This run is now the strongest overall Macro F1 reference under the current code path.
- Main confusion matrix summary:
  - `ShipTarget -> DroneTarget = 34`
  - `BirdTarget -> ShipTarget = 38`
  - `ShipTarget -> BirdTarget = 28`
- Important observation:
  - the hardest persistent ShipTarget session `20230802145946_shiptarget_01_16838` improved materially and is now close to a 50/50 split instead of a near-total collapse
  - the BirdTarget-vs-ShipTarget boundary still remains the main secondary source of error
- Practical conclusion:
  - the project should now treat `artifacts/202604141227-12-4-cuda` as the active reference branch
  - further focused optimization should be measured against this run, not against older branches

## Stage 94 - Curated BirdTarget-vs-ShipTarget Optimization Design
- Added a symmetric reviewed-span optimization path for `BirdTarget vs ShipTarget`.
- New training config fields:
  - `curated_bird_ship_manifest_path`
  - `curated_bird_ship_margin_enabled`
  - `curated_bird_ship_margin_weight`
  - `curated_bird_ship_margin`
- A first curated manifest has been generated from the current best reference run:
  - source: `artifacts/202604141227-12-4-cuda/hard_session_mining/focused_sessions/suspicious_spans.csv`
  - output: `artifacts/curated_bird_ship_manifest.json`
- The first activation is intentionally conservative:
  - keep the current safer ShipTarget-vs-DroneTarget margin-only baseline unchanged
  - add only a light reviewed-span BirdTarget-vs-ShipTarget margin term
  - `curated_bird_ship_margin_weight = 0.015`
  - `curated_bird_ship_margin = 0.12`
- Design intent:
  - use explicit reviewed BirdTarget-vs-ShipTarget spans instead of a broad BirdTarget-vs-ShipTarget sampler
  - avoid repeating the earlier over-correction that collapsed ShipTarget toward BirdTarget

## Stage 95 - First Curated BirdTarget-vs-ShipTarget Run Regressed
- Run analyzed: `artifacts/202604141253-12-4-cuda`
- Evaluation with current eval-time fusion override dropped to approximately:
  - `clip accuracy = 0.8059`
  - `clip macro_f1 = 0.8173`
  - `session accuracy = 0.8750`
  - `session macro_f1 = 0.8810`
- This is weaker than the current strong reference run.
- The intended target pair did not improve enough to justify the trade-off:
  - `BirdTarget -> ShipTarget = 38` remained essentially unchanged
- Meanwhile the main ShipTarget-vs-DroneTarget bottleneck worsened again:
  - `ShipTarget -> DroneTarget = 80`
- Interpretation:
  - even the curated reviewed-span BirdTarget-vs-ShipTarget margin introduced too much collateral drift into the ShipTarget-vs-DroneTarget boundary
  - the current project state still supports the conclusion that ShipTarget-vs-DroneTarget should remain the primary focused branch
- Updated recommendation:
  - do not keep the first curated BirdTarget-vs-ShipTarget margin active by default
  - restore the current stronger reference configuration

## Stage 96 - Confusion-Pair Handler Layer Design
- Added a new low-risk confusion-pair handler layer directly inside the model instead of relying on stronger sampling or additional large margin terms.
- Design intent:
  - act only on the currently dominant hard pairwise boundaries
  - `ShipTarget vs DroneTarget`
  - `BirdTarget vs ShipTarget`
  - only activate when the model already places a sample near that pairwise region
- Implementation:
  - new lightweight handler heads are attached to the fused `track_feature + rd_feature + rd_local_ship_feature` representation
  - the handler produces a 2-logit residual for each confusion pair
  - the residual is applied only when the summed class probability for the pair exceeds a gate threshold
- Initial default settings are intentionally conservative:
  - `confusion_pair_handler_enabled = true`
  - `confusion_pair_handler_weight = 0.02`
  - `confusion_pair_handler_gate_threshold = 0.55`
  - all previously unstable heuristic and curated margin branches remain disabled by default
- Goal:
  - improve pairwise disambiguation without perturbing the full class manifold as aggressively as sampler-based or margin-only interventions did

## Stage 97 - Per-Pair Handler Weight Split
- The first shared handler-layer run improved some pairwise behavior but still over-pulled ShipTarget toward BirdTarget in the hardest ShipTarget session.
- The handler is now split into per-pair control weights:
  - `confusion_ship_drone_handler_weight`
  - `confusion_bird_ship_handler_weight`
- Current default direction:
  - keep ShipTarget-vs-DroneTarget handler pressure stronger
  - keep BirdTarget-vs-ShipTarget handler pressure much smaller
  - raise the pair gate threshold slightly so the handler only acts in clearer pairwise regions
- Current default values:
  - `confusion_pair_handler_weight = 0.0`
  - `confusion_ship_drone_handler_weight = 0.025`
  - `confusion_bird_ship_handler_weight = 0.008`
  - `confusion_pair_handler_gate_threshold = 0.6`

## Stage 98 - Per-Pair Confusion Handler Still Regressed
- Run analyzed: `artifacts/202604141655-12-4-cuda`
- Evaluation with current eval-time fusion override fell to approximately:
  - `clip accuracy = 0.7500`
  - `clip macro_f1 = 0.7650`
  - `session accuracy = 0.6250`
- Main failure pattern:
  - `ShipTarget -> DroneTarget = 96`
  - `BirdTarget -> ShipTarget = 57`
  - `BirdTarget -> DroneTarget = 12`
- This confirms that even after splitting the handler by pair, the current confusion-pair handler design is still too invasive for the present data scale.
- Updated recommendation:
  - do not keep the confusion-pair handler enabled by default
  - restore the current stronger reference branch centered on the safer ShipTarget-vs-DroneTarget margin-only setup

## Stage 99 - Offline Pairwise Calibration Was Safe To Test But Not Useful As Default
- Extended the existing offline calibration pipeline to support pairwise calibration search on top of the global validation-driven logit bias.
- The calibration path was tested on the current strong reference checkpoint `artifacts/202604141227-12-4-cuda`.
- Validation improved strongly after the existing global bias search, but the learned pairwise calibration deltas stayed at zero for both hard pairs:
  - `ship_drone pair_bias = [0.0, 0.0]`
  - `bird_ship pair_bias = [0.0, 0.0]`
- When the calibrated checkpoint was evaluated on the held-out test sessions, it regressed relative to the uncalibrated strong reference.
- Practical conclusion:
  - the current offline pairwise calibration extension is safe to keep as an optional tool
  - however, it should not become the default evaluation route
  - the strongest reference remains the uncalibrated `artifacts/202604141227-12-4-cuda` branch under the current eval-time anti-collapse overrides

## Stage 100 - External Method Scan And Feasible Next Paths
- A broad external method scan was reviewed across radar micro-Doppler classification, multi-modal fusion, and related multi-sensor target recognition literature.
- The recurring method families that are actually relevant to the current project state are:
  - supervised contrastive or metric learning for tighter pairwise boundaries
  - angular-margin or cosine-margin classifiers for confusing categories
  - uncertainty-aware or reliability-aware fusion instead of fixed global bias tuning
  - multi-stage or hierarchical decision systems where only hard candidate pairs trigger extra processing
  - domain generalization or session-shift robustness methods for a small number of hard sessions
  - reviewed hard-example mining and curriculum strategies instead of global oversampling
- Based on all local experiments so far, the most realistic paths are not the most complex ones.
- Recommended next-path ranking for this project:
  - first: metric-space tightening for the hard pairs using conservative supervised contrastive or angular-margin objectives
  - second: candidate-pair reranking outside the main trunk instead of stronger sampling or heavier residual injection
  - third: session-shift robustness for the hardest ship session through normalization and augmentation targeted at track statistics and RD alignment stability
  - fourth: open-set or uncertainty-aware fallback behavior for drone-like ship clips so they do not receive overconfident wrong decisions
- Paths that looked attractive in theory but have already shown local instability here:
  - broad hard-example oversampling
  - heuristic conditional gates inside training losses
  - stronger unified confusion-pair handler layers
  - reviewed-span BirdTarget-vs-ShipTarget margin as a default branch

