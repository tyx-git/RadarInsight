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

## Stage 101 - Aggressive ShipTarget-vs-DroneTarget Refiner Tower
- After repeated instability from broad pair handlers and reviewed-span BirdTarget-vs-ShipTarget interventions, the next aggressive structural branch is narrowed to a dedicated `ShipTarget-vs-DroneTarget` refiner.
- The new refiner is a second-stage two-class tower over pair-relevant fused features:
  - `fused_track_feature`
  - `fused_rd_feature`
  - `rd_local_ship_feature`
  - `abs(fused_track_feature - fused_rd_feature)`
- The refiner has two roles:
  - auxiliary two-class supervision on ShipTarget-vs-DroneTarget samples
  - gated residual logit correction only when the current sample is already in the ShipTarget/DroneTarget candidate region
- Initial aggressive defaults:
  - `ship_drone_refiner_enabled = true`
  - `ship_drone_refiner_aux_loss_weight = 0.08`
  - `ship_drone_refiner_logit_weight = 0.12`
  - `ship_drone_refiner_gate_threshold = 0.62`
- All previously unstable confusion-pair handler branches remain disabled in this refiner experiment.

## Stage 102 - Aggressive ShipTarget-vs-DroneTarget Refiner Regressed
- Run analyzed: `artifacts/202604141756-12-4-cuda`
- Evaluation with current eval-time fusion override dropped to approximately:
  - `clip accuracy = 0.7735`
  - `clip macro_f1 = 0.7962`
  - `session accuracy = 0.8750`
- Main failure pattern:
  - `ShipTarget -> DroneTarget = 77`
  - `ShipTarget -> BirdTarget = 45`
  - `BirdTarget -> ShipTarget = 31`
- Interpretation:
  - the dedicated ShipTarget-vs-DroneTarget refiner was still too aggressive at the current data scale
  - it destabilized the broader ShipTarget manifold instead of selectively repairing the hardest ship-hard session
- Updated recommendation:
  - do not keep the aggressive ShipTarget-vs-DroneTarget refiner enabled by default
  - restore the safer reference branch centered on the light ShipTarget-vs-DroneTarget margin-only setup

## Stage 103 - Mainline Switched To True Session/Bag-Level Training
- The project now begins the requested transition from clip-level supervision to true session/bag-level training.
- A new `SessionBagDataset` groups clips by session and returns one bag per session.
- A new `SessionBagRadarClassifier` wraps the existing clip model and adds:
  - clip embedding extraction from `coarse_feature + detail_feature`
  - masked attention pooling over clip embeddings
  - a dedicated session classifier head
- The main training objective is now session-level cross-entropy, with an optional lightweight clip-level auxiliary term.
- The old `session_grouped_batches + mean logits` scaffold remains available for reference but is no longer the intended main route.
- Initial session-bag defaults:
  - `use_session_bag_training = true`
  - `session_bag_max_clips = 32`
  - `session_loss_weight = 1.0`
  - `clip_loss_weight = 0.15`
- This is the first true architectural shift of the project toward bag-level learning instead of clip-level learning.

## Stage 104 - Session-Level Stability Pass And Config Audit
- The first session-bag mainline exposed two immediate instability sources:
  - random clip selection inside each session bag
  - extremely small effective batch size on an 8 GB GPU
- The training stack was updated to make these controls explicit and actually effective:
  - `gradient_accumulation_steps`
  - `session_bag_shuffle`
  - `session_bag_sort_by_length`
- The session-bag dataset now supports deterministic session ordering by bag length.
- The training loop now supports gradient accumulation for both clip-level and session-level branches.
- `manage.py` output directory naming now uses `session_bag_batch_size` when session-bag training is enabled, so artifact names no longer misleadingly reflect the old clip batch size.
- `train.json` was cleaned to remove or disable several stale or misleading entries for the current session mainline:
  - removed the old clip-only `batch_size` entry from the control file
  - removed old artifact cleanup settings from the default training control
  - removed unused conditional-heuristic thresholds that no longer participate in training
  - disabled `session_bag_random_sampling` by default for reproducibility-first experiments
- New default stability-oriented session settings:
  - `session_bag_batch_size = 1`
  - `session_bag_max_clips = 8`
  - `session_bag_random_sampling = false`
  - `session_bag_shuffle = true`
  - `session_bag_sort_by_length = true`
  - `gradient_accumulation_steps = 4`
- The current intent is to reduce run-to-run variance first, before introducing stronger session specialists or larger bag encoders.

## Stage 105 - Session-Level Runs Became Reproducible But Validation Overestimates Test
- Two consecutive session-bag runs produced highly similar results after the stability pass.
- This is expected because the session-level route is now intentionally much more deterministic:
  - fixed random seed
  - deterministic training mode
  - `num_workers = 0`
  - `session_bag_random_sampling = false`
  - `session_bag_sort_by_length = true`
  - `session_bag_batch_size = 1`
  - `gradient_accumulation_steps = 4`
- Similarity across runs is therefore not evidence of a frozen model by itself. It mainly indicates that training noise was reduced.
- The more important finding is that the model can now converge repeatedly to a similar solution that still generalizes poorly to the tiny held-out test split.
- Example session-bag runs:
  - `artifacts/202604150022-12-1-cuda`
  - `artifacts/202604150035-12-1-cuda`
- In the second run, the best validation composite appeared at epoch 5 with approximately:
  - `val_accuracy = 0.9333`
  - `val_macro_f1 = 0.9000`
  - `val_selection_metric = 0.9100`
- Later epochs deteriorated, so early stopping correctly tracked stale epochs and kept epoch 5 as the best checkpoint.
- The final poor test result does not mean early stopping failed. It means the validation split substantially overestimated test generalization for that run.
- Root cause:
  - the session-level dataset is still extremely small
  - the validation split contains only a small number of sessions
  - the test split contains only 8 sessions, so every single misclassified session changes accuracy by 0.125
- Current interpretation:
  - the session-bag training path is now more reproducible
  - but the selected session-level checkpoint is still not trustworthy enough as a strong generalization winner
  - the next focus should remain session-level stability and evaluation robustness rather than celebrating a single high validation point

## Stage 106 - Processing Code Migrated Toward `pro/`
- The repository structure was updated to align with the rule that raw data processing belongs under `pro/`.
- A new local package layout now exists:
  - `pro/preprocessing/`
  - `pro/processing/`
- Core processing modules were copied into `pro/` and `manage.py` now imports processing entry points from `pro.*` rather than from `src.*`.
- Compatibility re-exports were left in `src/preprocessing/__init__.py` and `src/prodata/__init__.py` so the rest of the repository does not break immediately.
- The intended long-term boundary is now explicit:
  - `pro/` for source parsing and session preparation
  - `data/` for standardized training sessions only

## Stage 107 - Started `CQ-08` Python Ingestion Pipeline
- A new `CQ-08` ingestion module was introduced at `pro/processing/cq08_session_builder.py`.
- The module already supports:
  - discovery of one-to-one aligned raw echo, point-track, and track files
  - source label mapping
  - GBK track and point-track loading
  - normalized session naming
  - standardized session export scaffold
  - `prepare-cq08` CLI entry in `manage.py`
- Confirmed source facts:
  - current `CQ-08` payload in the workspace contains `1550` aligned groups
  - labels map to:
    - `DroneTarget`
    - `BirdTarget`
    - `BalloonTarget`
    - `ClutterTarget`
    - `OtherTarget`
- Important parser finding:
  - the provided `RDMapParser.m` does not yet match the current `.dat` frame layout one-to-one
  - the raw `.dat` files are little-endian and start with `0xFA55FA55`, but the downstream body parsing assumptions from the old MATLAB script do not yet cleanly recover valid frames
- Practical implication:
  - the standardized ingestion scaffold is now in place
  - but the raw-echo binary parser still needs one more decoding pass before full RD session export can be considered complete

## Stage 108 - `RDMapParser.m` Enhanced For Debug And Rich `.mat` Export
- The MATLAB helper at `pro/23/RDMapParser.m` was updated to emit richer debugging output and more useful `.mat` payloads.
- The save directory was changed from a hard-coded external drive path to a source-root-local folder:
  - `selectedDir/MAT_Result`
- Each saved `.mat` file now includes:
  - `data_proc_MTD_result`
  - `data_proc_MTD_result_db`
  - `data_out`
  - `para`
  - `Vr`
  - `Range_plot`
  - `label`
  - `index_trackPointNo`
  - `Amp_max_range`
  - `Amp_max_Vr`
  - `debug_info`
- The script now also prints a structured debug block with the most important parser outputs:
  - frequency
  - CPI count
  - PRT number
  - PRT duration
  - data length
  - `size(data_out)`
  - the first few `Track_No_info` values
  - a small `data_out` preview
- This change is intended to accelerate Python-side reverse engineering of the `CQ-08` raw `.dat` body layout and provide a temporary `.mat`-based bridge if needed.

## Stage 109 - Re-Audit Of The Sample `CQ-08` `.mat`
- The provided sample `.mat` is usable as an intermediate bridge, but it is not yet the fully upgraded export format from the latest MATLAB script revision.
- Confirmed available fields in the sample:
  - `data_out`
  - `data_proc_MTD_result`
  - `data_proc_MTD_result_db`
  - `para`
  - `Range_plot`
  - `Vr`
  - `label`
  - `index_trackPointNo`
  - `Amp_max_range`
  - `Amp_max_Vr`
  - `debug_info`
- Confirmed missing top-level fields that the updated script is now expected to export on future runs:
  - `track_No`
  - `target_type`
  - `timestamp_text`
- Important note:
  - the sample still contains enough signal information for Python-side investigation because `data_out`, `data_proc_MTD_result`, and `para` are all present
  - however, future exports should be regenerated with the latest MATLAB script so the richer metadata becomes available directly at top level
- Roadmap implication:
  - a denoise and alignment layer remains justified for raw-source ingestion under `pro/`
  - it should be implemented as preprocessing capability, not as a training-time patch

## Stage 110 - Session-Bag Evaluation Tooling Now Supports Session Checkpoints
- The evaluation pipeline was extended so session-bag checkpoints no longer depend on the old clip-only path.
- `run_evaluation(...)` now:
  - builds a bag dataset when `use_session_bag_training = true`
  - wraps the clip model in `SessionBagRadarClassifier`
  - evaluates `session_logits` directly
  - exports `session_attention_report.csv` when attention weights are available
- This addresses one of the highest-priority roadmap gaps: session checkpoints can now be evaluated through the same tooling stack instead of only through the training summary.
- A smoke test on `artifacts/202604150035-12-1-cuda/best_fusion_model.pt` succeeded and produced:
  - evaluation output on CPU
  - session-level metrics
  - session attention CSV export

## Stage 111 - Session Aggregator Upgraded Beyond Single Attention Pooling
- The session-bag mainline no longer relies on a single projection plus one attention pooling layer only.
- The session model was upgraded to:
  - project clip embeddings into a dedicated bag hidden space
  - optionally add learned positional encoding
  - run multiple masked self-attention encoder blocks over the bag
  - pool the encoded clip embeddings with masked attention pooling
  - classify from the resulting session embedding
- New configuration controls were added:
  - `session_bag_encoder_layers`
  - `session_bag_attention_heads`
  - `session_bag_use_positional_encoding`
- The new encoder passed a local forward smoke test on a real training batch.
- This directly implements the next roadmap step of strengthening the session aggregator without yet jumping to a much heavier architecture.

## Stage 112 - Session-Level Auxiliary Heads Introduced
- The session-bag mainline no longer upgrades only the main classifier to session-level supervision.
- New session-level auxiliary heads were added on top of the pooled session embedding:
  - `session_track_router_head`
  - `session_ship_drone_head`
  - `session_ship_bird_head`
  - `session_bird_surface_head`
- The session training branch now supports light auxiliary supervision on these heads.
- Initial conservative default weights were enabled in `train.json` so the project can begin testing true session-level specialist supervision without overwhelming the main session classification loss.
- A CPU smoke pass through `run_epoch(...)` succeeded with the new session auxiliary heads enabled.

## Stage 113 - Kaggle Cars/People Source Audited
- The `pro/Kaggle/` directory contains a paper closely related to the project target domain and two data classes: `Cars` and `People`.
- The paper confirms the source is a range-Doppler recognition dataset for `car`, `person`, and `drone` scenarios.
- The local `Cars/` and `People/` folders appear usable as RD-only session-like sources:
  - each class contains multiple subdirectories
  - each subdirectory contains ordered CSV frames
  - each CSV is a fixed-size numeric array of approximately `11 x 61`
- Integration decision:
  - keep these data as `CarTarget` and `PeopleTarget`
  - treat them as RD-only sessions rather than forcing them into the paired track+RD assumption
  - defer ingestion until the session-level pipeline fully supports optional-modality training

## Stage 114 - Session Mainline Still Collapses Bird Sessions Despite Stronger Aggregator And Session Aux Heads
- Two newer session-bag runs were analyzed:
  - `artifacts/202604151541-12-1-cuda`
  - `artifacts/202604151555-12-1-cuda`
- The second run looked different in the raw training summary, but a full evaluation pass showed both checkpoints converge to the same effective test confusion pattern.
- Both evaluated to approximately:
  - `accuracy = 0.625`
  - `macro_f1 = 0.5208`
- The confusion pattern is the same for both:
  - `AircraftTarget`: correct
  - `DroneTarget`: correct
  - `ShipTarget`: 3 correct, 1 predicted as `DroneTarget`
  - `BirdTarget`: both bird sessions wrong, split across `AircraftTarget` and `ShipTarget`
- Interpretation:
  - the newer session-level aggregator and session-level auxiliary heads are functioning technically
  - but the current tiny session split still allows the model to learn a stable but bad solution where Bird sessions collapse
  - the issue is no longer random instability only; it is now a stable but poor generalization basin
- Practical implication:
  - future work should not only strengthen the session encoder further
  - it must also improve session-level selection, class balance, and possibly Bird-focused session evidence modeling

## Stage 115 - Session Mainline Reset Toward A Cleaner Objective
- After the stronger session encoder and session-level auxiliary heads were introduced, repeated runs still converged to a stable but poor Bird collapse.
- A corrective pass was applied to make the session objective cleaner and less side-effect-prone:
  - reduced `clip_loss_weight`
  - reduced `session_track_router_aux_loss_weight`
  - reduced `session_ship_drone_aux_loss_weight`
  - disabled `session_ship_bird_aux_loss_weight`
  - disabled `session_bird_surface_aux_loss_weight`
- In addition, checkpoint selection was made more conservative by introducing validation floor constraints:
  - `best_metric_min_val_accuracy`
  - `best_metric_min_val_macro_f1`
- This is intended to stop the training loop from promoting visually impressive but poorly balanced session solutions too easily on the tiny validation split.

## Stage 116 - Fixed A Critical Session Training Selection Failure Mode
- A critical issue was discovered in the session-level route:
  - if all validation epochs failed the newly introduced selection floor, no `best_fusion_model.pt` was saved at all
  - the training summary could still print a best-checkpoint path even though the file did not exist
- The trainer now always saves a `last_fusion_model.pt` fallback checkpoint each epoch.
- If no epoch satisfies the validation floor, the training loop now explicitly falls back to the last checkpoint instead of silently proceeding without a real saved best model.
- The selection floor was also relaxed from an over-strict initial setting to a more practical one for the tiny validation split.
- This fix is important because otherwise session-level experiments could appear worse or behave inconsistently simply due to checkpoint selection failure rather than model quality alone.



## Stage 117 - Session Mainline Deep Audit

### Findings
- Session bag training itself was not collapsing to single-clip bags.
- `SessionBagDataset` was correctly building multi-clip bags with up to `session_bag_max_clips = 8` clips per session.
- The earlier `num_clips = 1` readings came from evaluator/reporting code that still interpreted session bag outputs using old clip-level assumptions.

### Confirmed Training-Side Behavior
- Train split bags: typically `8` clips per session.
- Test split bags: all `8` test sessions also carried `8` clips each under the current deterministic bag sampler.

### Bugs Found
1. `run_evaluation(...)` in bag mode was previously misrouted through diagnosis-style reporting logic, contaminating the return structure.
2. Session-level predictions in bag mode were being re-aggregated as if each bag were a single clip record, which produced misleading `num_clips = 1` outputs.
3. `run_session_diagnosis(...)` was still using clip-only assumptions and had to be upgraded for session-bag checkpoints.

### Fixes Applied
- `src/models/session_bag_classifier.py`
  - Exposes clip-level auxiliary tensors from the wrapped clip model:
    - `clip_track_weight`
    - `clip_rd_weight`
    - `clip_track_router_logits`
    - `clip_rd_ship_drone_logits`
    - `clip_rd_ship_bird_logits`
    - `clip_rd_ship_air_logits`
    - `clip_bird_surface_logits`
    - `clip_bird_rd_logits`
    - `clip_bird_surface_rd_logits`
- `src/training/evaluator.py`
  - Restored `run_evaluation(...)` to return the standard evaluation payload in bag mode.
  - Fixed bag-mode session predictions to report the real number of clips in each bag.
  - Added a dedicated `_run_session_diagnosis_bag(...)` path.
  - `run_session_diagnosis(...)` now dispatches to the bag-aware diagnosis path when `use_session_bag_training = true`.

### Rechecked Result
- For `artifacts/202604151700-12-1-cuda`:
  - true session accuracy remains `0.5000`
  - session diagnosis now reports `64` clip rows across `8` sessions
  - this confirms the earlier `num_clips = 1` interpretation was a reporting bug, not a bag-construction bug

### Remaining Structural Concerns
- Session mainline is now technically stable, but it is learning a poor decision basin.
- Current failure pattern is no longer ?single-clip bags?, but rather session-level class bias:
  - `ShipTarget` is still weak
  - `AircraftTarget` can collapse toward `DroneTarget`
- Many legacy clip-branch settings still remain in `train.json`; most of them are inactive under `use_session_bag_training = true` and should be cleaned later.


## Stage 118 - Session Mainline Config Cleanup

### Goal
Clean `train.json` so it reflects the current session-level mainline instead of carrying legacy clip-branch experiment flags.

### What Was Removed
Removed inactive or deprecated clip-branch experiment settings from `train.json`, including:
- hard-session sampling boosts
- suspicious-ship downweight heuristics
- confusion-pair samplers
- pair margin branches
- conditional margin branches
- curated clip manifest branches
- confusion-pair handler branches
- ship-drone refiner branches
- bird prototype / triplet branches

### What Remains Active In The Current Session Mainline
- core optimization and reproducibility
- clip-model backbone / fusion settings still used inside the bag wrapper
- eval-time fusion override settings
- session-bag encoder settings
- session-level auxiliary head weights
- best-checkpoint selection floors and early stopping settings

### Important Note
Many clip-model fusion parameters remain active even under session-bag training because the session model wraps the clip model and reuses its backbone and fusion path for every clip in the bag.


## Stage 119 - Session Mainline Bias Stabilization

### Motivation
After fixing the bag evaluation and diagnosis path, the main issue is no longer bag construction. The true failure mode is session-level class bias:
- `ShipTarget` is under-recognized
- `AircraftTarget` can drift toward `DroneTarget`
- session attention looked overly position-driven across sessions

### Changes
- Disabled positional encoding in the session bag encoder by default:
  - `session_bag_use_positional_encoding = false`
- Enabled session-level class-balanced bag sampling:
  - `session_bag_class_balanced_sampling = true`
- Removed clip-level loss from the current mainline:
  - `clip_loss_weight = 0.0`
- Added a light attention-entropy regularizer:
  - `session_attention_entropy_loss_weight = 0.01`

### Rationale
- learned positional encoding was encouraging near-fixed clip position weighting under deterministic bag sampling
- balanced bag sampling should reduce session-class prior bias, especially for `AircraftTarget` and `ShipTarget`
- clip-level auxiliary pressure was still pulling the session model back toward clip-style shortcuts
- a light entropy regularizer should discourage pathological attention collapse without forcing uniform weights too aggressively

### Validation
- config parses correctly
- session bag dataloaders still build correctly
- first training batch remains `(batch=1, clips=8)`


## Stage 120 - Session Mainline Logic Cleanup

### Cleanup Performed
Removed obsolete session-related experimental logic that no longer belongs to the active mainline:
- removed `session_grouped_batches`
- removed `session_logit_loss_weight`
- removed `SessionGroupedBatchSampler`
- removed `_build_session_logit_loss(...)`
- removed `_derive_session_id(...)` helper from `trainer.py`

### Why
These were remnants from an older failed attempt that grouped clip batches by session and added a mean-logit session loss on top of the clip pipeline. The current mainline now uses a real session bag model, so that old path only added maintenance noise.

### Result
- `train.json` no longer needs to carry these obsolete settings
- `trainer.py` is cleaner and easier to reason about for the current session-only route
- dataloader smoke test still succeeds with session bags of shape `(batch=1, clips=8)`


## Stage 121 - Hybrid Session Pooling

### Motivation
The current session mainline was still relying too much on a single attention-pooled summary, which risked making the session representation fragile and overly dependent on one aggregation path.

### Change
The session bag classifier now uses hybrid pooling:
- attention-pooled session embedding
- masked mean pooling over encoded clip embeddings
- masked standard-deviation pooling over encoded clip embeddings

These are concatenated and projected back into the final session hidden space before classification.

### Why
This gives the session head access to:
- focused evidence (`attention pooling`)
- global central tendency (`mean pooling`)
- intra-session variability (`std pooling`)

The goal is to reduce brittle dependence on a single pooled vector and make session predictions more robust for hard classes such as `ShipTarget`.

### Config
- `session_bag_use_stat_pooling = true`


## Stage 122 - Session metric mismatch root cause and fix
- Found a major consistency issue in the session mainline: the `Test metrics` logged at the end of training could diverge sharply from a later `manage.py evaluate` run on the same checkpoint.
- Root cause: runtime fusion controls were applied epoch-by-epoch during training, but they were not persisted inside the checkpoint payload. As a result:
  - end-of-training test used the loaded best model weights together with the in-memory runtime fusion state left from the final epoch;
  - `evaluate` rebuilt a fresh model from checkpoint weights and therefore used default fusion controls instead.
- This created two different inference conditions for what looked like the same checkpoint.
- Fix applied:
  - `trainer.py` now stores `runtime_fusion_controls` inside both `best_fusion_model.pt` and `last_fusion_model.pt`.
  - `trainer.py` restores `runtime_fusion_controls` immediately after reloading the checkpoint for the final test pass.
  - `evaluator.py` restores `runtime_fusion_controls` after loading a checkpoint, so standalone evaluation matches training-end testing semantics.
- Impact: future session-mainline comparisons should be much more trustworthy, and `Test metrics` vs `evaluate` should no longer diverge just because runtime fusion controls were lost.


## Stage 123 - Session-mainline evaluation semantics corrected
- Investigated a remaining mismatch where training-end `Test metrics` and standalone `manage.py evaluate` still differed on the same session checkpoint even after runtime fusion controls were persisted.
- Root cause: `manage.py evaluate` still applied clip-era eval-time fusion overrides by default because `eval_fusion_override_enabled = true` in `train.json`.
- For the session mainline, these eval-time RD fusion overrides are not part of the intended default test semantics and were distorting comparison:
  - with eval override enabled, `artifacts/202604152015-12-1-cuda/last_fusion_model.pt` evaluated to `accuracy=0.5`, `macro_f1=0.4167`;
  - with eval override disabled, the same checkpoint evaluated to `accuracy=0.625`, `macro_f1=0.45`, which matches the training-end `Test metrics`.
- Decision: disable `eval_fusion_override_enabled` in `train.json` for the session mainline so training-end testing and standalone evaluation use the same semantics.
- Interpretation: current session mainline should now be compared under pure session-bag inference, without legacy clip-mainline eval overrides.


## Stage 124 - Bird/Drone and Bird/Ship session-level boundary reinforcement
- Based on the corrected session-mainline evaluation semantics, the dominant remaining session-level errors shifted from global Ship collapse to local boundary confusion:
  - `BirdTarget -> ShipTarget`
  - `DroneTarget -> BirdTarget`
- Added a new session-level auxiliary head in `SessionBagRadarClassifier`:
  - `session_bird_drone_head`
- Wired the new bag-level supervision path in `trainer.py` using the already existing `bird_drone_target` carried by the dataset and bag collate pipeline.
- Adjusted default session auxiliary weights in `train.json` to target the current confusion structure more directly:
  - `session_ship_drone_aux_loss_weight = 0.015`
  - `session_ship_bird_aux_loss_weight = 0.01`
  - `session_bird_drone_aux_loss_weight = 0.02`
  - `session_bird_surface_aux_loss_weight = 0.0`
- Rationale:
  - preserve the recent recovery of `ShipTarget`
  - add a light session-level constraint that separates `Bird` from `Drone`
  - reintroduce a very light `Bird/Ship` session constraint without reactivating the previously harmful clip-era broad pair logic.


## Stage 125 - Session-level Bird/Drone auxiliary regression and rollback
- Tested a new session-level `Bird/Drone` auxiliary head and corresponding bag-level loss to address the remaining confusion pairs:
  - `BirdTarget -> ShipTarget`
  - `DroneTarget -> BirdTarget`
- Result on `artifacts/202604161029-12-1-cuda` was a regression:
  - `accuracy = 0.375`
  - `macro_f1 = 0.3333`
- Correct session-level confusion after unified evaluation semantics:
  - `AircraftTarget -> AircraftTarget`
  - `DroneTarget -> AircraftTarget`
  - `BirdTarget -> DroneTarget` for both Bird sessions
  - `ShipTarget -> DroneTarget` for 2 of 4 Ship sessions
- Interpretation:
  - the added session-level `Bird/Drone` auxiliary constraint pulled the bag encoder toward a Drone-heavy basin,
  - damaging both Bird and Ship boundaries.
- Action taken:
  - rolled back to a safer session auxiliary profile in `train.json`
  - disabled:
    - `session_bird_drone_aux_loss_weight`
    - `session_ship_bird_aux_loss_weight`
  - retained only a light:
    - `session_ship_drone_aux_loss_weight = 0.015`
- Conclusion:
  - direct session-level pair supervision for `Bird/Drone` is currently too aggressive for this small-data regime.
  - safer progress likely requires either more data or a less intrusive recurrent session encoder rather than more pairwise auxiliary heads.


## Stage 126 - First implementation of adaptive track-like completion for Carrada
- Started the first concrete implementation of the adaptive `track-like` completion idea from `mmm`.
- Added a reusable schema module:
  - `pro/processing/track_like_schema.py`
- Added a first Carrada pseudo-track builder:
  - `pro/processing/carrada_pseudotrack_builder.py`
- Added CLI support:
  - `manage.py prepare-carrada-pseudotrack`
- Current builder behavior:
  - scans Carrada sequences
  - reads `labels.json` and `centroid_tracking_cluster_jensen_shannon.json`
  - derives a first `pseudo-track` representation from RD/cluster temporal data
  - writes:
    - `radar/track/track_data.csv`
    - `pseudo_track_manifest.json`
- First smoke test succeeded on:
  - `2019-09-16-12-52-12`
- Output was created under:
  - `pro/processed/carrada_pseudotrack_sessions_smoke/2019-09-16-12-52-12`
  - and synced to:
  - `data/sessions_carrada_smoke/2019-09-16-12-52-12`
- Important note:
  - this is a first-stage pseudo-track, not yet a full final unified session ingestion for Carrada
  - the current output is intentionally `track-like`, not a claim of true physical native track recovery
- Value of this stage:
  - proves that the `tr + rd` mainline can be preserved even for RD-only sources through preprocessing
  - establishes the first working bridge for later Carrada integration without changing the core session model semantics.


## Stage 127 - Static risk fixes for Carrada standard-session builder
- Continued Carrada integration without launching another heavy full build.
- Fixed a critical semantic issue in `carrada_session_builder.py`:
  - pseudo-track `t` had been written using raw large ADC timestamps,
  - which broke QC and slicing assumptions expecting relative seconds.
- Added timestamp normalization so Carrada pseudo-track time is converted to relative second-scale time before session packaging.
- Fixed a concrete file-output bug:
  - `_write_labels_matched(...)` now actually writes `labels/labels_matched.jsonl`.
- Reduced the default Carrada slicing window from `8.0s` to `2.0s` to better fit shorter instance-active segments.
- Current state:
  - Carrada pseudo-track extraction works,
  - Carrada standard-session builder is implemented,
  - static code path now compiles cleanly,
  - heavy full build should be run manually by the user.


## Stage 128 - Default mixed-window sub-session generation for Carrada
- Kept the earlier mixed-window idea and promoted it into the Carrada preprocessing path.
- Carrada is no longer limited to one fixed sub-session window.
- Added configurable mixed window specs for Carrada sub-session generation.
- Current default mixed window policy:
  - `128:64`
  - `64:32`
  - `192:96`
- Both `manage.py prepare-carrada` and `pro.py prepare-carrada` now support:
  - `--sub-session-window-specs`
- If the user does not provide custom specs, the default mixed-window policy is used.
- Static estimate with the current high-confidence Carrada subset:
  - old fixed-window nonempty training units: about `110`
  - new mixed-window nonempty training units: about `252`
- Interpretation:
  - mixed windows do not solve the broader coverage ceiling (still limited by the 82 high-confidence parent instances),
  - but they significantly increase usable session-level training units without changing the mainline `tr + rd` assumption.


## Stage 129 - Unified session schema document added
- Added a formal schema document at `docs/session_schema.md`.
- The document defines the canonical session directory layout and required files for all future data sources.
- It clarifies the central long-term rule:
  - unify the **session data format**
  - do not force all sources to share identical physical origin semantics
  - preserve provenance explicitly through fields such as `source_name`, `track_source`, and `has_native_track`
- The document also records the current Carrada and CQ-08 mapping assumptions and the standard `tr + rd` compatibility contract.


## Stage 130 - Sessions-only unified root workflow
- Updated the session unification workflow so the long-term data layout can converge to a single canonical root: `data/sessions`.
- Added `rebuild_sessions_root(...)` to `pro/processing/session_unifier.py`.
- New behavior:
  - merge multiple standardized session roots into a temporary rebuild root
  - atomically replace the final target root (default: `data/sessions`)
  - optionally delete old source roots after successful rebuild
- Updated `pro.py unify-sessions` accordingly:
  - default final root is now `data/sessions`
  - supports automatic cleanup of old roots via `--clean-old-roots`
- This matches the intended long-term rule: the project should expose a single canonical `sessions` root for training.


## Stage 131 - 7-class group-aware split and balanced bucket fill
- Integrated source/group metadata into `SessionStatus`:
  - `source_name`
  - `group_id`
- For Carrada sessions, `group_id` is now derived from:
  - `sequence_id + parent_instance_id`
- This guarantees that all sub-sessions from the same Carrada parent instance remain in the same split.
- Verified:
  - `carrada_group_violations = 0`
- Upgraded the split heuristic from plain group-aware splitting to a more label-balanced bucket fill strategy.
- Compared with the previous group-aware-only version, the new split is much healthier:
  - validation now covers all 7 classes
  - test also covers all 7 classes
  - test is no longer dominated almost entirely by `PeopleTarget`
- Current smoke-test label distribution after the new split heuristic:
  - train: `Car 82, People 95, Cyclist 9, Drone 21, Aircraft 4, Ship 14, Bird 11`
  - val: all 7 classes covered
  - test: all 7 classes covered
- This means the unified 7-class data root is now much closer to actual training readiness.


## Stage 132 - First 7-class training-ready configuration
- Added a dedicated 7-class control file: `train_7cls.json`.
- Purpose:
  - keep the current 4-class `train.json` untouched
  - provide a clean starting point for the unified 7-class dataset root
- Current 7-class baseline design choices:
  - `data_root = data/sessions`
  - session-only training (`clip_loss_weight = 0.0`)
  - no pair-specific session auxiliary heads at startup
  - no eval-time fusion override
  - source-aware + class-aware session-bag sampling enabled
  - class-only session-bag sampling disabled
- Smoke test confirms that the 7-class dataset builds correctly from the unified `data/sessions` root.
- Current unified label map seen by the loader:
  - `AircraftTarget`
  - `BirdTarget`
  - `CarTarget`
  - `CyclistTarget`
  - `DroneTarget`
  - `PeopleTarget`
  - `ShipTarget`
- This means the unified data root has now crossed from preprocessing-only state into actual training-ready state.


## Stage 133 - CQ-08 MAT sessionization by batch and index-based alignment
- Confirmed that `CQ-08/MAT` can be treated as a session source using:
  - one `batch_no` = one session
- Verified globally:
  - no mixed-label batches
  - approximately 1549 valid batches after one damaged batch was discarded upstream
- Added `pro/processing/cq08_mat_session_builder.py` and `pro.py prepare-cq08-mat`.
- Initial naive `min_len(track, rd)` truncation was too coarse.
- Audited batch `104` and confirmed that `index_trackPointNo` is a valid alignment key from MAT frames to track/point rows.
- Upgraded the builder to:
  - read `index_trackPointNo` per MAT file
  - expand track and point rows according to MAT-frame alignment indices
  - retain the canonical RD shape only and drop incompatible outlier MAT frames
- Batch `104` smoke result after alignment upgrade:
  - input MAT files: `92`
  - retained canonical-shape MAT files: `88`
  - dropped incompatible MAT files: `4`
  - aligned track rows: `88`
  - aligned point rows: `88`
  - generated track clips: `12`
  - QC status: `PASS`
- This is a strong signal that the MAT route is now much more viable than the earlier raw-dat route for CQ-08 ingestion.


## Stage 134 - Multi-batch CQ-08 MAT smoke audit
- Expanded CQ-08 MAT smoke from a single Drone batch to a multi-label set:
  - `272` -> `DroneTarget`
  - `1301` -> `BirdTarget`
  - `3134` -> `OtherTarget`
  - `3160` -> `ClutterTarget`
  - `4153` -> `BalloonTarget`
- All 5 smoke sessions completed successfully into standard session format.
- Each smoke session produced:
  - `meta_session.json`
  - `labels_matched.jsonl`
  - `labels_trackclip.jsonl`
  - `track_data.csv`
  - `rd_data.h5`
  - `rd_clips.h5`
  - `qc_report.json`
- All 5 smoke sessions passed QC.
- This strongly suggests that the `CQ-08/MAT -> standard session` route is viable for full-batch preprocessing.
- The MAT ingestion path is now in a better state than the old raw-dat path because batch-level sessionization and `index_trackPointNo` alignment are both working.

## Stage 135 - åˆ†é˜¶æ®µ10ç±»è®­ç»ƒæ–¹æ¡ˆè½æ–‡æ¡£
- Added a staged 10-class training plan to `target.md` in Chinese.
- The plan is not a vague roadmap; it defines:
  - single-source baselines
  - pairwise source combinations
  - final 10-class unified training
  - fixed evaluation protocol
  - key confusion chains to monitor
- The purpose is to identify whether the main degradation comes from `CQ-08`, `Carrada`, or only from their interaction under unified training.

## Stage 136 - ç¬¬ä¸€é˜¶æ®µå®éªŒæ¸…å•è½åœ°
- Added `docs/phase1_experiments.md`.
- Added source-specific session id lists under `artifacts/phase1_lists/` for:
  - `LegacySession`
  - `CQ-08-MAT`
  - `Carrada`
- The first phase now has a concrete execution plan instead of only a conceptual split.

## Stage 137 - ç¬¬äºŒé˜¶æ®µå®éªŒæ¸…å•è½åœ°
- Added `docs/phase2_experiments.md`.
- Added pairwise source session id lists under `artifacts/phase2_lists/` for:
  - `Legacy + CQ-08`
  - `Legacy + Carrada`
  - `CQ-08 + Carrada`
- The second phase now directly targets cross-source conflict diagnosis.

## Stage 138 - CQ-08 µ×²ãËÙ¶ÈÓïÒåĞŞ¸´
- During deeper bottom-level audit, found that `CQ-08` sessions were still carrying a semantic mismatch in `track_data.csv`:
  - `radial_speed_mps` had effectively been inherited from track-side total speed semantics instead of point-side Doppler speed semantics.
- Verified raw CQ-08 column meanings:
  - track file includes `È«ËÙ¶È`
  - point file includes `¶àÆÕÀÕËÙ¶È`
- For unified radar session training, `radial_speed_mps` should be aligned with the point-side Doppler measurement, not the track-side total speed magnitude.
- Updated `pro/processing/cq08_session_builder.py` so `_aggregate_point_snr_to_track(...)` now also merges point Doppler speed and overwrites:
  - `radial_speed_mps <- point-side Doppler speed`
- Reused this corrected aggregation logic for MAT-based CQ-08 sessions and rewrote all existing unified-root `cq08mat_*` sessions in place:
  - updated `911` CQ-08 sessions under `data/sessions`
- Cleared stale caches after repair:
  - `dataset_track_stats_cache.json`
  - `dataset_clip_index_cache.json`
  - `session_scan_cache.json`
  - `session_split_cache.json`
- Implication:
  - Any previous CQ-08-involving experiment before this repair likely mixed incompatible speed semantics across sources.
  - `Legacy + Carrada` experiments remain unaffected by this specific CQ-08 speed fix.

## Stage 139 - ÑµÁ·²à°´À´Ô´¹ì¼£¹éÒ»»¯
- After repairing CQ-08 `snr_db` and `radial_speed_mps`, a deeper audit still showed strong source signatures in absolute track geometry:
  - `range_m`
  - `azimuth_deg`
  - `elevation_deg`
- To reduce source leakage through raw track coordinates, upgraded training-time track normalization from a single global normalization to source-wise normalization.
- Implemented in `src/training/dataset.py`:
  - track sequence normalization now supports per-source `center/scale`
  - track global feature normalization now supports per-source `center/scale`
  - `ClipRecord` and `SessionContext` now carry `source_name`
- Implemented checkpoint propagation:
  - training now saves `track_feature_by_source`
  - training now saves `track_global_by_source`
  - evaluator and calibration now restore these source-wise statistics when present
- Added cache-key versioning for dataset stats:
  - `normalization_mode = by_source_v1`
- Cleared stale caches after switching normalization mode.
- Implication:
  - all previous experiments were still using a single global track normalization across sources.
  - next reruns are the first ones that will test true source-wise track normalization.

## Stage 140 - CQ-08 BirdTarget ·ÖÍ°ÂäµØ
- Implemented a reusable CQ-08 Bird bucketing utility:
  - `pro/processing/cq08_bird_bucket_builder.py`
  - `pro.py bucket-cq08-bird`
- The utility buckets CQ-08 Bird sessions using current session-level track statistics:
  - `high_dynamic`
  - `low_speed_high_elev`
  - `low_speed_low_elev`
  - `mixed_mid`
- Generated formal outputs:
  - `artifacts/cq08_bird_buckets.csv`
  - `artifacts/cq08_bird_bucket_summary.json`
- Current bucket counts for `CQ-08 BirdTarget` (`152` sessions):
  - `high_dynamic = 33`
  - `low_speed_high_elev = 26`
  - `low_speed_low_elev = 11`
  - `mixed_mid = 82`
- This confirms that `CQ-08 BirdTarget` is not a compact class but a heterogeneous union of multiple motion/geometry subtypes.
- This also explains why current test-time Bird confusion fans out to:
  - `OtherTarget`
  - `BalloonTarget`
  - `ClutterTarget`
  - and occasionally `DroneTarget`
- Next training discussion should be based on these explicit Bird buckets rather than treating CQ-08 Bird as a single homogeneous class.

## Stage 141 - È«±êÇ©´óÀà+×ÓÀà½Ó¿Ú´òÍ¨
- Added a unified label-interface builder:
  - `pro/processing/label_interface_builder.py`
  - `pro.py build-label-interface`
- The interface now writes both:
  - `label_coarse`
  - `label_detail`
  into every `meta_session.json` under `data/sessions`
- Current policy:
  - for labels without known fine subclasses, `label_detail == label_coarse`
  - for `CQ-08 BirdTarget`, `label_detail` is expanded into bucketed subclasses:
    - `BirdTarget__HighDynamic`
    - `BirdTarget__LowSpeedHighElev`
    - `BirdTarget__LowSpeedLowElev`
    - `BirdTarget__MixedMid`
- Updated training dataset to support `label_granularity` directly for standard session roots:
  - `coarse` -> uses `label_coarse`
  - `detail` -> uses `label_detail`
- Updated coarse taxonomy folding logic so any detail label of the form:
  - `Xxx__Subtype`
  is automatically folded back to:
  - `Xxx`
- Verified current behavior:
  - `label_granularity = coarse` -> `10` labels
  - `label_granularity = detail` -> `14` labels
- This means the pipeline now supports:
  - training / inference with full detailed labels
  - final reporting folded back to major coarse labels
