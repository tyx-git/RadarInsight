# Architecture

## Current Direction
The project is now centered on a track-first real-data pipeline.

Raw paired Excel files under `pro/excel` are normalized into standardized TSV sessions under `pro/tsv`, reviewed there, and then copied into `data/tsv` for training.

## Main Pipeline
- `pro/excel`: raw paired `bz_*.xlsx` and `tck_*.xlsx` files
- `prepare-tsv`: converts Excel annotations and track points into a normalized TSV dataset
- `pro/tsv`: processed TSV sessions plus review manifests
- `data/tsv`: training-ready TSV dataset
- `train/evaluate/report/calibrate`: model workflow over standardized TSV data

## Model Direction
- Current primary route: track-first model
- RD branch remains an extension path for future datasets that actually provide RD or TD data
- Calibration remains part of the fine-grained decision workflow

## Design Rule
Data-source-specific parsing belongs in processors under `src/prodata`.
The training stack should consume standardized data only, not raw vendor exports.


## Track Backbone Options

The current real-data training route is `Excel -> TSV -> data/tsv -> track-only training`.
Two track backbones are now supported:

- `temporal`: the restored best validated TSV baseline using TCN + Transformer + pattern branches.
- `graph`: a graph-based track backbone that builds a temporal-similarity graph from each track clip and performs graph message passing before classification.

The default remains `temporal`. The `graph` backbone is an experimental branch built on top of the restored baseline.


## Dual-Modal Temporal Backbone Reinforcement

The current dual-modal route keeps the validated temporal track encoder as the default backbone, but now strengthens it with real multi-window fusion. The session-based dataset emits short, mid, and long track windows. The temporal encoder consumes these windows together, adds a lightweight time-by-feature 2D convolution branch, and fuses local, mid-scale, and long-range motion context before entering the cross-modal fusion stack.


## Bird-Surface Specialist Correction

The current dual-modal fusion route now includes a Bird-vs-Ship specialist head. It is trained only on BirdTarget and ShipTarget samples and supplies a small residual correction to the final logits. The goal is to directly stabilize the strongest remaining error boundary without replacing the main fusion architecture.


## DG-GCN Track Backbone

The default track backbone has now been switched from the reinforced temporal stack to a DG-GCN-style track encoder. The new encoder builds local, similarity, and dynamic relation graphs over each track clip, performs graph message passing, and then fuses graph-pooled context with global physical summary features. The goal is to improve robustness under session shift and to better separate BirdTarget, DroneTarget, and ShipTarget without depending only on heavier temporal convolutions.


## DG-GCN Rollback Decision

DG-GCN remains available as an experimental track backbone, but it is no longer the default route. The default track backbone is restored to the validated temporal encoder because the first DG-GCN dual-modal production run regressed badly under the current split.


## RD-First Fusion Shift

The current optimization focus has been moved from track-centric enhancement back to RD-first fusion. The fusion stack now biases reliability toward the RD branch, strengthens RD unimodal logit contribution, and adds Bird-specialized RD auxiliary heads to address the strongest remaining confusion boundary.


## RD Representation First for Bird Recovery

The project now keeps the RD-first direction but no longer uses an aggressive RD-dominant fusion bias. Bird-specific optimization is moved down into the RD representation stack through a Bird-aware spectral enhancement branch, while fusion is kept more conservative.


## Bird-Surface Prototype Constraint

The current RD-first route now adds a Bird-vs-Ship prototype-style loss on RD features. This moves Bird discrimination deeper into the RD embedding space so that BirdTarget is encouraged to form a more stable cluster instead of relying only on logit-level correction.

Bird-Surface Focus Reinforcement
- Training loader now boosts BirdTarget and ShipTarget sample probability via weighted sampling.
- RD feature optimization now combines prototype compactness with an additional Bird-vs-Ship triplet-style margin constraint.
- This branch remains under the stable temporal + dual-modal backbone, without reintroducing aggressive RD dominance.

Validation Strategy Adjustment
- The split policy now explicitly preserves Bird session coverage in both validation and test partitions.
- Bird-specific reinforcement is softened by default: lighter sampling boost and triplet constraint disabled unless explicitly re-enabled.

Track Router + RD Specialist
- Track branch now exposes a 3-way coarse router: ShipTarget / AircraftTarget / AmbiguousAirTarget.
- RD branch now exposes a dedicated Bird-vs-Drone specialist head on RD pooled features.
- Final logits receive light structured residuals from the track router and RD specialist instead of relying only on generic fusion outputs.


## Current Data Purification
The active training route now only uses paired TR/RD sessions under `data/sessions` generated from `pro/raw`. Excel-derived track-only data under `pro/excel`, `pro/tsv`, and `data/tsv` is retained for review and future auxiliary use, but does not participate in the current supervised dual-modal training route.

## Current Coarse Labels
The current coarse labels are `AircraftTarget`, `BirdTarget`, `DroneTarget`, and `ShipTarget`. File-name source labels are normalized as follows: `Bird -> BirdTarget`, `Rotary drone -> DroneTarget`, `Fixed-wing UAV -> DroneTarget`, `Helicopter -> AircraftTarget`, `Passenger ship -> ShipTarget`, `Speedboat -> ShipTarget`.
