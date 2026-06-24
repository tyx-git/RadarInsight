# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RadarInsight is a **Mixture-of-Experts (MoE) radar target classification system** for low-altitude airspace security. It classifies radar targets into 9 categories: Aircraft, Balloon, Bird, Car, Clutter, Drone, People, Ship, Unknown.

Primary metric: low-altitude drone/bird accuracy (goal ≥0.95). Selection metric: macro F1. System accuracy alone is misleading because CQ08 large-sample data dominates — always check per-class, by-source, and route per_route breakdowns.

## Commands

**Full pipeline:**
```bash
python -m src.core.train_main_pipeline --config config/config.json
```

**Individual training stages:**
```bash
python -m src.core.train_pretrain_track --config config/config.json   # Pretrain trajectory encoder
python -m src.core.train_router --config config/config.json           # Router
python -m src.core.train_expert1 --config config/config.json          # Expert1 (low-altitude, multimodal)
python -m src.core.train_expert2 --config config/config.json          # Expert2 (air)
python -m src.core.train_expert4 --config config/config.json          # Expert4 (ground)
python -m src.core.train_system --config config/config.json           # Full system integration
```

**Data preprocessing:**
```bash
python -m src.process.build_all_processed                              # Run all preprocessing
python -m src.process.build_manifests                                  # Build data manifests
python -m src.process.standardize_tracks                               # Standardize raw track data
python -m src.process.build_expert1_rd_segment_store                   # Build RD segment store
```

**Tests:**
```bash
python -m pytest tests/
python -m pytest tests/test_expert1_rd_segment_store_smoke.py          # Single test
```

## Architecture

### Expert Routing

`RadarClassificationSystem` (src/core/system.py) owns a shared `TrackRepresentationBuilder` (Transformer encoder + physical feature projector) and a `DeepGatingRouter` that dispatches to 4 experts:

| Route | Expert | Classes | Key file |
|---|---|---|---|
| `low_altitude_suspect` | Expert1 | Drone, Bird, Balloon, Clutter, Unknown | src/core/experts.py |
| `conventional_air` | Expert2 | Aircraft vs non-Aircraft (open-set) | src/core/experts.py |
| `high_altitude_filter` | Expert3 | Aircraft (rule-based, no NN) | src/core/experts.py |
| `ground_target` | Expert4 | People, Car, Ship | src/core/experts.py |

Router uses rule-based overrides: >5000m + >100m/s → air; <2m + <5m/s → ground. Gumbel-Softmax during training.

During inference, `_forward_reroute` evaluates only the routed expert (lazy evaluation via closures). Low-confidence rejects can reroute to another expert (up to 3 passes).

### Expert1 Dual-Path Design

Expert1 has two branches selected by RD availability:
- **Trajectory-only branch**: fused features → CosineMarginHead (ArcFace) → 5-class logits
- **TR+RD joint branch**: trajectory tokens + RD tokens → CrossModalFusionBlock (bidirectional cross-attention) → quality-aware joint head

RD features are auxiliary evidence bounded by `rd_weight` — they modify final logits as small corrections, never overriding trajectory judgments. Multiple auxiliary RD heads (bird, unknown, clutter, residual, unknown-clutter, clutter-morphology) provide gradient signal.

### Training Pipeline Stages

Stages run sequentially, each feeding checkpoints forward:
1. **Pretrain** track encoder (masked token modeling + contrastive learning)
2. **Train Router** (balanced sampling)
3. **Train Expert1** (stage1: trajectory-only → stage2: ramp in TR+RD fusion)
4. **Train Expert2** (open-set aircraft detection)
5. **Train Expert4** (ground target classification)
6. **Train System** (staged warmup: warmup → expert_warmup → joint)

### Data Pipeline

Raw data (gitignored in `OriData/`) → preprocessing (`src/process/`) → processed data in `data/processed/` with CSV indexes and manifests in `data/manifests/`.

Key datasets: `TorchTrackDataset` (src/core/data/torch_dataset.py) loads track CSVs, encodes sequences, extracts 17-dim physical features (src/core/physical.py), loads RD tensors (4-modal: amplitude_range, phase_range, amplitude_doppler, phase_doppler), supports RD segment store for windowed access.

Track schema: 25 columns defined in src/process/schema.py. Label vocabulary: 9 classes in src/core/data/labels.py.

## Configuration

Single master config: `config/config.json` loaded via `config/configmanager.py` into frozen dataclasses:

| Config class | File | JSON section |
|---|---|---|
| `MainPipelineConfig` | src/core/pipeline_config.py | `pipeline` |
| `TrackPretrainConfig` | src/core/pretrain_track_config.py | `pretrain_track` |
| `RouterTrainConfig` | src/core/router_train_config.py | `router` |
| `Expert1TrainConfig` | src/core/expert1_config.py | `expert1` |
| `ExpertOtherTrainConfig` | src/core/expert_other_config.py | `expert2`, `expert4` |
| `SystemTrainConfig` | src/core/system_config.py | `system` |

## Key Design Rules

- **Grouped splits**: Never split RD frames from the same track across train/test. CQ08 uses `cq08_grouped_split.json`; other sources use stratified splits.
- **Source-aware normalization**: Multi-source data (CQ08, excel, raw_tr_rd) requires source normalization and balanced sampling to prevent distribution bias.
- **No fake RD**: When RD is unavailable, use the trajectory-only branch — never feed synthetic/zero RD tensors through the fusion path.
- **Unknown reject**: Unknown rejection threshold is calibrated separately. Do not hard-reject Unknown before threshold calibration.
- **Checkpoints**: Training saves multiple checkpoints (fallback, best_final_macro_f1, best_joint_gain, last_full_val) with automatic post-training evaluation and comparison tables.

## Documentation

All design docs are in Chinese (中文). Key files at project root:
- `main.md` — Authoritative architecture document
- `core.md` — Core innovation points and competitive analysis
- `plan.md` — Development plan with progress tracking
- `target.md` — Expert1 V2 precision recovery plan
- `debug.md` — Architecture issues and training result tracking
