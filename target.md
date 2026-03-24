# Project Target

## Project Background

This project focuses on real radar `.mat` data that has already been processed by the radar signal side. `TR` provides trajectory time-series information. `RD` files contain complex `DPL` data, which is converted into distance-Doppler representations for micro-motion learning.

The project does not directly connect to live radar hardware in the current stage. The stable main path is:

`pro/raw -> prepare-pro -> pro/sessions -> data/sessions -> train / evaluate / report`

The project is intended to run in a GPU training environment, with `cuda` as the main training target. The data responsibilities and engineering rules should remain unchanged across hardware upgrades.

## Core Goals

- Build a high-accuracy recognition framework for real radar data.
- Keep the design general enough for future datasets with `30k+` samples, `5` main classes, and `20+` subclasses.
- Continue improving `DPL -> RD` processing, target alignment, trajectory modeling, fusion, and training stability.
- Avoid overfitting the whole engineering design to the current small sample dataset.

## Current Main Technical Path

- Track branch: `Residual TCN + Transformer`
- RD branch: aligned local ROI + global context + `X3D-M` style 3D CNN
- Fusion: bidirectional `Cross-Attention`
- Data split: strict `session`-level `train / val / test`
- Reporting: real results only, written into `reports/real/`

## Directory Responsibilities

- `pro/raw/`: raw `.mat` input data
- `pro/sessions/`: processed intermediate session outputs
- `data/sessions/`: only dataset root used by training and evaluation
- `src/`: source code for preprocessing, models, training, and evaluation
- `docs/algorithms/`: learning and explanation documents
- `artifacts/`: checkpoints, histories, evaluation outputs
- `reports/real/`: real charts and plots
- `debug.md`: iteration log and problem tracking

## Coding Style

- Use `UTF-8` encoding for text files.
- Use `4` spaces for indentation.
- Keep type hints, readable names, and small focused functions.
- Fix root causes instead of adding temporary patches.
- Avoid meaningless comments and unnecessary complexity.
- New runtime settings should go into `config/system_config.yaml` and the related config class.
- Any architecture-level change must also update related documents and `debug.md`.

## Training And Experiment Rules

- Training commands are executed manually by the human operator.
- After code changes, do static checks first, then minimal forward validation, then manual training.
- Normal experiments should prefer `5` epochs unless there is a clear reason to go longer.
- Output directories must use the format `YYYYMMDDHHMM-epochs-batch-device`.
- Example: `202603232359-5-4-gpu`

## Git Save Rules

- Save Git only when model quality shows a real improvement.
- Improvement is judged by independent evaluation, mainly:
  - `Accuracy`
  - `Macro F1`
  - whether the confusion matrix becomes more reasonable
- Commit message format must be:
  - `Accuracy 0.7972 MacroF1 0.5644`
- Do not save a new best commit for noisy or partial changes without real generalization improvement.

## Debug Maintenance Rules

- Every meaningful architecture, data pipeline, or training-strategy change must update `debug.md`.
- Each record should contain at least:
  - current stage
  - current problem
  - solution
  - pros and cons
- `debug.md` is the project evolution history and should not skip key iterations.

## GPU Training Environment Rules

- Keep the current project structure unchanged while using a GPU-first training workflow.
- Keep `cpu` compatibility, but use `cuda` as the default main training target.
- In the GPU environment, validate first:
  - CUDA-enabled `torch`
  - larger batch size stability
  - mixed precision only if it improves throughput without hurting accuracy
- Even in the GPU environment, always validate the data path and evaluation path before longer training.

## Collaboration Principles

- Real environment only. No fake data, fake reports, or fake-related code.
- Optimization priority: real accuracy, generalization, interpretability, and maintainability.
- If any document conflicts with the real code path, update the document immediately.
- When the dataset grows, reuse the current main path first and then scale the training strategy and model size step by step.
