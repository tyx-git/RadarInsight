# Artifacts

This directory stores local training outputs such as checkpoints, histories, manifests, evaluation reports, and calibrated artifacts.

Recommended structure:

- `artifacts/<run_name>` for normal training runs
- `artifacts/calibrated/<run_name>-calibrated` for calibrated checkpoints

It is intentionally ignored by Git.
