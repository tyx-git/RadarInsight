"""Versioned paths and integrity helpers for the V26e long-horizon L1 OOF.

V26e changes only the L1 training horizon of V26c from 20 to 32 epochs.  The
fold assignment, feature caches, fold-train normalization, and V26c L2
epoch-15 outputs are reused read-only.  Unknown parents remain outside every
training and calibration population.
"""

from __future__ import annotations

from pathlib import Path


from radar_ln3 import known_gate_v26c_oof as _v26c


FOLD_COUNT = 5


ARTIFACT_ROOT = Path(
    "artifacts/RADAR-LN3-V26e-cq08-l1-epoch32-known-only-oof-calibration"
)

























def fold_directory(fold_id: int) -> Path:
    if not 0 <= int(fold_id) < FOLD_COUNT:
        raise ValueError(f"invalid V26e fold: {fold_id}")
    return ARTIFACT_ROOT / f"fold-{int(fold_id):02d}"


def l1_config_path(fold_id: int) -> Path:
    return Path(
        "config/RADAR-LN3/"
        f"RADAR-LN3-V26e-cq08-known-gate-oof-fold-{int(fold_id):02d}-seed43-config.json"
    )


def l1_data_contract_path(fold_id: int) -> Path:
    return Path(
        "data/processed/"
        f"RADAR-LN3-V26e-cq08-known-gate-oof-fold-{int(fold_id):02d}-data-contract.json"
    )


def l1_checkpoint_path(fold_id: int) -> Path:
    return fold_directory(fold_id) / "l1" / "epoch-32-model.pt"




def l1_training_summary_path(fold_id: int) -> Path:
    return fold_directory(fold_id) / "l1" / "training-summary.json"


def l1_prediction_path(fold_id: int) -> Path:
    return fold_directory(fold_id) / "l1-held-out-predictions.json"




# These are deliberately aliases to V26c's physical files.  V26e changes no
# feature values, normalization values, or L2 outputs.
l1_normalization_path = _v26c.l1_normalization_path
l1_train_feature_cache_path = _v26c.l1_train_feature_cache_path
l1_held_out_feature_cache_path = _v26c.l1_held_out_feature_cache_path
