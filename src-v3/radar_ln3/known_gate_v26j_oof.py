"""Versioned paths and integrity helpers for V26j RD causal-GRU L1 OOF.

V26j is a single-variable continuation of V26e.  It preserves V26e's five
provenance-grouped folds, Known-only population, selected causal-course cache,
normalization, optimizer protocol, epoch-32 endpoint and V26c L2 reuse.  The
only model variable is the aggregation of frames *within one RD event*:
``masked_mean`` becomes ``causal_gru``.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from radar_ln3 import known_gate_v26c_oof as _v26c
from radar_ln3 import known_gate_v26e_oof as _v26e


FOLD_COUNT = 5
L1_TRAIN_EPOCHS = 32
L1_EPOCH = 32
L2_EPOCH = _v26c.L2_EPOCH
TRAINING_SEED_L1 = _v26c.TRAINING_SEED_L1
TRAINING_SEED_L2 = _v26c.TRAINING_SEED_L2
MINIMUM_EVIDENCE_EVENTS = _v26c.MINIMUM_EVIDENCE_EVENTS
L1_KNOWN_TARGET_ACTION_RECALL_FLOOR = _v26c.L1_KNOWN_TARGET_ACTION_RECALL_FLOOR
L2_CORRECT_ACTION_RECALL_FLOOR = _v26c.L2_CORRECT_ACTION_RECALL_FLOOR
CLUTTER_PRECISION_FLOOR = _v26c.CLUTTER_PRECISION_FLOOR
CLUTTER_RECALL_FLOOR = _v26c.CLUTTER_RECALL_FLOOR

V26C_CONTRACT = _v26c.CONTRACT
V26C_READINESS = _v26c.READINESS
V26E_CONTRACT = _v26e.CONTRACT
V26E_READINESS = _v26e.READINESS
V21E_ASSIGNMENT = _v26c.V21E_ASSIGNMENT
V21E_FOLD_CONTRACT = _v26c.V21E_FOLD_CONTRACT
INDEX = _v26e.INDEX
L1_SOURCE_CONFIG = _v26e.L1_SOURCE_CONFIG
L1_LABELS = _v26e.L1_LABELS
L2_LABELS = _v26e.L2_LABELS
BOUND_FILE_MAX_BYTES = _v26e.BOUND_FILE_MAX_BYTES

CONTRACT = Path(
    "data/processed/"
    "RADAR-LN3-V26j-cq08-l1-rd-causal-gru-epoch32-known-only-oof-calibration-contract.json"
)
READINESS = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V26j-l1-rd-causal-gru-epoch32-known-only-oof-calibration-readiness.json"
)
ARTIFACT_ROOT = Path(
    "artifacts/RADAR-LN3-V26j-cq08-l1-rd-causal-gru-epoch32-known-only-oof-calibration"
)
CONTRACT_SCHEMA = "RADAR-LN3-V26j-l1-rd-causal-gru-epoch32-known-only-oof-calibration-contract-v1"
READINESS_SCHEMA = "RADAR-LN3-V26j-l1-rd-causal-gru-epoch32-known-only-oof-calibration-readiness-v1"
L1_DATA_CONTRACT_SCHEMA = "RADAR-LN3-V26j-known-gate-fold-data-contract-v1"
L1_CONFIG_SCHEMA = "RADAR-LN3-V26j-OOF-L1"
L1_PREDICTION_SCHEMA = "RADAR-LN3-V26j-l1-rd-causal-gru-epoch32-known-only-oof-prediction-v1"
L1_CHECKPOINT_SCHEMA = "RADAR-LN3-V26j-l1-epoch32-rd-causal-gru-oof-checkpoint-v1"
L3_REPORT_SCHEMA = "RADAR-LN3-V26j-l1-rd-causal-gru-epoch32-known-only-cross-fitted-l3-report-v1"
L3_REPORT = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V26j-l1-rd-causal-gru-epoch32-known-only-cross-fitted-l3-calibration.json"
)
L3_REGISTRY = Path(
    "data/processed/"
    "RADAR-LN3-V26j-cq08-semantic-decision-layer-known-oof-calibrated-registry.json"
)
PARENT_DETAIL_CSV = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V26j-l1-rd-causal-gru-epoch32-known-only-cross-fitted-l3-parent-details.csv"
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sequence_sha256(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(4, byteorder="big", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(str(value).replace("\\", "/"))
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def relative_path(root: Path, path: str | Path) -> str:
    return resolve_path(root, path).relative_to(root.resolve()).as_posix()


def bound_path(root: Path, value: str | Path, description: str) -> Path:
    path = resolve_path(root, value)
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"V26j {description} escapes the project root: {value}") from exc
    return path


def require_bound_file(
    root: Path,
    value: str | Path,
    description: str,
    *,
    expected_sha256: str | None = None,
    max_bytes: int = BOUND_FILE_MAX_BYTES,
) -> Path:
    path = bound_path(root, value, description)
    if not path.is_file():
        raise FileNotFoundError(f"V26j {description} is missing: {path}")
    if path.stat().st_size > int(max_bytes):
        raise MemoryError(
            f"V26j {description} exceeds the bounded size: "
            f"{path.stat().st_size} > {int(max_bytes)} bytes"
        )
    if expected_sha256 is not None and sha256_file(path) != str(expected_sha256):
        raise ValueError(f"V26j {description} hash differs: {path}")
    return path


def write_new_or_identical(path: Path, payload: Mapping[str, Any]) -> str:
    text = json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"refusing to overwrite a different V26j artifact: {path}")
        return "reused_identical"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists() and temporary.read_text(encoding="utf-8") != text:
        raise FileExistsError(f"refusing to replace a different V26j temporary artifact: {temporary}")
    if not temporary.exists():
        temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)
    return "created"


def fold_directory(fold_id: int) -> Path:
    if not 0 <= int(fold_id) < FOLD_COUNT:
        raise ValueError(f"invalid V26j fold: {fold_id}")
    return ARTIFACT_ROOT / f"fold-{int(fold_id):02d}"


def l1_config_path(fold_id: int) -> Path:
    return Path(
        "config/RADAR-LN3/"
        f"RADAR-LN3-V26j-cq08-known-gate-oof-fold-{int(fold_id):02d}-seed43-epoch32-rd-causal-gru-config.json"
    )


def l1_data_contract_path(fold_id: int) -> Path:
    return Path(
        "data/processed/"
        f"RADAR-LN3-V26j-cq08-known-gate-oof-fold-{int(fold_id):02d}-rd-causal-gru-data-contract.json"
    )


def l1_checkpoint_path(fold_id: int) -> Path:
    return fold_directory(fold_id) / "l1" / "epoch-32-model.pt"


def l1_last_checkpoint_path(fold_id: int) -> Path:
    return fold_directory(fold_id) / "l1" / "last_model.pt"


def l1_training_summary_path(fold_id: int) -> Path:
    return fold_directory(fold_id) / "l1" / "training-summary.json"


def l1_prediction_path(fold_id: int) -> Path:
    return fold_directory(fold_id) / "l1-held-out-predictions.json"


def fold_record(contract: Mapping[str, Any], fold_id: int) -> dict[str, Any]:
    rows = [
        dict(row)
        for row in list(contract.get("folds") or ())
        if int(dict(row).get("fold_id", -1)) == int(fold_id)
    ]
    if len(rows) != 1:
        raise ValueError(f"V26j fold is unavailable or repeated: {fold_id}")
    row = rows[0]
    train_ids = sorted(str(value) for value in list(row.get("train_parent_ids") or ()))
    held_out_ids = sorted(str(value) for value in list(row.get("held_out_parent_ids") or ()))
    if (
        not train_ids
        or not held_out_ids
        or len(train_ids) != len(set(train_ids))
        or len(held_out_ids) != len(set(held_out_ids))
        or set(train_ids) & set(held_out_ids)
        or len(train_ids) + len(held_out_ids) != 986
        or sequence_sha256(train_ids) != row.get("train_parent_ids_sha256")
        or sequence_sha256(held_out_ids) != row.get("held_out_parent_ids_sha256")
    ):
        raise ValueError(f"V26j fold partition is invalid: {fold_id}")
    row["train_parent_ids"] = train_ids
    row["held_out_parent_ids"] = held_out_ids
    return row


def model_runtime_binding(model: Any) -> dict[str, Any]:
    """Return the non-negotiable actual RD aggregation binding for V26j."""

    config_mode = str(model.config.model.rd_frame_aggregation)
    encoder_mode = str(getattr(model.rd_encoder, "aggregation_mode", ""))
    temporal = getattr(model.rd_encoder, "temporal", None)
    temporal_parameter_count = (
        int(sum(parameter.numel() for parameter in temporal.parameters()))
        if temporal is not None
        else 0
    )
    state_keys = sorted(
        name for name in model.state_dict() if name.startswith("rd_encoder.temporal.")
    )
    result = {
        "rd_frame_aggregation_config": config_mode,
        "rd_frame_aggregation_runtime": encoder_mode,
        "rd_temporal_module": "GRUCell" if temporal is not None else None,
        "rd_temporal_parameter_count": temporal_parameter_count,
        "rd_temporal_state_keys": state_keys,
    }
    if (
        config_mode != "causal_gru"
        or encoder_mode != "causal_gru"
        or temporal is None
        or temporal_parameter_count <= 0
        or not state_keys
    ):
        raise ValueError("V26j runtime model is not an instantiated causal-GRU RD aggregator")
    return result


def build_v26j_l1_model(root: Path, config: Any) -> tuple[Any, dict[str, Any]]:
    """Build V26j with V26e-equivalent shared initialization.

    Direct construction is required because the generic paired-treatment
    builder deliberately only supports feature-appending treatments.  Simply
    assigning ``model.config`` after constructing a V26 masked-mean model is
    invalid: the RD encoder would remain masked-mean and no GRU parameters
    would exist.  We therefore construct the V26e-equivalent source model,
    construct the real causal-GRU model, then copy every shape-compatible
    shared state entry.  Only ``rd_encoder.temporal.*`` remains new.
    """

    from dataclasses import asdict

    from radar_ln3.known_gate import (
        KnownGateClassifier,
        build_known_gate_model_for_config,
        load_known_gate_config,
        seed_everything,
    )

    if (
        str(config.variant) != "v26j_rd_causal_gru"
        or str(config.model.rd_frame_aggregation) != "causal_gru"
        or int(config.train.seed) != 43
    ):
        raise ValueError("V26j runtime configuration is not the sealed causal-GRU treatment")
    source_path = require_bound_file(root, L1_SOURCE_CONFIG, "V26j masked-mean source config")
    source = load_known_gate_config(source_path)
    if (
        str(source.variant) != "v26_known_only_open_set_baseline"
        or str(source.model.rd_frame_aggregation) != "masked_mean"
        or int(source.train.seed) != int(config.train.seed)
        or int(source.train.dataloader_seed) != int(config.train.dataloader_seed)
    ):
        raise ValueError("V26j masked-mean source protocol differs")
    source_model_fields = asdict(source.model)
    treatment_model_fields = asdict(config.model)
    source_model_fields.pop("rd_frame_aggregation")
    treatment_model_fields.pop("rd_frame_aggregation")
    if source_model_fields != treatment_model_fields:
        raise ValueError("V26j model differs from V26e beyond RD aggregation")
    source_model, reference, source_pairing = build_known_gate_model_for_config(root, source)
    del reference
    if str(source_model.rd_encoder.aggregation_mode) != "masked_mean":
        raise AssertionError("V26j source initialization model is not masked_mean")
    # Seed the causal-GRU-only parameters deterministically.  All common state
    # is overwritten from the V26e-equivalent source below.
    seed_everything(int(config.train.seed))
    model = KnownGateClassifier(config)
    source_state = source_model.state_dict()
    treatment_state = model.state_dict()
    source_names = set(source_state)
    treatment_names = set(treatment_state)
    unique_names = sorted(treatment_names - source_names)
    if not unique_names or any(not name.startswith("rd_encoder.temporal.") for name in unique_names):
        raise ValueError("V26j model has an unexpected non-GRU structural difference")
    if source_names - treatment_names:
        raise ValueError("V26j model lost a V26e-equivalent shared state entry")
    copied: list[str] = []
    with __import__("torch").no_grad():
        for name, source_value in source_state.items():
            target = treatment_state[name]
            if target.shape != source_value.shape or target.dtype != source_value.dtype:
                raise ValueError(f"V26j shared state differs: {name}")
            target.copy_(source_value)
            copied.append(name)
    model.load_state_dict(treatment_state, strict=True)
    runtime = model_runtime_binding(model)
    return model, {
        "initialization": "V26e_equivalent_shared_state_plus_seeded_causal_gru",
        "source_initialization": dict(source_pairing),
        "source_config_path": relative_path(root, source_path),
        "source_config_sha256": sha256_file(source_path),
        "shared_state_entry_count": len(copied),
        "shared_state_entries_sha256": sequence_sha256(sorted(copied)),
        "new_state_entries": unique_names,
        "runtime": runtime,
    }


# All value tensors and normalizations are frozen V26c/V26e reuses.  Only the
# V26j config/checkpoint/prediction namespace is new.
l1_normalization_path = _v26c.l1_normalization_path
l1_train_feature_cache_path = _v26c.l1_train_feature_cache_path
l1_held_out_feature_cache_path = _v26c.l1_held_out_feature_cache_path
l1_source_feature_cache_path = _v26c.l1_source_feature_cache_path
l2_config_path = _v26c.l2_config_path
l2_data_contract_path = _v26c.l2_data_contract_path
l2_normalization_path = _v26c.l2_normalization_path
l2_checkpoint_path = _v26c.l2_checkpoint_path
l2_last_checkpoint_path = _v26c.l2_last_checkpoint_path
l2_training_summary_path = _v26c.l2_training_summary_path
l2_prediction_path = _v26c.l2_prediction_path
l2_train_physical_cache_path = _v26c.l2_train_physical_cache_path
l2_source_physical_cache_path = _v26c.l2_source_physical_cache_path
l2_held_out_physical_cache_path = _v26c.l2_held_out_physical_cache_path
l2_train_micro_cache_path = _v26c.l2_train_micro_cache_path
l2_source_micro_cache_path = _v26c.l2_source_micro_cache_path
l2_held_out_micro_cache_path = _v26c.l2_held_out_micro_cache_path
l2_source_mixed_physical_cache_path = _v26c.l2_source_mixed_physical_cache_path
l2_source_mixed_micro_cache_path = _v26c.l2_source_mixed_micro_cache_path
l2_held_out_mixed_physical_cache_path = _v26c.l2_held_out_mixed_physical_cache_path
l2_held_out_mixed_micro_cache_path = _v26c.l2_held_out_mixed_micro_cache_path

# Reuse the audited V26e/V26c source verifiers rather than weakening their
# transitive hash checks.  They only inspect frozen source artifacts.
validate_v26c_source_payload = _v26e.validate_v26c_source_payload
validate_frozen_v26c_sources = _v26e.validate_frozen_v26c_sources
selected_event_sample_ids = _v26e.selected_event_sample_ids
read_feature_archive = _v26e.read_feature_archive


__all__ = [name for name in globals() if not name.startswith("_")]
