"""Contract smoke tests for the V26 Known-only/open-set boundary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from build.known_gate_v26 import (
    AUDIT_VALIDATION_CACHE,
    CONFIG,
    READINESS,
)
from radar_ln3.known_gate import (
    KNOWN_GATE_CLASS_NAMES,
    KNOWN_GATE_LABEL_TO_ID,
    KNOWN_LABELS,
    UNKNOWN_LABEL,
    build_known_gate_dataset,
    build_known_gate_model_for_config,
    collate_known_gate_sequences,
    load_known_gate_config,
)
from radar_ln3.known_gate_features import (
    COURSE_PREFIX_FEATURE_NAMES,
    V26_KNOWN_ONLY_OPEN_SET_BASELINE_VARIANT,
    load_known_gate_feature_cache,
)
from radar_ln3.known_gate_v26_audit import (
    V26OpenSetAuditDataset,
    collate_v26_open_set_audit,
)
from train.known_gate_v21 import _verify_contract, _verify_readiness


def _random_batch(evidence_dim: int) -> dict[str, torch.Tensor]:
    torch.manual_seed(26043)
    batch, length, frames = 2, 5, 3
    return {
        "tr_features": torch.randn(batch, length, 14),
        "tr_missing_mask": torch.zeros(batch, length, 11),
        "tr_dt_seconds": torch.tensor(
            [[0.0, 0.0, 1.0, 1.0, 2.0], [0.0, 0.0, 2.0, 1.0, 1.0]]
        ),
        "tr_dt_valid": torch.tensor(
            [[False, False, True, True, True], [False, False, True, True, True]]
        ),
        "tr_dt_reason": torch.tensor(
            [[1, 2, 3, 3, 3], [1, 2, 3, 3, 3]], dtype=torch.long
        ),
        "rd_frames": torch.randn(batch, length, frames, 2, 21, 384),
        "rd_frame_mask": torch.ones(batch, length, frames, dtype=torch.bool),
        "rd_quality": torch.rand(batch, length, 7),
        "tr_padding_mask": torch.zeros(batch, length, dtype=torch.bool),
        "known_gate_features": torch.rand(batch, length, evidence_dim),
        "known_gate_feature_valid": torch.ones(
            batch, length, evidence_dim, dtype=torch.bool
        ),
    }


def run(project_root: Path) -> dict[str, object]:
    root = project_root.resolve()
    config_path = root / CONFIG
    readiness_path = root / READINESS
    if not config_path.is_file() or not readiness_path.is_file():
        raise FileNotFoundError("V26 prepare artifacts are incomplete")
    config = load_known_gate_config(config_path)
    contract = _verify_contract(root, config)
    readiness = _verify_readiness(root, config_path, config, contract)
    if (
        config.variant != V26_KNOWN_ONLY_OPEN_SET_BASELINE_VARIANT
        or config.feature_names != COURSE_PREFIX_FEATURE_NAMES
        or config.model.evidence_dim != 1
        or config.train.seed != 43
        or UNKNOWN_LABEL not in config.forbidden_training_labels
        or UNKNOWN_LABEL in config.label_to_id
        or readiness.get("schema_version")
        != "RADAR-LN3-V26-known-gate-known-only-open-set-baseline-training-readiness-v1"
        or readiness.get("validation_used_for_readiness_gate") is not False
    ):
        raise AssertionError("V26 config/readiness does not bind strict Known-only semantics")
    readiness_view = _verify_readiness(root, config_path, config, contract)
    if readiness_view.get("initial_model_state_sha256") != readiness.get(
        "initial_model_state_sha256"
    ) or not str(readiness_view.get("initial_model_state_sha256", "")):
        raise AssertionError("V26 validated readiness did not preserve the initialization hash")
    train_dataset = build_known_gate_dataset(root, config, split_name="train")
    val_dataset = build_known_gate_dataset(root, config, split_name="val")
    if train_dataset.label_counts() != {
        KNOWN_GATE_CLASS_NAMES[0]: 953,
        KNOWN_GATE_CLASS_NAMES[1]: 33,
    } or val_dataset.label_counts() != {
        KNOWN_GATE_CLASS_NAMES[0]: 223,
        KNOWN_GATE_CLASS_NAMES[1]: 9,
    }:
        raise AssertionError("V26 training Dataset contains unexpected parent support")
    if train_dataset.base._tensor_cache.max_size != 1:
        raise AssertionError("V26 training RD tensor cache is not bounded to one archive")
    audit_val = V26OpenSetAuditDataset(
        root,
        config,
        split_name="val",
        cache_path=root / AUDIT_VALIDATION_CACHE,
    )
    if audit_val.label_counts().get(UNKNOWN_LABEL) != 15:
        raise AssertionError("V26 mixed validation audit support does not include the frozen Unknown parents")
    if audit_val.base._tensor_cache.max_size != 1:
        raise AssertionError("V26 audit RD tensor cache is not bounded to one archive")
    known_item = train_dataset[0]
    unknown_item = audit_val[
        next(index for index, parent_id in enumerate(audit_val.parent_ids) if audit_val.base._groups[parent_id].iloc[0]["label_name"] == UNKNOWN_LABEL)
    ]
    try:
        collate_known_gate_sequences([unknown_item])
    except ValueError:
        pass
    else:
        raise AssertionError("UnknownTarget crossed into the training collate")
    audit_batch = collate_v26_open_set_audit([known_item, unknown_item])
    if audit_batch["labels"].tolist()[1] != -1 or audit_batch["task_label_names"][1] != "UNKNOWN_OPEN_SET_TRUTH":
        raise AssertionError("V26 audit Unknown truth was mapped to a model class")
    treatment, baseline, pairing = build_known_gate_model_for_config(root, config)
    if (
        baseline is None
        or pairing.get("paired_baseline") is not True
        or pairing.get("new_fusion_columns_exactly_zero") is not True
    ):
        raise AssertionError("V26 fresh seed43 initialization is not paired to the baseline")
    tensors = _random_batch(config.model.evidence_dim)
    treatment.eval()
    baseline.eval()
    with torch.inference_mode():
        treatment_logits = treatment(**tensors)["sequence_logits"]
        baseline_batch = dict(tensors)
        baseline_batch.pop("known_gate_features")
        baseline_batch.pop("known_gate_feature_valid")
        baseline_logits = baseline(**baseline_batch)["sequence_logits"]
    if not torch.equal(treatment_logits, baseline_logits):
        raise AssertionError("V26 zero-initialized course column changed initial logits")
    for path in (root / AUDIT_VALIDATION_CACHE,):
        values, valid, metadata = load_known_gate_feature_cache(
            path,
            expected_sample_ids=(
                audit_val.base.frame["sample_id"].astype(str).tolist()
                if path.name == (root / AUDIT_VALIDATION_CACHE).name
                else []
            ),
            variant=config.variant,
        )
        if not values or not valid or metadata.get("feature_names") != list(COURSE_PREFIX_FEATURE_NAMES):
            raise AssertionError("V26 audit course cache is not aligned to the feature contract")
    return {
        "status": "passed",
        "schema_version": "RADAR-LN3-V26-known-gate-known-only-open-set-contract-smoke-v1",
        "known_train_parent_count": len(train_dataset),
        "known_validation_parent_count": len(val_dataset),
        "mixed_validation_parent_count": len(audit_val),
        "planned_mixed_test_parent_count": 244,
        "unknown_train_exposure": 0,
        "unknown_is_model_class": False,
        "unknown_audit_label_id": -1,
        "initial_logits_exactly_equal": True,
        "audit_cache_alignment": True,
        "rd_tensor_cache_size": 1,
        "test": "not_executed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the V26 Known-only/open-set contract.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    print(json.dumps(run(args.project_root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
