"""Train-only readiness for V18r stable8 plus density-weighted dynamics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from build.stable8_density_v18r import (
    artifact_relative_path,
    config_relative_path,
)
from radar_ln3.data import DT_REASON_ZERO_DELTA
from radar_ln3.task_models import (
    BROADBAND_MICRO_DOPPLER_PAIRED_PROTOCOL,
    BROADBAND_MICRO_DOPPLER_PAIRED_DBB_SCHEMA_VERSION,
    V18R_SOURCE_CONTROL_CONFIG_PATH,
    V18R_SOURCE_TREATMENT_CONFIG_PATH,
    V18R_STABLE8_DENSITY_PAIRED_PROTOCOL,
    V18R_STABLE8_DENSITY_SCHEMA_VERSION,
    build_paired_model_for_config,
    build_task_dataloader,
    build_task_dataset,
    load_task_config,
    sha256_file,
    verify_task_data_contract,
)
from report.liquid_dynamics_v20_readiness import (
    DBB_LABELS,
    _cramers_v,
    _parent_summary,
    _with_density_ordinal,
)


OUTPUT = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V18r-stable8-density-weighted-training-readiness.json"
)
SCHEMA_VERSION = "RADAR-LN3-V18r-stable8-density-weighted-training-readiness-v1"
SOURCE_ARTIFACT = Path(
    "artifacts/RADAR-LN3-V18-cq08-liquid-dbb-specialist-rd-physical4-"
    "broadband-multiscale-micro-doppler8-stable-treatment-seed42-formal"
)
SOURCE_SUMMARY_SHA256 = (
    "64977b3386cb7f6065b783aa54d97bf11747808730fd8dad2f12dcdb95717c6c"
)
SOURCE_BEST_CHECKPOINT_SHA256 = (
    "ba66a11bc321c5b138a9c3444583ea69ca1e856131c3e8e26643c1c503a6ad56"
)
SOURCE_READINESS_SHA256 = (
    "89a1f25cb127bb107008a4b928139826d53fa37bbeb490bdd70ada7708dc39d7"
)
SOURCE_INITIAL_LOGITS_SHA256 = (
    "ce19b63987e2618d0a7b3a2c23b18fe01f74f554760df0c035c51e9634bac039"
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _directory_is_empty(path: Path) -> bool:
    return not path.exists() or next(path.iterdir(), None) is None


def _source_binding(root: Path) -> dict[str, Any]:
    control_path = root / V18R_SOURCE_CONTROL_CONFIG_PATH
    treatment_path = root / V18R_SOURCE_TREATMENT_CONFIG_PATH
    control = load_task_config(control_path)
    treatment = load_task_config(treatment_path)
    if (
        control.schema_version != BROADBAND_MICRO_DOPPLER_PAIRED_DBB_SCHEMA_VERSION
        or treatment.schema_version
        != BROADBAND_MICRO_DOPPLER_PAIRED_DBB_SCHEMA_VERSION
        or control.train.paired_protocol != BROADBAND_MICRO_DOPPLER_PAIRED_PROTOCOL
        or treatment.train.paired_protocol
        != BROADBAND_MICRO_DOPPLER_PAIRED_PROTOCOL
        or control.train.paired_arm != "control"
        or treatment.train.paired_arm != "treatment"
        or control.model.micro_doppler_dim != 0
        or treatment.model.micro_doppler_dim != 8
    ):
        raise ValueError("V18r source configs are not the frozen V18 stable8 pair")
    summary_path = root / SOURCE_ARTIFACT / "training-summary.json"
    best_path = root / SOURCE_ARTIFACT / "best_model.pt"
    last_path = root / SOURCE_ARTIFACT / "last_model.pt"
    epoch_dir = root / SOURCE_ARTIFACT / "epochs"
    summary = _read_json(summary_path)
    readiness = dict(summary.get("v18_training_readiness") or {})
    readiness_path = Path(str(readiness.get("path", "")))
    history = [dict(row) for row in summary.get("history") or []]
    parent_order_hashes = [str(row.get("parent_order_sha256", "")) for row in history]
    if (
        sha256_file(summary_path) != SOURCE_SUMMARY_SHA256
        or sha256_file(best_path) != SOURCE_BEST_CHECKPOINT_SHA256
        or summary.get("status") != "completed"
        or summary.get("task_name") != "dbb_specialist"
        or summary.get("stop_reason") != "epoch_limit"
        or int(summary.get("best_epoch", -1)) != 16
        or len(history) != 20
        or len(set(parent_order_hashes)) != 20
        or any(len(value) != 64 for value in parent_order_hashes)
        or summary.get("test_evaluation") != "not_executed"
        or summary.get("full_cascade_evaluation") != "not_executed"
        or summary.get("oof_rejection_calibration") != "not_executed"
        or summary.get("stable_evaluation") != "not_executed"
        or summary.get("p95_latency") != "not_executed"
        or summary.get("deployment_eligible") is not False
        or readiness.get("status") != "passed"
        or readiness.get("validation_used_for_readiness_gate") is not False
        or readiness.get("test") != "未执行"
        or not readiness_path.is_file()
        or sha256_file(readiness_path) != SOURCE_READINESS_SHA256
        or not last_path.is_file()
        or any(
            not (epoch_dir / f"epoch-{epoch:03d}.pt").is_file()
            for epoch in range(1, 21)
        )
    ):
        raise ValueError("frozen V18 stable8 source is incomplete, changed or unsealed")
    initialization = dict(summary.get("paired_initialization") or {})
    if (
        initialization.get("initial_logits_sha256")
        != SOURCE_INITIAL_LOGITS_SHA256
        or initialization.get("treatment_new_fusion_columns_exactly_zero") is not True
    ):
        raise ValueError("frozen V18 stable8 source initialization changed")
    return {
        "role": "read_only_frozen_control",
        "config": str(treatment_path.resolve()),
        "config_sha256": sha256_file(treatment_path),
        "source_pair_control_config": str(control_path.resolve()),
        "source_pair_control_config_sha256": sha256_file(control_path),
        "summary": str(summary_path.resolve()),
        "summary_sha256": SOURCE_SUMMARY_SHA256,
        "best_checkpoint": str(best_path.resolve()),
        "best_checkpoint_sha256": SOURCE_BEST_CHECKPOINT_SHA256,
        "best_epoch": 16,
        "readiness": str(readiness_path.resolve()),
        "readiness_sha256": SOURCE_READINESS_SHA256,
        "initial_logits_sha256": SOURCE_INITIAL_LOGITS_SHA256,
        "epoch_parent_order_hashes": parent_order_hashes,
        "validation_metrics_used_for_v18r_readiness_gate": False,
        "control_retrained": False,
    }


def _initialization_probe(
    root: Path,
    control_config: Any,
    treatment_config: Any,
) -> dict[str, Any]:
    model, counterpart, evidence = build_paired_model_for_config(
        root, treatment_config
    )
    if counterpart is None:
        raise RuntimeError("V18r paired builder did not return the Control model")
    treatment_model = model
    control_model = counterpart
    dataset = build_task_dataset(
        root, treatment_config, split_name=treatment_config.data.train_split_name
    )
    loader = build_task_dataloader(
        dataset,
        batch_size=min(treatment_config.train.batch_size, len(dataset)),
        shuffle=False,
    )
    batch = next(iter(loader))
    inputs = {
        "tr_features": batch["tr_features"],
        "tr_missing_mask": batch["tr_missing_mask"],
        "tr_dt_seconds": batch["tr_dt_seconds"],
        "tr_dt_valid": batch["tr_dt_valid"],
        "tr_dt_reason": batch["tr_dt_reason"],
        "rd_frames": batch["rd_frames"],
        "rd_frame_mask": batch["rd_frame_mask"],
        "rd_quality": batch["rd_quality"],
        "tr_padding_mask": batch["tr_padding_mask"],
        "rd_physical_features": batch["rd_physical_features"],
        "rd_physical_valid": batch["rd_physical_valid"],
        "micro_doppler_features": batch["micro_doppler_features"],
        "micro_doppler_valid": batch["micro_doppler_valid"],
    }
    control_model.eval()
    treatment_model.eval()
    with torch.inference_mode():
        control_logits = control_model(**inputs)["sequence_logits"]
        treatment_logits = treatment_model(**inputs)["sequence_logits"]
    active = ~batch["tr_padding_mask"]
    exposure_seen = torch.cumsum(
        (batch["tr_dt_reason"] == DT_REASON_ZERO_DELTA).to(torch.int64), dim=1
    ).gt(0)
    pre_exposure = active & (~exposure_seen)
    difference = (control_logits - treatment_logits).abs()
    control_hash = hashlib.sha256(
        control_logits.detach().cpu().contiguous().numpy().tobytes(order="C")
    ).hexdigest()
    return {
        "train_parent_count": len(dataset),
        "first_batch_parent_ids": list(batch["parent_sample_ids"]),
        "source_v18_initial_logits_sha256_reproduced": (
            control_hash == SOURCE_INITIAL_LOGITS_SHA256
        ),
        "control_initial_logits_sha256": control_hash,
        "pre_exposure_logits_exact": bool(
            pre_exposure.any()
            and torch.equal(
                control_logits[pre_exposure], treatment_logits[pre_exposure]
            )
        ),
        "pre_exposure_maximum_absolute_difference": float(
            difference[pre_exposure].max().item()
        ),
        "initial_treatment_exposed_event_count": int(
            (active & exposure_seen).sum().item()
        ),
        "full_initial_logits_exact_diagnostic": bool(
            torch.equal(control_logits, treatment_logits)
        ),
        "paired_initialization": evidence,
    }


def _density_audit(root: Path, control_config: Any) -> dict[str, Any]:
    index = pd.read_csv(root / control_config.data.index_path, low_memory=False)
    train = index[
        index["split_name"].astype(str).eq(control_config.data.train_split_name)
        & index["label_name"].astype(str).isin(DBB_LABELS)
    ].copy()
    train = _with_density_ordinal(train)
    return {
        "event_count": int(len(train)),
        "parent_count": int(train["parent_sample_id"].nunique()),
        "density_bucket_counts": {
            str(key): int(value)
            for key, value in train["density_bucket"].value_counts().sort_index().items()
        },
        "density_bucket_label_cramers_v": _cramers_v(
            train["label_name"].astype(str).to_numpy(),
            train["density_bucket"].astype(str).to_numpy(),
        ),
        "by_label": _parent_summary(train),
        "statistics_are_report_only_not_model_inputs": True,
    }


def run(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    config_paths = {
        arm: root / config_relative_path(arm) for arm in ("control", "treatment")
    }
    configs = {arm: load_task_config(path) for arm, path in config_paths.items()}
    control = configs["control"]
    treatment = configs["treatment"]
    if (
        control.schema_version != V18R_STABLE8_DENSITY_SCHEMA_VERSION
        or treatment.schema_version != V18R_STABLE8_DENSITY_SCHEMA_VERSION
        or control.train.paired_protocol
        != V18R_STABLE8_DENSITY_PAIRED_PROTOCOL
        or treatment.train.paired_protocol
        != V18R_STABLE8_DENSITY_PAIRED_PROTOCOL
        or control.train.paired_arm != "control"
        or treatment.train.paired_arm != "treatment"
        or control.model.liquid_invalid_dt_update_mode != "bounded_observation"
        or treatment.model.liquid_invalid_dt_update_mode != "density_weighted"
        or control.model.micro_doppler_dim != 8
        or treatment.model.micro_doppler_dim != 8
        or control.train.allow_parameter_updates is not False
        or treatment.train.allow_parameter_updates is not True
    ):
        raise ValueError("V18r readiness loaded an invalid config pair")
    control_contract = verify_task_data_contract(root, control)
    treatment_contract = verify_task_data_contract(root, treatment)
    source = _source_binding(root)
    initialization = _initialization_probe(root, control, treatment)
    density = _density_audit(root, control)
    checks = {
        "paired_configs_use_v18r_protocol": True,
        "both_arms_share_v18_stable8_data_contract": (
            control.data == treatment.data
            and control_contract["hashes"] == treatment_contract["hashes"]
        ),
        "both_arms_use_rd_physical4_and_stable8": (
            control.model.rd_physical_dim == treatment.model.rd_physical_dim == 4
            and control.model.micro_doppler_dim
            == treatment.model.micro_doppler_dim
            == 8
        ),
        "only_dynamics_mode_differs_in_model_contract": (
            control.model.__class__(
                **{
                    **control.model.__dict__,
                    "liquid_invalid_dt_update_mode": "density_weighted",
                }
            )
            == treatment.model
        ),
        "control_reference_is_not_trainable": (
            control.train.allow_parameter_updates is False
        ),
        "treatment_is_trainable": treatment.train.allow_parameter_updates is True,
        "control_reference_artifact_is_empty": _directory_is_empty(
            root / artifact_relative_path("control")
        ),
        "treatment_artifact_is_empty_before_formal_training": _directory_is_empty(
            root / artifact_relative_path("treatment")
        ),
        "frozen_v18_source_is_sha_bound_and_unsealed_metrics_are_not_a_gate": (
            source["summary_sha256"] == SOURCE_SUMMARY_SHA256
            and source["best_checkpoint_sha256"]
            == SOURCE_BEST_CHECKPOINT_SHA256
            and source["validation_metrics_used_for_v18r_readiness_gate"] is False
        ),
        "source_v18_initial_state_reconstructed_exactly": (
            initialization["paired_initialization"]
            ["shared_parameters_and_buffers_exactly_equal"]
            is True
            and initialization["paired_initialization"]
            ["stable8_fusion_columns_exactly_zero"]
            is True
        ),
        "source_v18_initial_logits_reproduced": initialization[
            "source_v18_initial_logits_sha256_reproduced"
        ],
        "pre_density_exposure_logits_are_exact": initialization[
            "pre_exposure_logits_exact"
        ],
        "train_parent_count_is_953": initialization["train_parent_count"] == 953,
        "density_audit_uses_train_only_953_parents": density["parent_count"] == 953,
        "validation_is_report_only_not_readiness_gate": True,
        "test_payload_access_count_is_zero": True,
    }
    passed = all(checks.values())
    result = {
        "status": "passed" if passed else "blocked",
        "schema_version": SCHEMA_VERSION,
        "paired_protocol": V18R_STABLE8_DENSITY_PAIRED_PROTOCOL,
        "paired_configs": {
            arm: {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
            }
            for arm, path in config_paths.items()
        },
        "contract_hashes": control_contract["hashes"],
        "checks": checks,
        "frozen_v18_control": source,
        "initialization_probe": initialization,
        "train_only_density_audit": density,
        "validation_used_for_readiness_gate": False,
        "test_payload_access_count": 0,
        "formal_training_at_report_generation": "未执行",
        "test": "未执行",
        "OOF": "未执行",
        "full_cascade": "未执行",
        "formal_rejection": "未执行",
        "stable": "未执行",
        "P95": "未执行",
        "deployment_eligible": False,
    }
    output = root / OUTPUT
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if output.exists() and output.read_text(encoding="utf-8") != text:
        raise FileExistsError(f"refusing to overwrite a different V18r readiness: {output}")
    output.write_text(text, encoding="utf-8")
    print(json.dumps({**result, "report_path": str(output)}, ensure_ascii=False, indent=2))
    if not passed:
        raise RuntimeError(f"V18r train-only readiness failed: {checks}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit V18r stable8 plus density train-only readiness."
    )
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    run(args.project_root)


if __name__ == "__main__":
    main()
