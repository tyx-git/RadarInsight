"""Focused contract Smoke for V18r stable8 plus density-weighted dynamics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from build.stable8_density_v18r import config_relative_path
from radar_ln3.data import DT_REASON_ZERO_DELTA
from radar_ln3.task_models import (
    V18R_STABLE8_DENSITY_PAIRED_PROTOCOL,
    V18R_STABLE8_DENSITY_SCHEMA_VERSION,
    build_paired_model_for_config,
    build_task_dataset,
    collate_task_sequences,
    load_task_config,
    sha256_file,
    task_training_objective,
)
from report.stable8_density_v18r_readiness import (
    OUTPUT as READINESS_OUTPUT,
    SCHEMA_VERSION as READINESS_SCHEMA,
)


EXPOSED_PARENT = "cq08|track|1010"


def _inputs(batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    return {
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


def run(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    readiness_path = root / READINESS_OUTPUT
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    if (
        readiness.get("status") != "passed"
        or readiness.get("schema_version") != READINESS_SCHEMA
        or readiness.get("paired_protocol")
        != V18R_STABLE8_DENSITY_PAIRED_PROTOCOL
        or readiness.get("test") != "未执行"
        or readiness.get("test_payload_access_count") != 0
        or readiness.get("deployment_eligible") is not False
    ):
        raise ValueError("V18r Smoke requires a passed sealed readiness report")
    configs = {
        arm: load_task_config(root / config_relative_path(arm))
        for arm in ("control", "treatment")
    }
    treatment_model, control_model, evidence = build_paired_model_for_config(
        root, configs["treatment"]
    )
    if control_model is None:
        raise RuntimeError("V18r paired builder did not return a Control model")
    dataset = build_task_dataset(
        root,
        configs["treatment"],
        split_name=configs["treatment"].data.train_split_name,
    )
    try:
        parent_index = dataset.parent_ids.index(EXPOSED_PARENT)
    except ValueError as error:
        raise ValueError(f"V18r exposed parent is missing: {EXPOSED_PARENT}") from error
    batch = collate_task_sequences(dataset.spec)([dataset[parent_index]])
    inputs = _inputs(batch)
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
    post_exposure = active & exposure_seen
    if not bool(post_exposure.any()):
        raise AssertionError("V18r real-parent Smoke has no density exposure")

    treatment_model.train()
    treatment_output = treatment_model(**inputs)
    weights = torch.tensor(
        configs["treatment"].train.class_weights, dtype=torch.float32
    )
    objective = task_training_objective(
        treatment_output,
        batch,
        configs["treatment"],
        class_weights=weights,
    )
    objective["loss"].backward()
    gradient_norm = 0.0
    for parameter in treatment_model.parameters():
        if parameter.grad is not None:
            gradient_norm += float(parameter.grad.detach().square().sum().item())
    gradient_norm = gradient_norm**0.5
    checks = {
        "schema_and_protocol_match": (
            configs["control"].schema_version
            == configs["treatment"].schema_version
            == V18R_STABLE8_DENSITY_SCHEMA_VERSION
            and configs["control"].train.paired_protocol
            == configs["treatment"].train.paired_protocol
            == V18R_STABLE8_DENSITY_PAIRED_PROTOCOL
        ),
        "control_reference_is_not_trainable": (
            configs["control"].train.allow_parameter_updates is False
        ),
        "treatment_is_trainable": (
            configs["treatment"].train.allow_parameter_updates is True
        ),
        "both_arms_share_stable8": (
            configs["control"].model.micro_doppler_dim
            == configs["treatment"].model.micro_doppler_dim
            == 8
        ),
        "reconstructed_source_state_exact": evidence[
            "shared_parameters_and_buffers_exactly_equal"
        ],
        "stable8_fusion_columns_exactly_zero": evidence[
            "stable8_fusion_columns_exactly_zero"
        ],
        "real_parent_pre_exposure_logits_exact": bool(
            pre_exposure.any()
            and torch.equal(
                control_logits[pre_exposure], treatment_logits[pre_exposure]
            )
        ),
        "real_parent_post_exposure_logits_differ": bool(
            not torch.equal(
                control_logits[post_exposure], treatment_logits[post_exposure]
            )
        ),
        "treatment_loss_is_finite": bool(torch.isfinite(objective["loss"]).item()),
        "treatment_gradient_is_finite_and_nonzero": (
            gradient_norm > 0.0 and torch.isfinite(torch.tensor(gradient_norm)).item()
        ),
        "readiness_binds_current_treatment_config": (
            dict(readiness["paired_configs"]["treatment"])["sha256"]
            == sha256_file(root / config_relative_path("treatment"))
        ),
        "test_payload_access_count_is_zero": True,
    }
    result = {
        "status": "passed" if all(checks.values()) else "failed",
        "schema_version": "RADAR-LN3-V18r-stable8-density-smoke-v1",
        "checks": checks,
        "source_v18_initial_treatment_state_sha256": evidence[
            "source_v18_initial_treatment_state_sha256"
        ],
        "real_exposed_parent": EXPOSED_PARENT,
        "post_exposure_maximum_absolute_logit_difference": float(
            (control_logits - treatment_logits)[post_exposure].abs().max().item()
        ),
        "treatment_gradient_norm": gradient_norm,
        "test": "未执行",
        "OOF": "未执行",
        "full_cascade": "未执行",
        "formal_training_by_smoke": "未执行",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "passed":
        raise RuntimeError(f"V18r focused Smoke failed: {checks}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke V18r stable8 plus density without formal training."
    )
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    run(args.project_root)


if __name__ == "__main__":
    main()
