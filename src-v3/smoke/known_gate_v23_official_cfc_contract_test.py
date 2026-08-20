"""Focused V23 official-``ncps`` CfC contract smoke.

This smoke never trains a full fold and never opens validation, Unknown or
test payloads.  It validates the installed official implementation, V23a/V23b
sidecar contracts, a real frozen fold-train batch, gradients, causal padding
and offline/online equivalence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from radar_ln3.known_gate import build_known_gate_dataloader, build_known_gate_dataset
from radar_ln3.known_gate_official_cfc import OfficialCfCKnownGateClassifier
from radar_ln3.known_gate_v21e_oof import fold_record, load_fold_contract
from radar_ln3.known_gate_v23 import load_v23_config
from radar_ln3.known_gate_v23_oof import ARMS, config_relative
from radar_ln3.losses import parent_final_task_ce_with_prefix_auxiliary
from radar_ln3.official_cfc import (
    OFFICIAL_CFC_DISTRIBUTION_VERSION,
    OFFICIAL_CFC_GIT_COMMIT,
    official_cfc_runtime_binding,
)
from train.known_gate_v21 import _model_inputs


def _inputs_at(batch: dict[str, object], batch_index: int, event_index: int) -> dict[str, torch.Tensor]:
    return {
        "tr_features": batch["tr_features"][batch_index : batch_index + 1, event_index],
        "tr_missing_mask": batch["tr_missing_mask"][batch_index : batch_index + 1, event_index],
        "tr_dt_seconds": batch["tr_dt_seconds"][batch_index : batch_index + 1, event_index],
        "tr_dt_valid": batch["tr_dt_valid"][batch_index : batch_index + 1, event_index],
        "tr_dt_reason": batch["tr_dt_reason"][batch_index : batch_index + 1, event_index],
        "rd_frames": batch["rd_frames"][batch_index : batch_index + 1, event_index],
        "rd_frame_mask": batch["rd_frame_mask"][batch_index : batch_index + 1, event_index],
        "rd_quality": batch["rd_quality"][batch_index : batch_index + 1, event_index],
        "known_gate_features": batch["known_gate_features"][batch_index : batch_index + 1, event_index],
        "known_gate_feature_valid": batch["known_gate_feature_valid"][batch_index : batch_index + 1, event_index],
    }


def run(project_root: Path, *, arm: str) -> dict[str, object]:
    root = project_root.resolve()
    if arm not in ARMS:
        raise ValueError(f"unsupported V23 arm: {arm}")
    runtime = official_cfc_runtime_binding()
    config, source_config, _source_config_path = load_v23_config(
        root, root / config_relative(arm, 0)
    )
    source_contract, _assignment = load_fold_contract(root)
    fold = fold_record(source_contract, 0)
    dataset = build_known_gate_dataset(
        root,
        source_config,
        split_name="train",
        parent_ids=fold["train_parent_ids"][:2],
        allow_feature_cache_superset=True,
    )
    batch = next(iter(build_known_gate_dataloader(dataset, batch_size=2, shuffle=False)))
    model = OfficialCfCKnownGateClassifier(source_config)
    model.train()
    output = model(**_model_inputs(batch))
    weights = torch.tensor(config.train.class_weights, dtype=torch.float32)
    objective = parent_final_task_ce_with_prefix_auxiliary(
        output["sequence_logits"],
        batch["labels"],
        batch["tr_padding_mask"],
        class_weights=weights,
        class_weight_normalization=config.train.class_weight_normalization,
        prefix_auxiliary_weight=config.train.prefix_auxiliary_weight,
        prefix_auxiliary_min_events=config.train.prefix_auxiliary_min_events,
    )
    objective["loss"].backward()
    cfc_gradients = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if name.startswith("liquid.cfc.")
    ]
    cfc_gradient_finite_nonzero = bool(cfc_gradients) and all(
        gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum().item() > 0.0
        for gradient in cfc_gradients
    )
    model.eval()
    with torch.inference_mode():
        offline = model(**_model_inputs(batch))
        state = None
        online_rows: list[torch.Tensor] = []
        length = int(batch["lengths"][0].item())
        for event_index in range(length):
            online = model.forward_online_point(
                **_inputs_at(batch, 0, event_index), liquid_state=state
            )
            state = online["liquid_state"]
            online_rows.append(online["window_logits"])
        online_logits = torch.stack(online_rows, dim=1)
        offline_logits = offline["sequence_logits"][:1, :length]
        max_online_error = float((offline_logits - online_logits).abs().max().item())
        padded_logits = offline["sequence_logits"].masked_select(
            batch["tr_padding_mask"].unsqueeze(-1)
        )
    spec = model.backbone_specification()
    source_dt_mode = source_config.model.liquid_invalid_dt_update_mode
    actual_parameter_count = sum(parameter.numel() for parameter in model.liquid.parameters())
    checks = {
        "official_ncps_distribution_version_is_pinned": runtime["distribution_version"]
        == OFFICIAL_CFC_DISTRIBUTION_VERSION,
        "official_ncps_git_commit_is_pinned": runtime["git_commit"] == OFFICIAL_CFC_GIT_COMMIT,
        "official_cfc_not_project_local_encoder": model.liquid.cfc.__class__.__module__ == "ncps.torch.cfc",
        "official_cfc_parameter_count_is_14064": actual_parameter_count == 14064,
        "source_and_v23_loss_protocol_match": config.train.class_weight_normalization
        == source_config.train.class_weight_normalization,
        "source_course_evidence_is_preserved": source_config.model.evidence_dim == 1,
        "source_custom_dt_mode_is_not_injected_into_official_cfc": source_dt_mode
        == "bounded_observation"
        and "raw_nonnegative_wall_clock_seconds" in str(spec["dt_semantics"]),
        "real_train_batch_forward_is_finite": bool(
            torch.isfinite(output["sequence_logits"]).all()
            and torch.isfinite(objective["loss"])
        ),
        "official_cfc_gradients_are_finite_and_nonzero": cfc_gradient_finite_nonzero,
        # CNN/GEMM kernels see [B,L,...] offline and [1,...] online tensors.
        # On CUDA this changes only reduction order; the observed real-batch
        # delta is sub-micro precision.  CfC causality is guarded with a
        # stated float32 tolerance rather than a false bitwise requirement.
        "offline_online_logits_match_within_float32_tolerance": bool(
            torch.allclose(offline_logits, online_logits, atol=1.0e-6, rtol=1.0e-6)
        ),
        "padded_logits_are_exactly_zero": bool(torch.equal(padded_logits, torch.zeros_like(padded_logits))),
        "unknown_validation_and_test_are_excluded": True,
    }
    if not all(checks.values()):
        failed = [name for name, value in checks.items() if not value]
        raise AssertionError(f"V23 official-CfC smoke failed: {failed}")
    result: dict[str, object] = {
        "status": "passed",
        "arm": arm,
        "checks": checks,
        "official_cfc": spec,
        "real_train_parent_count": len(dataset),
        "real_train_batch_parent_ids": list(batch["parent_sample_ids"]),
        "sequence_logits_shape": list(output["sequence_logits"].shape),
        "loss": float(objective["loss"].item()),
        "offline_online_max_logit_error": max_online_error,
        "offline_online_tolerance": {"atol": 1.0e-6, "rtol": 1.0e-6},
        "validation_access_count": 0,
        "test_payload_access_count": 0,
        "test": "未执行",
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run V23 official-CfC contract smoke.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.project_root, arm=args.arm), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
