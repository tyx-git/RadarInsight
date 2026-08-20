"""Focused V22b OOF contract smoke without long training or held-out access."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from radar_ln3.known_gate import build_known_gate_dataloader, build_known_gate_dataset, build_known_gate_model_for_config
from radar_ln3.known_gate_v22b import build_v22b_model
from radar_ln3.known_gate_v22b_oof import FOLD_COUNT, V22B_OOF_CONFIG_SCHEMA, V22B_OOF_READINESS, V22B_OOF_READINESS_SCHEMA, config_relative
from train.known_gate_v21 import _model_inputs
from train.known_gate_v21 import _state_dict_sha256
from train.known_gate_v22b_oof import _prepared


def run(project_root: Path) -> dict[str, object]:
    root = project_root.resolve()
    readiness = json.loads((root / V22B_OOF_READINESS).read_text(encoding="utf-8"))
    if readiness.get("status") != "passed" or readiness.get("schema_version") != V22B_OOF_READINESS_SCHEMA:
        raise AssertionError("V22b OOF readiness failed")
    fold_checks: list[dict[str, object]] = []
    for fold_id in range(FOLD_COUNT):
        source_config, fold, _, config_path, _ = _prepared(root, fold_id)
        protocol = json.loads(config_path.read_text(encoding="utf-8"))
        if protocol.get("schema_version") != V22B_OOF_CONFIG_SCHEMA:
            raise AssertionError("V22b OOF config schema differs")
        backbone, baseline_reference, _ = build_known_gate_model_for_config(root, source_config)
        if baseline_reference is None:
            raise AssertionError("V22b OOF source lacks paired reference")
        model = build_v22b_model(backbone)
        if not torch.equal(
            model.late_residual_weight,
            torch.zeros_like(model.late_residual_weight),
        ):
            raise AssertionError(f"V22b fold {fold_id} late weight is not zero")
        dataset = build_known_gate_dataset(
            root,
            source_config,
            split_name="train",
            parent_ids=fold["train_parent_ids"][:2],
            allow_feature_cache_superset=True,
        )
        batch = next(iter(build_known_gate_dataloader(dataset, batch_size=2, shuffle=False)))
        model.eval()
        with torch.inference_mode():
            treatment = model(**_model_inputs(batch))
            control = backbone(**_model_inputs(batch))
        if not torch.equal(treatment["sequence_logits"], control["sequence_logits"]):
            raise AssertionError(f"V22b fold {fold_id} zero-init logits differ")
        control_prediction_path = root / (
            "artifacts/RADAR-LN3-V21e-cq08-known-gate-causal-course-paired-oof/"
            f"fold-{fold_id:02d}/treatment/held-out-predictions.json"
        )
        control_prediction = json.loads(
            control_prediction_path.read_text(encoding="utf-8")
        )
        frozen_initial_logits_sha = dict(
            control_prediction.get("paired_initialization") or {}
        ).get("initial_logits_sha256")
        actual_initial_logits_sha = hashlib.sha256(
            treatment["sequence_logits"]
            .detach()
            .cpu()
            .contiguous()
            .numpy()
            .tobytes(order="C")
        ).hexdigest()
        if (
            _state_dict_sha256(backbone.state_dict())
            != control_prediction.get("initial_model_state_sha256")
            or actual_initial_logits_sha != frozen_initial_logits_sha
        ):
            raise AssertionError(
                f"V22b fold {fold_id} does not reproduce frozen V21e initialization"
            )
        residual_l2 = torch.linalg.vector_norm(treatment["rd_temporal_residual"], dim=-1)
        active = ~batch["tr_padding_mask"]
        multiframe = active & (batch["rd_frame_mask"].sum(dim=2) >= 2)
        if not bool((multiframe & (residual_l2 > 0.0)).any()):
            raise AssertionError(f"V22b fold {fold_id} has no residual activity")
        model.train()
        model.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(**_model_inputs(batch))["window_logits"], batch["labels"])
        loss.backward()
        if model.late_residual_weight.grad is None or float(model.late_residual_weight.grad.abs().max()) <= 0.0:
            raise AssertionError(f"V22b fold {fold_id} late weight has no gradient")
        fold_checks.append({
            "fold_id": fold_id,
            "train_parent_count": len(fold["train_parent_ids"]),
            "held_out_parent_count": len(fold["held_out_parent_ids"]),
            "initial_logits_exactly_equal": True,
            "frozen_v21e_initial_state_and_logits_reproduced": True,
            "all_model_parameters_trainable": all(parameter.requires_grad for parameter in model.parameters()),
            "late_weight_receives_gradient": True,
            "nonzero_residual_event_count": int((multiframe & (residual_l2 > 0.0)).sum().item()),
        })
    return {
        "status": "passed",
        "schema_version": "RADAR-LN3-V22b-known-gate-late-readout-oof-smoke-v1",
        "fold_count": FOLD_COUNT,
        "fold_checks": fold_checks,
        "held_out_forward_count": 0,
        "validation_access_count": 0,
        "test_payload_access_count": 0,
        "unknown_training_exposure": 0,
        "test": "未执行",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test V22b OOF.")
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.project_root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
