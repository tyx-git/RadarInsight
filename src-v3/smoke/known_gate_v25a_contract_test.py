"""Static and runtime contract checks for the V25a telemetry-only audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import torch

from build.known_gate_v24 import CONFIG as V24_CONFIG
from build.known_gate_v25a import CONFIG, READINESS
from radar_ln3.known_gate import (
    V25A_GRADIENT_TELEMETRY_VARIANT,
    build_known_gate_dataset,
    build_known_gate_model_for_config,
    load_known_gate_config,
)
from radar_ln3.known_gate_gradient_telemetry import (
    V25A_GRADIENT_MODULE_NAMES,
    V25A_RECORD_FIELDS,
    V25AGradientTelemetryRecorder,
)
from radar_ln3.known_gate_features import COURSE_PREFIX_FEATURE_NAMES
from radar_ln3.losses import parent_final_task_ce_with_prefix_auxiliary
from train.known_gate_v21 import _model_inputs, _verify_contract, _verify_readiness
from train.known_gate_v25a import (
    _reconcile_resumed_epoch_checkpoint_artifacts,
    _require_v24_equivalent_config,
)


def _batch() -> dict[str, Any]:
    torch.manual_seed(25043)
    batch, length, frames = 2, 5, 3
    return {
        "parent_sample_ids": ["opaque-parent-a", "opaque-parent-b"],
        "task_label_names": ["KNOWN_TARGET_EVIDENCE", "KNOWN_CLUTTER_EVIDENCE"],
        "base_label_names": ["BirdTarget", "ClutterTarget"],
        "tr_features": torch.randn(batch, length, 14),
        "tr_missing_mask": torch.zeros(batch, length, 11),
        "tr_dt_seconds": torch.tensor(
            [[0.0, 0.0, 1.0, 1.0, 2.0], [0.0, 0.0, 2.0, 1.0, 1.0]],
            dtype=torch.float32,
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
        "known_gate_features": torch.rand(batch, length, 1),
        "known_gate_feature_valid": torch.ones(batch, length, 1, dtype=torch.bool),
        "labels": torch.tensor([0, 1], dtype=torch.long),
    }


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dataset_cache_binding_smoke(root: Path, config: Any) -> dict[str, int]:
    """Load both real V25a Dataset splits without reading RD payloads.

    The V25a cache is the one-column V21e causal-course cache.  This catches
    a regression where a reuse-only variant is registered in the feature
    schema but omitted from the cache-local column-selection path.
    """

    expected = {
        "train": {"parents": 986, "events": 18923},
        "val": {"parents": 232, "events": 4452},
    }
    result: dict[str, int] = {}
    for split_name, expected_counts in expected.items():
        dataset = build_known_gate_dataset(root, config, split_name=split_name)
        metadata = dataset.feature_cache_metadata
        values = dataset._evidence_values
        valid = dataset._evidence_valid
        if (
            len(dataset) != expected_counts["parents"]
            or metadata is None
            or values is None
            or valid is None
            or metadata.get("variant") != V25A_GRADIENT_TELEMETRY_VARIANT
            or metadata.get("feature_names") != list(COURSE_PREFIX_FEATURE_NAMES)
            or int(metadata.get("event_count", -1)) != expected_counts["events"]
            or len(values) != expected_counts["events"]
            or len(valid) != expected_counts["events"]
        ):
            raise AssertionError(
                f"V25a {split_name} Dataset/cache binding differs from the sealed contract"
            )
        value_shape = next(iter(values.values())).shape
        valid_shape = next(iter(valid.values())).shape
        if value_shape != (1,) or valid_shape != (1,):
            raise AssertionError(
                f"V25a {split_name} Dataset did not load one-dimensional course evidence"
            )
        result[f"{split_name}_parent_count"] = len(dataset)
        result[f"{split_name}_evidence_dim"] = int(value_shape[0])
    return result


def _exactly_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and torch.equal(left, right)
        )
    if isinstance(left, dict) or isinstance(right, dict):
        if not isinstance(left, dict) or not isinstance(right, dict) or set(left) != set(right):
            return False
        return all(_exactly_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            isinstance(left, type(right))
            and len(left) == len(right)
            and all(_exactly_equal(a, b) for a, b in zip(left, right, strict=True))
        )
    return left == right


def _v24_one_step_equivalence_smoke(
    root: Path,
    v25_config: Any,
    batch: dict[str, Any],
) -> bool:
    """Prove telemetry does not alter the V24 forward/backward/clip/AdamW step."""

    v24_config = load_known_gate_config(root / V24_CONFIG)
    control, control_reference, _ = build_known_gate_model_for_config(root, v24_config)
    treatment, treatment_reference, _ = build_known_gate_model_for_config(
        root, v25_config
    )
    if (
        control_reference is None
        or treatment_reference is None
        or not _exactly_equal(control.state_dict(), treatment.state_dict())
    ):
        raise AssertionError("V25a smoke could not reconstruct V24-equivalent initial weights")
    del control_reference, treatment_reference
    control.train()
    treatment.train()
    control_optimizer = torch.optim.AdamW(
        control.parameters(), lr=v24_config.train.learning_rate
    )
    treatment_optimizer = torch.optim.AdamW(
        treatment.parameters(), lr=v25_config.train.learning_rate
    )
    class_weights = torch.tensor(v24_config.train.class_weights, dtype=torch.float32)
    control_optimizer.zero_grad(set_to_none=True)
    control_objective = parent_final_task_ce_with_prefix_auxiliary(
        control(**_model_inputs(batch))["sequence_logits"],
        batch["labels"],
        batch["tr_padding_mask"],
        class_weights=class_weights,
        prefix_auxiliary_weight=v24_config.train.prefix_auxiliary_weight,
        prefix_auxiliary_min_events=v24_config.train.prefix_auxiliary_min_events,
    )
    control_objective["loss"].backward()
    control_norm = float(
        torch.nn.utils.clip_grad_norm_(
            control.parameters(), v24_config.train.gradient_clip_norm
        ).item()
    )
    control_optimizer.step()

    treatment_optimizer.zero_grad(set_to_none=True)
    treatment_objective = parent_final_task_ce_with_prefix_auxiliary(
        treatment(**_model_inputs(batch))["sequence_logits"],
        batch["labels"],
        batch["tr_padding_mask"],
        class_weights=class_weights,
        prefix_auxiliary_weight=v25_config.train.prefix_auxiliary_weight,
        prefix_auxiliary_min_events=v25_config.train.prefix_auxiliary_min_events,
    )
    treatment_objective["loss"].backward()
    with tempfile.TemporaryDirectory(prefix="v25a-one-step-equivalence-") as temporary:
        root_path = Path(temporary)
        source_summary = root_path / "source-summary.json"
        source_summary.write_text("{}\n", encoding="utf-8")
        recorder = V25AGradientTelemetryRecorder(
            artifact_dir=root_path / "artifact",
            config_path=root_path / "config.json",
            config_file_sha256="config-hash",
            config_identity_sha256="identity-hash",
            source_summary_path=source_summary,
            source_summary_sha256=_sha(source_summary),
            expected_parent_order_hashes={
                epoch: f"order-{epoch}" for epoch in range(1, 21)
            },
        )
        recorder.start(
            model=treatment,
            steps_per_epoch=1,
            initial_model_state_sha256="initial-state",
            completed_history=[],
            resumed=False,
        )
        record = recorder.capture_after_backward(
            batch=batch,
            tensors=batch,
            objective=treatment_objective,
            epoch=1,
            step=1,
            global_step=1,
        )
        treatment_norm = float(
            torch.nn.utils.clip_grad_norm_(
                treatment.parameters(), v25_config.train.gradient_clip_norm
            ).item()
        )
        recorder.record_after_clip_before_optimizer_step(
            record,
            pre_clip_global_norm=treatment_norm,
            clip_threshold=v25_config.train.gradient_clip_norm,
        )
        treatment_optimizer.step()
    result = (
        float(control_objective["loss"].item()) == float(treatment_objective["loss"].item())
        and control_norm == treatment_norm
        and _exactly_equal(control.state_dict(), treatment.state_dict())
        and _exactly_equal(control_optimizer.state_dict(), treatment_optimizer.state_dict())
    )
    control.zero_grad(set_to_none=True)
    treatment.zero_grad(set_to_none=True)
    return result


def _recorder_smoke(model: torch.nn.Module, batch: dict[str, Any]) -> dict[str, Any]:
    state_before = {
        name: parameter.grad.detach().clone() if parameter.grad is not None else None
        for name, parameter in model.named_parameters()
    }
    objective = parent_final_task_ce_with_prefix_auxiliary(
        model(**_model_inputs(batch))["sequence_logits"],
        batch["labels"],
        batch["tr_padding_mask"],
        class_weights=torch.ones(2),
        prefix_auxiliary_weight=0.1,
        prefix_auxiliary_min_events=3,
    )
    objective["loss"].backward()
    gradients_before = {
        name: parameter.grad.detach().clone() if parameter.grad is not None else None
        for name, parameter in model.named_parameters()
    }
    with tempfile.TemporaryDirectory(prefix="v25a-telemetry-smoke-") as temporary:
        artifact = Path(temporary) / "artifact"
        source_summary = Path(temporary) / "source-summary.json"
        source_summary.write_text(
            json.dumps(
                {
                    "status": "completed",
                    "dataloader_contract": {"epoch_parent_order_hashes": []},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        recorder = V25AGradientTelemetryRecorder(
            artifact_dir=artifact,
            config_path=Path(temporary) / "config.json",
            config_file_sha256="config-hash",
            config_identity_sha256="identity-hash",
            source_summary_path=source_summary,
            source_summary_sha256=_sha(source_summary),
            expected_parent_order_hashes={epoch: f"order-{epoch}" for epoch in range(1, 21)},
        )
        recorder.start(
            model=model,
            steps_per_epoch=1,
            initial_model_state_sha256="initial-state",
            completed_history=[],
            resumed=False,
        )
        record = recorder.capture_after_backward(
            batch=batch,
            tensors=batch,
            objective=objective,
            epoch=1,
            step=1,
            global_step=1,
        )
        gradients_after_capture = {
            name: parameter.grad.detach().clone() if parameter.grad is not None else None
            for name, parameter in model.named_parameters()
        }
        if any(
            (left is None) != (right is None)
            or (left is not None and not torch.equal(left, right))
            for left, right in zip(gradients_before.values(), gradients_after_capture.values(), strict=True)
        ):
            raise AssertionError("V25a telemetry capture modified a gradient")
        global_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0).item())
        recorder.record_after_clip_before_optimizer_step(
            record,
            pre_clip_global_norm=global_norm,
            clip_threshold=100.0,
        )
        finalized_record = dict(recorder._records[-1])
        staged = recorder.stage_epoch(epoch=1, global_step=1, parent_order_sha256="order-1")
        committed = recorder.commit_epoch(
            epoch=1,
            staged=staged,
            expected_global_step_end=1,
            expected_parent_order_sha256="order-1",
        )
        descriptor = recorder.committed_epoch_descriptor(
            epoch=1,
            expected_global_step_end=1,
            expected_parent_order_sha256="order-1",
        )
        payload = json.loads(Path(committed["path"]).read_text(encoding="utf-8"))
    model.zero_grad(set_to_none=True)
    return {
        "pre_capture_gradients_were_none": all(value is None for value in state_before.values()),
        "telemetry_capture_did_not_modify_gradients": True,
        "module_partition": list(record["pre_clip_module_l2"]),
        "record_field_schema_is_exact": set(finalized_record) == V25A_RECORD_FIELDS,
        "parent_identifier_is_hashed": record["parent_batch_sha256"]
        != hashlib.sha256("opaque-parent-a".encode("utf-8")).hexdigest(),
        "raw_parent_identifier_absent_from_payload": "opaque-parent-a" not in json.dumps(payload),
        "module_norm_matches_global_clip_norm": abs(
            float(record["pre_clip_module_total_l2"]) - global_norm
        )
        <= max(1.0e-4, 1.0e-4 * max(1.0, global_norm)),
        "staged_commit_descriptor_matches": committed == descriptor,
    }


def _resume_recovery_smoke(model: torch.nn.Module, batch: dict[str, Any]) -> bool:
    """Exercise the checkpoint-bound pending-file recovery path without training."""

    model.zero_grad(set_to_none=True)
    objective = parent_final_task_ce_with_prefix_auxiliary(
        model(**_model_inputs(batch))["sequence_logits"],
        batch["labels"],
        batch["tr_padding_mask"],
        class_weights=torch.ones(2),
        prefix_auxiliary_weight=0.1,
        prefix_auxiliary_min_events=3,
    )
    objective["loss"].backward()
    with tempfile.TemporaryDirectory(prefix="v25a-telemetry-resume-smoke-") as temporary:
        root = Path(temporary)
        artifact = root / "artifact"
        source_summary = root / "source-summary.json"
        source_summary.write_text("{}\n", encoding="utf-8")
        common = {
            "artifact_dir": artifact,
            "config_path": root / "config.json",
            "config_file_sha256": "config-hash",
            "config_identity_sha256": "identity-hash",
            "source_summary_path": source_summary,
            "source_summary_sha256": _sha(source_summary),
            "expected_parent_order_hashes": {
                epoch: f"order-{epoch}" for epoch in range(1, 21)
            },
        }
        first = V25AGradientTelemetryRecorder(**common)
        first.start(
            model=model,
            steps_per_epoch=1,
            initial_model_state_sha256="initial-state",
            completed_history=[],
            resumed=False,
        )
        record = first.capture_after_backward(
            batch=batch,
            tensors=batch,
            objective=objective,
            epoch=1,
            step=1,
            global_step=1,
        )
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0).item())
        first.record_after_clip_before_optimizer_step(
            record, pre_clip_global_norm=norm, clip_threshold=100.0
        )
        staged = first.stage_epoch(epoch=1, global_step=1, parent_order_sha256="order-1")
        history = [
            {
                "epoch": 1,
                "global_step": 1,
                "parent_order_sha256": "order-1",
                "gradient_telemetry": staged,
            }
        ]
        resumed = V25AGradientTelemetryRecorder(**common)
        resumed.start(
            model=model,
            steps_per_epoch=1,
            initial_model_state_sha256="initial-state",
            completed_history=history,
            resumed=True,
        )
        final = resumed.committed_epoch_descriptor(
            epoch=1,
            expected_global_step_end=1,
            expected_parent_order_sha256="order-1",
        )
        final_path = Path(final["path"])
        pending_path = final_path.with_suffix(final_path.suffix + ".pending")
        result = final_path.is_file() and not pending_path.exists()
    model.zero_grad(set_to_none=True)
    return result


def _checkpoint_companion_recovery_smoke(
    root: Path,
    *,
    config: Any,
    readiness_sha256: str,
    model: torch.nn.Module,
) -> bool:
    """Exercise the interruption window after last checkpoint persistence.

    The telemetry recovery smoke above covers pending-to-final promotion.  This
    companion smoke covers the independent file-system boundary where the
    latest epoch/best checkpoint has not yet been created when a durable
    ``last_model.pt`` is available for resume.
    """

    model.zero_grad(set_to_none=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.learning_rate)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.train.dataloader_seed)
    selection = {
        "metric": "parent_min_precision_recall",
        "primary_metric": "parent_min_class_precision_recall",
        "primary_value": 0.5,
        "tie_break_metric": "parent_macro_f1",
        "tie_break_value": 0.5,
        "ranking_key": [0.5, 0.5],
    }
    history = [
        {
            "epoch": 1,
            "global_step": 1,
            "checkpoint_selection": selection,
            "parent_order_sha256": "order-1",
            "gradient_telemetry": {
                "path": "final.json",
                "sha256": "hash",
                "epoch": 1,
                "record_count": 1,
            },
        }
    ]
    resume_payload = {
        "epoch": 1,
        "global_step": 1,
        "metrics": {"epoch": 1},
    }
    with tempfile.TemporaryDirectory(prefix="v25a-checkpoint-companion-smoke-") as temporary:
        artifact = Path(temporary) / "artifact"
        result = _reconcile_resumed_epoch_checkpoint_artifacts(
            model=model,
            optimizer=optimizer,
            generator=generator,
            config=config,
            config_path=root / CONFIG,
            readiness_sha256=readiness_sha256,
            history=history,
            best_epoch=1,
            resume_payload=resume_payload,
            last_checkpoint=artifact / "last_model.pt",
            best_checkpoint=artifact / "best_model.pt",
            epoch_checkpoint_dir=artifact / "epochs",
        )
        expected_paths = {
            artifact / "last_model.pt",
            artifact / "best_model.pt",
            artifact / "epochs" / "epoch-001.pt",
        }
        if not all(path.is_file() for path in expected_paths):
            return False
        recovered = torch.load(artifact / "last_model.pt", map_location="cpu", weights_only=False)
        return (
            result.get("performed") is True
            and int(result.get("checkpoint_epoch", -1)) == 1
            and int(result.get("best_epoch", -1)) == 1
            and set(Path(path) for path in result.get("rewritten_paths", ())) == expected_paths
            and int(recovered.get("epoch", -1)) == 1
            and int(recovered.get("global_step", -1)) == 1
            and recovered.get("metrics", {}).get("gradient_telemetry", {}).get("path") == "final.json"
        )


def run(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    config_path = root / CONFIG
    readiness_path = root / READINESS
    if not config_path.is_file() or not readiness_path.is_file():
        raise FileNotFoundError("V25a prepare artifacts are incomplete; run prepare first")
    config = load_known_gate_config(config_path)
    _require_v24_equivalent_config(config)
    contract = _verify_contract(root, config)
    readiness = _verify_readiness(root, config_path, config, contract)
    if (
        config.variant != V25A_GRADIENT_TELEMETRY_VARIANT
        or config.feature_names != COURSE_PREFIX_FEATURE_NAMES
        or config.model.evidence_dim != 1
        or readiness.get("source_v24_initial_model_state_sha256")
        != "fecd0ca1aa21c0a9cdb4b0425c64703620cb2ea0461d37e7fa5a2057cea8c877"
        or not str(readiness.get("paired_v21_seed43_baseline_initial_model_state_sha256", ""))
    ):
        raise AssertionError("V25a config/readiness identity is stale")
    dataset_cache_binding = _dataset_cache_binding_smoke(root, config)
    treatment, reference, pairing = build_known_gate_model_for_config(root, config)
    if (
        reference is None
        or pairing.get("new_fusion_columns_exactly_zero") is not True
        or int(pairing.get("new_fusion_columns", -1)) != 1
    ):
        raise AssertionError("V25a paired initialization is incomplete")
    treatment.eval()
    reference.eval()
    with torch.inference_mode():
        paired_treatment_logits = treatment(**_model_inputs(_batch()))["sequence_logits"]
        paired_reference_inputs = _model_inputs(_batch())
        paired_reference_inputs["known_gate_features"] = None
        paired_reference_inputs["known_gate_feature_valid"] = None
        paired_reference_logits = reference(**paired_reference_inputs)["sequence_logits"]
    if not torch.equal(paired_treatment_logits, paired_reference_logits):
        raise AssertionError("V25a zero-initialized course evidence no longer preserves paired logits")
    batch = _batch()
    treatment.train()
    one_step_equivalent = _v24_one_step_equivalence_smoke(root, config, batch)
    audit = _recorder_smoke(treatment, batch)
    resume_recovery = _resume_recovery_smoke(treatment, batch)
    checkpoint_companion_recovery = _checkpoint_companion_recovery_smoke(
        root,
        config=config,
        readiness_sha256=str(readiness["sha256"]),
        model=treatment,
    )
    if tuple(audit["module_partition"]) != V25A_GRADIENT_MODULE_NAMES:
        raise AssertionError("V25a module partition is incomplete")
    if not all(
        audit[name]
        for name in (
            "telemetry_capture_did_not_modify_gradients",
            "parent_identifier_is_hashed",
            "raw_parent_identifier_absent_from_payload",
            "module_norm_matches_global_clip_norm",
            "staged_commit_descriptor_matches",
            "record_field_schema_is_exact",
        )
    ) or not resume_recovery or not checkpoint_companion_recovery or not one_step_equivalent:
        raise AssertionError("V25a telemetry-only smoke check failed")
    return {
        "status": "passed",
        "schema_version": "RADAR-LN3-V25a-known-gate-preclip-gradient-telemetry-smoke-v1",
        "v24_initialization_binding": True,
        "v21_reference_initialization_binding": True,
        "initial_logits_exactly_equal_to_paired_v21_reference": True,
        "telemetry_is_read_only": True,
        "one_step_update_exactly_matches_v24": True,
        "train_validation_dataset_cache_binding": True,
        "train_parent_count": dataset_cache_binding["train_parent_count"],
        "validation_parent_count": dataset_cache_binding["val_parent_count"],
        "evidence_dim": dataset_cache_binding["train_evidence_dim"],
        "preclip_module_partition": list(V25A_GRADIENT_MODULE_NAMES),
        "raw_parent_identifiers_not_stored": True,
        "staged_epoch_recovery_contract": True,
        "latest_checkpoint_companion_recovery_contract": True,
        "unknown_training_exposure": 0,
        "test": "not_executed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test V25a telemetry-only contracts.")
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    args = parser.parse_args()
    print(json.dumps(run(args.project_root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
