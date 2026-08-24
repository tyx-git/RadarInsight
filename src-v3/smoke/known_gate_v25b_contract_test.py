"""Synthetic and real-source contract tests for V25b exact replay.

The smoke deliberately replays one frozen V25a epoch-7 clipped mixed batch
from epoch-6 boundary state.  It never creates a validation/test Dataset and
never persists a diagnostic report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from build.known_gate_v25b import (
    EXPECTED_STEPS_PER_EPOCH,
    EXPERIMENT_CONTRACT,
    READINESS,
    assert_prepared_v25b_binding,
    load_v25a_source,
    v25b_implementation_binding,
)
from radar_ln3.known_gate import sha256_file
from radar_ln3.known_gate import (
    build_known_gate_dataloader,
    build_known_gate_dataset,
)
from radar_ln3.known_gate_v25b_gradient_decomposition import (
    all_trainable_parameters,
    assert_no_raw_identifiers,
    gradient_add,
    gradient_difference_metrics,
    gradient_l2,
    gradients_exactly_equal,
    gradients_from_scalar,
    module_parameter_indices,
    parent_batch_sha256,
    parent_loss_terms_batch_weight_sum,
    snapshot_parameter_gradients,
)
from radar_ln3.losses import parent_final_task_ce_with_prefix_auxiliary
from report.known_gate_v25b import (
    GRADIENT_NORM_ABSOLUTE_TOLERANCE,
    GRADIENT_NORM_RELATIVE_TOLERANCE,
    LOSS_ABSOLUTE_TOLERANCE,
    LOSS_RELATIVE_TOLERANCE,
    _initial_model_and_generator,
    _assert_decomposition_gradient_reconstruction,
    _decomposition_gradient_allowances,
    _load_epoch_boundary,
    _objective,
    _telemetry_for_epoch,
    _tensors_on_device,
    _verify_step_replay,
)


def _synthetic_loss_and_gradient_smoke() -> dict[str, bool]:
    torch.manual_seed(25051)
    logits = torch.randn(2, 6, 2, requires_grad=True)
    labels = torch.tensor([0, 1], dtype=torch.long)
    padding = torch.tensor(
        [[False, False, False, False, False, False], [False, False, False, False, True, True]],
        dtype=torch.bool,
    )
    weights = torch.tensor([0.5173137460650578, 14.93939393939394])
    terms = parent_loss_terms_batch_weight_sum(
        logits,
        labels,
        padding,
        class_weights=weights,
        prefix_auxiliary_weight=0.1,
        prefix_auxiliary_min_events=3,
    )
    original_objective = parent_final_task_ce_with_prefix_auxiliary(
        logits,
        labels,
        padding,
        class_weights=weights,
        prefix_auxiliary_weight=0.1,
        prefix_auxiliary_min_events=3,
        class_weight_normalization="batch_weight_sum",
    )
    original = original_objective["loss"]
    original.backward(retain_graph=True)
    actual = (logits.grad.detach().clone(),)
    before = (logits.grad.detach().clone(),)
    reconstructed = gradient_add(
        *(gradients_from_scalar(term, (logits,)) for term in terms["historical_total_terms"])
    )
    after = (logits.grad.detach().clone(),)
    gradient_metrics = gradient_difference_metrics(actual, reconstructed)
    raw_parent_terms = [
        gradients_from_scalar(terms["raw_total_terms"][index], (logits,))
        for index in range(2)
    ]
    raw_parent_nonzero = all(gradient_l2(value) > 0.0 for value in raw_parent_terms)
    return {
        "historical_parent_terms_sum_to_original_loss": bool(
            torch.allclose(
                terms["historical_total_terms"].sum(),
                original,
                atol=1.0e-7,
                rtol=1.0e-7,
            )
        ),
        "autograd_grad_does_not_mutate_parameter_grad": gradients_exactly_equal(before, after),
        "historical_parent_gradients_reconstruct_batch_gradient": (
            float(gradient_metrics["relative_residual_l2"]) < 1.0e-6
            and float(gradient_metrics["maximum_absolute_component_residual"]) < 1.0e-6
        ),
        "raw_parent_gradients_are_nonzero": raw_parent_nonzero,
    }


def _identifier_smoke() -> bool:
    safe = {
        "parent_batch_sha256": parent_batch_sha256(["opaque-a", "opaque-b"]),
        "nested": {"pair": "ClutterTarget+DroneTarget"},
    }
    assert_no_raw_identifiers(safe, raw_parent_ids=["opaque-a", "opaque-b"])
    leaked_rejected = False
    try:
        assert_no_raw_identifiers(
            {"parent_sample_id": "opaque-a"}, raw_parent_ids=["opaque-a", "opaque-b"]
        )
    except ValueError:
        leaked_rejected = True
    return leaked_rejected


def _prepared_binding_smoke(root: Path) -> dict[str, bool]:
    """Prove V25b rejects an internally valid but differently bound source."""

    source = load_v25a_source(root)
    contract_path = root / EXPERIMENT_CONTRACT
    readiness_path = root / READINESS
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    assert_prepared_v25b_binding(
        root,
        source,
        contract=contract,
        readiness=readiness,
        contract_sha256=sha256_file(contract_path),
    )
    tampered = json.loads(json.dumps(contract))
    tampered["source_v25a"]["training_summary_sha256"] = "0" * 64
    tampering_rejected = False
    try:
        assert_prepared_v25b_binding(
            root,
            source,
            contract=tampered,
            readiness=readiness,
            contract_sha256=sha256_file(contract_path),
        )
    except ValueError:
        tampering_rejected = True
    return {
        "current_prepared_binding_matches_exact_source": True,
        "mutated_source_summary_binding_is_rejected": tampering_rejected,
    }


def _implementation_binding_smoke(root: Path) -> bool:
    contract = json.loads((root / EXPERIMENT_CONTRACT).read_text(encoding="utf-8"))
    return dict(contract.get("implementation_sha256") or {}) == v25b_implementation_binding(root)


def _real_epoch7_step83_replay_smoke(root: Path) -> dict[str, Any]:
    source = load_v25a_source(root)
    config = source["config"]
    model, optimizer, generator, device = _load_epoch_boundary(
        root=root, source=source, config=config, epoch_before=6
    )
    dataset = build_known_gate_dataset(root, config, split_name="train")
    loader = build_known_gate_dataloader(
        dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        generator=generator,
    )
    telemetry_rows = _telemetry_for_epoch(source, 7)
    parameters = all_trainable_parameters(model)
    module_indices = module_parameter_indices(model, parameters)
    if tuple(module_indices) != (
        "tr_encoder",
        "rd_encoder",
        "fusion",
        "liquid",
        "classifier",
    ):
        raise AssertionError("V25b real replay has an incomplete module partition")
    class_weights = torch.tensor(config.train.class_weights, dtype=torch.float32, device=device)
    model.train()
    target_step = 83
    target_result: dict[str, Any] | None = None
    for step, (batch, telemetry) in enumerate(zip(loader, telemetry_rows, strict=True), start=1):
        tensors = _tensors_on_device(batch, device)
        output, objective = _objective(model, tensors, class_weights=class_weights, config=config)
        optimizer.zero_grad(set_to_none=True)
        objective["loss"].backward(retain_graph=step == target_step)
        actual = snapshot_parameter_gradients(parameters)
        actual_norm = gradient_l2(actual)
        checks = _verify_step_replay(
            batch=batch,
            telemetry=telemetry,
            objective=objective,
            actual_gradient_l2=actual_norm,
        )
        if step == target_step:
            if not bool(telemetry["contains_clutter_parent"]) or not bool(
                telemetry["was_globally_clipped"]
            ):
                raise AssertionError("V25b selected real replay probe lost its clipped mixed-batch binding")
            terms = parent_loss_terms_batch_weight_sum(
                output["sequence_logits"],
                tensors["labels"],
                tensors["tr_padding_mask"],
                class_weights=class_weights,
                prefix_auxiliary_weight=config.train.prefix_auxiliary_weight,
                prefix_auxiliary_min_events=config.train.prefix_auxiliary_min_events,
            )
            before = snapshot_parameter_gradients(parameters)
            explicit_proxy = gradients_from_scalar(
                terms["historical_total_loss"], parameters
            )
            reconstructed = gradient_add(
                *(
                    gradients_from_scalar(term, parameters)
                    for term in terms["historical_total_terms"]
                )
            )
            after = snapshot_parameter_gradients(parameters)
            source_to_explicit_proxy = gradient_difference_metrics(actual, explicit_proxy)
            parent_terms_to_explicit_proxy = gradient_difference_metrics(
                explicit_proxy, reconstructed
            )
            source_to_parent = gradient_difference_metrics(actual, reconstructed)
            loss_error = abs(
                float(terms["historical_total_loss"].detach().item())
                - float(objective["loss"].detach().item())
            )
            target_result = {
                "epoch": 7,
                "step": target_step,
                "parent_batch_sha256": parent_batch_sha256(batch["parent_sample_ids"]),
                "telemetry_parent_batch_sha256": telemetry["parent_batch_sha256"],
                "batch_hash_exact": bool(checks["batch_hash_exact"] == 1.0),
                "loss_absolute_error": float(checks["total_absolute_error"]),
                "pre_clip_norm_absolute_error": float(
                    checks["pre_clip_global_l2_absolute_error"]
                ),
                "clip_decision_exact": bool(checks["clip_decision_exact"] == 1.0),
                "diagnostic_autograd_did_not_mutate_gradients": gradients_exactly_equal(
                    before, after
                ),
                "decomposition_loss_absolute_error": loss_error,
                "source_to_explicit_proxy_gradient": source_to_explicit_proxy,
                "parent_terms_to_explicit_proxy_gradient": parent_terms_to_explicit_proxy,
                "source_to_parent_gradient": source_to_parent,
            }
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.gradient_clip_norm)
        optimizer.step()
        if step == target_step:
            break
    if target_result is None:
        raise AssertionError("V25b real replay did not reach selected probe step")
    if (
        not target_result["batch_hash_exact"]
        or target_result["loss_absolute_error"] > LOSS_ABSOLUTE_TOLERANCE
        or target_result["pre_clip_norm_absolute_error"] > max(
            GRADIENT_NORM_ABSOLUTE_TOLERANCE,
            GRADIENT_NORM_RELATIVE_TOLERANCE * 124.0,
        )
        or not target_result["clip_decision_exact"]
        or not target_result["diagnostic_autograd_did_not_mutate_gradients"]
        or target_result["decomposition_loss_absolute_error"] > 3.0e-6
    ):
        raise AssertionError(f"V25b real V25a replay probe failed: {target_result}")
    bridge_allowances = {
        "source_to_explicit_proxy": _assert_decomposition_gradient_reconstruction(
            target_result["source_to_explicit_proxy_gradient"],
            context="smoke_epoch7_step83_source_to_explicit_proxy",
        ),
        "parent_terms_to_explicit_proxy": _assert_decomposition_gradient_reconstruction(
            target_result["parent_terms_to_explicit_proxy_gradient"],
            context="smoke_epoch7_step83_parent_terms_to_explicit_proxy",
        ),
        "source_to_parent": _assert_decomposition_gradient_reconstruction(
            target_result["source_to_parent_gradient"],
            context="smoke_epoch7_step83_source_to_parent",
        ),
    }
    target_result["decomposition_gradient_allowances"] = bridge_allowances
    return target_result


def run(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    if not (root / READINESS).is_file() or not (root / EXPERIMENT_CONTRACT).is_file():
        raise FileNotFoundError("V25b prepare artifacts are missing; run prepare first")
    source = load_v25a_source(root)
    contract_path = root / EXPERIMENT_CONTRACT
    readiness_path = root / READINESS
    assert_prepared_v25b_binding(
        root,
        source,
        contract=json.loads(contract_path.read_text(encoding="utf-8")),
        readiness=json.loads(readiness_path.read_text(encoding="utf-8")),
        contract_sha256=sha256_file(contract_path),
    )
    synthetic = _synthetic_loss_and_gradient_smoke()
    if not all(synthetic.values()):
        raise AssertionError(f"V25b synthetic loss decomposition smoke failed: {synthetic}")
    identifiers = _identifier_smoke()
    if not identifiers:
        raise AssertionError("V25b identifier-leak smoke failed")
    prepared_binding = _prepared_binding_smoke(root)
    if not all(prepared_binding.values()):
        raise AssertionError(f"V25b prepared-source binding smoke failed: {prepared_binding}")
    implementation_binding = _implementation_binding_smoke(root)
    if not implementation_binding:
        raise AssertionError("V25b implementation binding smoke failed")
    real = _real_epoch7_step83_replay_smoke(root)
    return {
        "status": "passed",
        "schema_version": "RADAR-LN3-V25b-known-gate-exact-replay-parent-gradient-decomposition-smoke-v1",
        "synthetic_weighted_loss_and_gradient_reconstruction": synthetic,
        "raw_parent_identifiers_rejected_from_persisted_payload": identifiers,
        "prepared_source_binding": prepared_binding,
        "implementation_hash_binding_matches_current_code": implementation_binding,
        "real_v25a_epoch7_step83_replay": real,
        "validation_dataset_constructed": False,
        "test_payload_access_count": 0,
        "unknown_payload_access_count": 0,
        "checkpoint_written": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke-test V25b V25a exact-replay gradient decomposition."
    )
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    args = parser.parse_args()
    print(json.dumps(run(args.project_root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
