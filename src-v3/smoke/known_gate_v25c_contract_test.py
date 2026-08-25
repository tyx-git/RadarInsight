"""Synthetic and real-source contract tests for V25c module conflict replay.

The real probe replays only V25a epoch 7 through its fixed clipped mixed batch.
It never builds validation/test/Unknown data and never writes a V25c report.
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch

from build.known_gate_v25b import load_v25a_source
from build.known_gate_v25c import (
    EXPERIMENT_CONTRACT,
    READINESS,
    assert_prepared_v25c_binding,
    v25c_implementation_binding,
)
from radar_ln3.known_gate import build_known_gate_dataloader, build_known_gate_dataset, sha256_file
from radar_ln3.known_gate_v25b_gradient_decomposition import (
    all_trainable_parameters,
    assert_no_raw_identifiers,
    gradient_add,
    gradient_difference_metrics,
    gradient_l2,
    gradients_exactly_equal,
    parent_loss_terms_batch_weight_sum,
    snapshot_parameter_gradients,
)
from radar_ln3.known_gate_v25c_module_gradient_conflict import (
    V25C_COMPONENT_NAMES,
    V25C_COMPONENT_ADDITIVITY_MAX_COMPONENT_RELATIVE_TO_PEAK_TOLERANCE,
    V25C_WORKER_FAILURE_OUTPUT_LIMIT,
    V25C_WORKER_PAYLOAD_MAX_BYTES,
    V25C_MODULE_NAMES,
    module_and_full_pair_conflict_metrics,
    parent_component_additivity_metrics,
    parent_component_gradients,
    v25c_component_additivity_tolerance_policy,
    v25c_execution_resource_safety_policy,
)
from report.known_gate_v25c import (
    GRADIENT_NORM_ABSOLUTE_TOLERANCE,
    GRADIENT_NORM_RELATIVE_TOLERANCE,
    LOSS_ABSOLUTE_TOLERANCE,
    _assert_decomposition_gradient_reconstruction,
    _assert_component_additivity,
    _decomposition_gradient_allowances,
    _load_epoch_boundary,
    _mixed_conflict_record,
    _objective,
    _read_limited_text,
    _read_worker_payload,
    _telemetry_for_epoch,
    _tensors_on_device,
    _verify_step_replay,
    _write_worker_payload,
)


def _synthetic_component_conflict_smoke() -> dict[str, bool]:
    torch.manual_seed(25052)
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
    terms["historical_total_loss"].backward(retain_graph=True)
    before = (logits.grad.detach().clone(),)
    known = parent_component_gradients(terms, (logits,), 0)
    clutter = parent_component_gradients(terms, (logits,), 1)
    after = (logits.grad.detach().clone(),)
    known_additivity = parent_component_additivity_metrics(known)
    clutter_additivity = parent_component_additivity_metrics(clutter)
    # The production module partition has five nonempty modules.  Use a compact
    # synthetic vector with five tensors so the exact same partition check is
    # exercised without constructing a model.
    synthetic_known = tuple(torch.randn(2, requires_grad=False) for _ in V25C_MODULE_NAMES)
    synthetic_clutter = tuple(torch.randn(2, requires_grad=False) for _ in V25C_MODULE_NAMES)
    conflict = module_and_full_pair_conflict_metrics(
        synthetic_known,
        synthetic_clutter,
        {module: (index,) for index, module in enumerate(V25C_MODULE_NAMES)},
    )
    return {
        "component_order_is_frozen": tuple(known) == V25C_COMPONENT_NAMES,
        "diagnostic_autograd_does_not_mutate_parameter_grad": gradients_exactly_equal(before, after),
        "known_raw_total_equals_final_plus_prefix": float(
            known_additivity["raw_total_equals_final_plus_prefix"]["relative_residual_l2"]
        ) < 1.0e-6,
        "known_historical_total_equals_final_plus_prefix": float(
            known_additivity["historical_total_equals_final_plus_prefix"]["relative_residual_l2"]
        ) < 1.0e-6,
        "clutter_raw_total_equals_final_plus_prefix": float(
            clutter_additivity["raw_total_equals_final_plus_prefix"]["relative_residual_l2"]
        ) < 1.0e-6,
        "clutter_historical_total_equals_final_plus_prefix": float(
            clutter_additivity["historical_total_equals_final_plus_prefix"]["relative_residual_l2"]
        ) < 1.0e-6,
        "full_and_each_module_conflict_metrics_are_finite": (
            set(conflict) == {"full_model", "modules"}
            and tuple(conflict["modules"]) == V25C_MODULE_NAMES
            and all(
                -1.0 <= float(value["cosine"]) <= 1.0
                for value in [conflict["full_model"], *conflict["modules"].values()]
                if value["cosine"] is not None
            )
        ),
    }


def _identifier_smoke() -> bool:
    safe = {"parent_batch_sha256": "a" * 64, "known_base_label": "BirdTarget"}
    assert_no_raw_identifiers(safe, raw_parent_ids=["opaque-a", "opaque-b"])
    try:
        assert_no_raw_identifiers(
            {"parent_sample_ids": ["opaque-a"]}, raw_parent_ids=["opaque-a", "opaque-b"]
        )
    except ValueError:
        return True
    return False


def _resource_transport_smoke(root: Path) -> dict[str, bool]:
    """Exercise bounded sidecar and stderr-tail helpers without CUDA replay."""

    policy = v25c_execution_resource_safety_policy()
    additivity_policy = v25c_component_additivity_tolerance_policy()
    stderr = io.BytesIO(b"prefix-" + b"x" * 128 + b"-tail")
    tail = _read_limited_text(stderr, limit=16)
    with tempfile.TemporaryDirectory(dir=root / "tmp", prefix="v25c-smoke-") as directory:
        path = Path(directory) / "epoch.json"
        _write_worker_payload(
            path,
            {
                "schema_version": "synthetic",
                "epoch": 1,
                "rows": [],
                "integrity": {},
                "boundary": {},
            },
        )
        payload = _read_worker_payload(path, epoch=1)
    return {
        "resource_policy_is_frozen": (
            policy["epoch_worker_transport"]
            == "atomic_json_sidecar_in_project_tmp_not_stdout"
            and int(policy["worker_payload_max_bytes"]) == V25C_WORKER_PAYLOAD_MAX_BYTES
            and int(policy["worker_stderr_tail_max_bytes"])
            == V25C_WORKER_FAILURE_OUTPUT_LIMIT
            and policy["completed_report_stdout"] == "compact_completion_only"
        ),
        "worker_payload_round_trip_is_atomic_and_bounded": payload["schema_version"] == "synthetic",
        "worker_stderr_tail_is_bounded_and_preserves_suffix": len(tail.encode("utf-8")) <= 16
        and tail.endswith("-tail"),
        "component_additivity_relative_peak_cap_is_exactly_registered_1e_4": (
            V25C_COMPONENT_ADDITIVITY_MAX_COMPONENT_RELATIVE_TO_PEAK_TOLERANCE == 1.0e-4
            and float(
                additivity_policy["maximum_component_relative_to_parent_total_peak"]
            )
            == 1.0e-4
        ),
    }


def _prepared_binding_smoke(root: Path) -> dict[str, bool]:
    source = load_v25a_source(root)
    contract_path = root / EXPERIMENT_CONTRACT
    readiness_path = root / READINESS
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    assert_prepared_v25c_binding(
        root,
        source,
        contract=contract,
        readiness=readiness,
        contract_sha256=sha256_file(contract_path),
    )
    tampered = json.loads(json.dumps(contract))
    tampered["source_v25a"]["training_summary_sha256"] = "0" * 64
    rejected = False
    try:
        assert_prepared_v25c_binding(
            root,
            source,
            contract=tampered,
            readiness=readiness,
            contract_sha256=sha256_file(contract_path),
        )
    except ValueError:
        rejected = True
    return {
        "current_prepared_binding_matches_exact_source": True,
        "mutated_source_summary_binding_is_rejected": rejected,
    }


def _implementation_binding_smoke(root: Path) -> bool:
    contract = json.loads((root / EXPERIMENT_CONTRACT).read_text(encoding="utf-8"))
    return dict(contract.get("implementation_sha256") or {}) == v25c_implementation_binding(root)


def _real_epoch_replay_smoke(
    root: Path,
    *,
    target_epoch: int,
    target_step: int,
    require_clipped: bool,
) -> dict[str, Any]:
    """Replay one frozen mixed batch and expose only scalar bridge metrics."""

    source = load_v25a_source(root)
    config = source["config"]
    model, optimizer, generator, device = _load_epoch_boundary(
        root=root, source=source, config=config, epoch_before=int(target_epoch) - 1
    )
    dataset = build_known_gate_dataset(root, config, split_name="train")
    loader = build_known_gate_dataloader(
        dataset, batch_size=config.train.batch_size, shuffle=True, generator=generator
    )
    telemetry_rows = _telemetry_for_epoch(source, target_epoch)
    parameters = all_trainable_parameters(model)
    from radar_ln3.known_gate_v25b_gradient_decomposition import module_parameter_indices

    module_indices = module_parameter_indices(model, parameters)
    if tuple(module_indices) != V25C_MODULE_NAMES:
        raise AssertionError("V25c real replay has an incomplete module partition")
    class_weights = torch.tensor(config.train.class_weights, dtype=torch.float32, device=device)
    model.train()
    result: dict[str, Any] | None = None
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
            base_labels = [str(value) for value in batch["base_label_names"]]
            if (
                base_labels.count("ClutterTarget") != 1
                or bool(telemetry["was_globally_clipped"]) is not bool(require_clipped)
            ):
                raise AssertionError("V25c selected real probe lost its mixed/clipping binding")
            terms = parent_loss_terms_batch_weight_sum(
                output["sequence_logits"],
                tensors["labels"],
                tensors["tr_padding_mask"],
                class_weights=class_weights,
                prefix_auxiliary_weight=config.train.prefix_auxiliary_weight,
                prefix_auxiliary_min_events=config.train.prefix_auxiliary_min_events,
            )
            record = _mixed_conflict_record(
                batch=batch,
                telemetry=telemetry,
                terms=terms,
                original_loss=objective["loss"],
                actual_gradient=actual,
                parameters=parameters,
                module_indices=module_indices,
            )
            result = {
                "epoch": int(target_epoch),
                "step": target_step,
                "batch_hash_exact": bool(checks["batch_hash_exact"] == 1.0),
                "pre_clip_norm_absolute_error": float(checks["pre_clip_global_l2_absolute_error"]),
                "clip_decision_exact": bool(checks["clip_decision_exact"] == 1.0),
                "diagnostic_autograd_did_not_mutate_gradients": bool(
                    record["parent_gradient_terms_do_not_mutate_parameter_grad"]
                ),
                "source_to_parent_gradient": record["gradient_reconstruction"],
                "known_component_additivity": record["known_component_additivity"],
                "clutter_component_additivity": record["clutter_component_additivity"],
                "historical_total_conflict": record["component_conflicts"]["historical_total"],
                "source_to_parent_gradient_allowances": record[
                    "gradient_reconstruction_allowances"
                ],
            }
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.gradient_clip_norm)
        optimizer.step()
        if step == target_step:
            break
    if result is None:
        raise AssertionError("V25c real replay did not reach selected probe step")
    if (
        not result["batch_hash_exact"]
        or result["pre_clip_norm_absolute_error"]
        > max(GRADIENT_NORM_ABSOLUTE_TOLERANCE, GRADIENT_NORM_RELATIVE_TOLERANCE * 124.0)
        or not result["clip_decision_exact"]
        or not result["diagnostic_autograd_did_not_mutate_gradients"]
    ):
        raise AssertionError(f"V25c real V25a replay probe failed: {result}")
    _assert_decomposition_gradient_reconstruction(
        result["source_to_parent_gradient"],
        context=(
            f"smoke_epoch{int(target_epoch)}_step{int(target_step)}_source_to_parent"
        ),
    )
    for role in ("known", "clutter"):
        for name, metrics in result[f"{role}_component_additivity"].items():
            _assert_component_additivity(
                metrics,
                context=(
                    f"smoke_epoch{int(target_epoch)}_step{int(target_step)}_{role}_{name}"
                ),
            )
    return result


def _real_epoch7_step83_replay_smoke(root: Path) -> dict[str, Any]:
    return _real_epoch_replay_smoke(
        root,
        target_epoch=7,
        target_step=83,
        require_clipped=True,
    )


def _real_epoch3_step467_replay_smoke(root: Path) -> dict[str, Any]:
    """Probe the largest observed pre-registered additivity bridge excursion."""

    return _real_epoch_replay_smoke(
        root,
        target_epoch=3,
        target_step=467,
        require_clipped=False,
    )


def _run_isolated_real_probe(
    root: Path,
    *,
    target_epoch: int = 7,
    target_step: int = 83,
) -> dict[str, Any]:
    """Run the CUDA replay probe in a fresh process, like full V25c analysis.

    Windows has previously surfaced native CUDA fast-fail exits after a parent
    Python process has already performed unrelated GPU work.  The test itself
    is unchanged; this wrapper simply gives the one real probe the same fresh
    driver context guaranteed for every V25c analysis epoch.
    """

    command = [
        sys.executable,
        "-X",
        "utf8",
        "-m",
        "smoke.known_gate_v25c_contract_test",
        "--project-root",
        str(root),
        "--isolated-real-probe",
        "--probe-epoch",
        str(int(target_epoch)),
        "--probe-step",
        str(int(target_step)),
    ]
    completed = subprocess.run(
        command,
        cwd=root / "src-v3",
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    stdout = str(completed.stdout).strip()
    stderr = str(completed.stderr).strip()
    if completed.returncode != 0:
        raise RuntimeError(
            "V25c isolated real smoke probe failed; "
            f"returncode={completed.returncode}; stdout={stdout[-8000:]}; stderr={stderr[-8000:]}"
        )
    try:
        payload = dict(json.loads(stdout))
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError(
            "V25c isolated real smoke probe returned invalid JSON; "
            f"stdout={stdout[-8000:]}"
        ) from error
    if payload.get("schema_version") != "RADAR-LN3-V25c-known-gate-exact-replay-module-gradient-conflict-localization-real-probe-v1":
        raise RuntimeError("V25c isolated real smoke probe has an invalid identity")
    result = dict(payload.get("result") or {})
    if (
        int(result.get("epoch", -1)) != int(target_epoch)
        or int(result.get("step", -1)) != int(target_step)
    ):
        raise RuntimeError("V25c isolated real smoke probe returned an unexpected target")
    return result


def run(
    project_root: Path,
    *,
    isolated_real_probe: bool = False,
    probe_epoch: int = 7,
    probe_step: int = 83,
) -> dict[str, Any]:
    root = project_root.resolve()
    if not (root / READINESS).is_file() or not (root / EXPERIMENT_CONTRACT).is_file():
        raise FileNotFoundError("V25c prepare artifacts are missing; run prepare first")
    source = load_v25a_source(root)
    contract_path = root / EXPERIMENT_CONTRACT
    readiness_path = root / READINESS
    assert_prepared_v25c_binding(
        root,
        source,
        contract=json.loads(contract_path.read_text(encoding="utf-8")),
        readiness=json.loads(readiness_path.read_text(encoding="utf-8")),
        contract_sha256=sha256_file(contract_path),
    )
    if isolated_real_probe:
        probes = {
            (7, 83): _real_epoch7_step83_replay_smoke,
            (3, 467): _real_epoch3_step467_replay_smoke,
        }
        try:
            probe = probes[(int(probe_epoch), int(probe_step))]
        except KeyError as error:
            raise ValueError("V25c isolated real probe is not registered") from error
        return {
            "schema_version": "RADAR-LN3-V25c-known-gate-exact-replay-module-gradient-conflict-localization-real-probe-v1",
            "result": probe(root),
        }
    synthetic = _synthetic_component_conflict_smoke()
    if not all(synthetic.values()):
        raise AssertionError(f"V25c synthetic component/conflict smoke failed: {synthetic}")
    resource_transport = _resource_transport_smoke(root)
    if not all(resource_transport.values()):
        raise AssertionError(
            f"V25c bounded worker transport smoke failed: {resource_transport}"
        )
    identifiers = _identifier_smoke()
    if not identifiers:
        raise AssertionError("V25c identifier-leak smoke failed")
    prepared_binding = _prepared_binding_smoke(root)
    if not all(prepared_binding.values()):
        raise AssertionError(f"V25c prepared-source binding smoke failed: {prepared_binding}")
    implementation_binding = _implementation_binding_smoke(root)
    if not implementation_binding:
        raise AssertionError("V25c implementation binding smoke failed")
    real = _run_isolated_real_probe(root)
    bridge_edge = _run_isolated_real_probe(root, target_epoch=3, target_step=467)
    edge_metrics = dict(bridge_edge["clutter_component_additivity"])[
        "raw_total_equals_final_plus_prefix"
    ]
    edge_allowances = _assert_component_additivity(
        edge_metrics,
        context="smoke_epoch3_step467_clutter_raw_total_equals_final_plus_prefix",
    )
    if (
        float(edge_metrics["maximum_component_residual_relative_to_actual_peak"])
        <= 5.0e-5
        or float(edge_metrics["maximum_component_residual_relative_to_actual_peak"])
        > float(edge_allowances["allowed_maximum_absolute_component_residual"])
        / max(float(edge_metrics["actual_maximum_absolute_component"]), 1.0e-12)
    ):
        raise AssertionError(
            "V25c registered maximum-component calibration edge is absent or unbounded"
        )
    return {
        "status": "passed",
        "schema_version": "RADAR-LN3-V25c-known-gate-exact-replay-module-gradient-conflict-localization-smoke-v1",
        "synthetic_component_and_module_conflict_metrics": synthetic,
        "bounded_worker_transport": resource_transport,
        "raw_parent_identifiers_rejected_from_persisted_payload": identifiers,
        "prepared_source_binding": prepared_binding,
        "implementation_hash_binding_matches_current_code": implementation_binding,
        "real_v25a_epoch7_step83_replay": real,
        "real_v25a_epoch3_step467_additivity_edge_replay": bridge_edge,
        "epoch3_step467_registered_component_bridge": {
            "observed_relative_to_parent_total_peak": float(
                edge_metrics["maximum_component_residual_relative_to_actual_peak"]
            ),
            "registered_allowed_absolute_component_residual": float(
                edge_allowances["allowed_maximum_absolute_component_residual"]
            ),
            "old_5e-5_is_exceeded": True,
        },
        "validation_dataset_constructed": False,
        "test_payload_access_count": 0,
        "unknown_payload_access_count": 0,
        "checkpoint_written": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke-test V25c V25a exact-replay module-gradient conflict localization."
    )
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument(
        "--isolated-real-probe",
        action="store_true",
        help="Internal: replay the sealed CUDA smoke probe in a fresh process.",
    )
    parser.add_argument("--probe-epoch", type=int, default=7, help=argparse.SUPPRESS)
    parser.add_argument("--probe-step", type=int, default=83, help=argparse.SUPPRESS)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.project_root.resolve(),
                isolated_real_probe=args.isolated_real_probe,
                probe_epoch=args.probe_epoch,
                probe_step=args.probe_step,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
