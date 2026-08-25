"""Prepare the sealed V25c module-gradient conflict localization diagnosis.

V25c is intentionally a read-only analysis of the completed V25a run.  It
does not create a new model, mutate a V25a/V24 checkpoint, construct a held-out
Dataset, or fit a threshold.  Preparation only freezes provenance and code
bindings that the later exact replay must satisfy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from build.known_gate_v25b import (
    EXPECTED_EPOCHS,
    EXPECTED_STEPS_PER_EPOCH,
    EXPECTED_TOTAL_STEPS,
    V25B_IMPLEMENTATION_FILES,
    load_v25a_source,
    v25a_source_binding,
)
from radar_ln3.known_gate import sha256_file
from radar_ln3.known_gate_v25c_module_gradient_conflict import (
    V25C_COMPONENT_NAMES,
    V25C_EXPERIMENT_SCHEMA,
    V25C_MODULE_NAMES,
    V25C_PRIMARY_COMPONENT,
    V25C_READINESS_SCHEMA,
    V25C_SHARED_MODULE_NAMES,
    V25C_VARIANT,
    v25c_component_additivity_tolerance_policy,
    v25c_execution_resource_safety_policy,
)


EXPERIMENT_DOCUMENT = Path(
    "src-v3/docs/radar_ln3_v25c_exact_replay_module_gradient_conflict_localization.md"
)
EXPERIMENT_CONTRACT = Path(
    "data/processed/RADAR-LN3-V25c-cq08-known-gate-exact-replay-module-gradient-"
    "conflict-localization-contract.json"
)
READINESS = Path(
    "src-v3/processed/report/RADAR-LN3-V25c-known-gate-exact-replay-module-gradient-"
    "conflict-localization-readiness.json"
)
REPORT = Path(
    "src-v3/processed/report/RADAR-LN3-V25c-known-gate-exact-replay-module-gradient-"
    "conflict-localization.json"
)
COMPLETION = Path(
    "src-v3/processed/report/RADAR-LN3-V25c-known-gate-exact-replay-module-gradient-"
    "conflict-localization-completion.json"
)

V25C_IMPLEMENTATION_FILES = tuple(V25B_IMPLEMENTATION_FILES) + (
    Path("src-v3/radar_ln3/known_gate_v25c_module_gradient_conflict.py"),
    Path("src-v3/build/known_gate_v25c.py"),
    Path("src-v3/report/known_gate_v25c.py"),
    Path("src-v3/smoke/known_gate_v25c_contract_test.py"),
    Path("src-v3/scripts/v25c_known_gate_module_gradient_conflict_pipeline.ps1"),
    EXPERIMENT_DOCUMENT,
)

EXPECTED_MIXED_KNOWN_CLUTTER_BATCHES = 638
EXPECTED_CLUTTER_ONLY_BATCHES = 11
V25C_SCOPE = (
    "development-only offline exact-gradient replay of frozen V25a train "
    "batches; no training intervention or product-performance claim"
)
V25C_SOLE_VARIABLE = (
    "read-only module-level Known/Clutter parent-gradient conflict localization "
    "for frozen mixed batches after the historical V25a backward pass"
)
V25C_REPLAY_VERIFICATION = {
    "prepared_source_hash_binding_must_match_exactly_before_replay": True,
    "parent_batch_hash_must_match_exactly": True,
    "loss_preclip_norm_and_clip_decision_must_match_with_cuda_tolerances": True,
    "historical_parent_terms_must_reconstruct_loss_backward_gradient": True,
    "raw_and_historical_total_components_must_equal_final_plus_prefix": True,
    "module_partition_must_cover_each_trainable_parameter_once": True,
    "autograd_diagnostic_path_must_not_mutate_parameter_grad": True,
    "source_epoch_model_state_must_match_with_cuda_tolerances": True,
    "each_epoch_is_replayed_in_a_fresh_cuda_worker_context": True,
    "source_gradient_bridge_must_preserve_v25b_parent_call_order_before_v25c_extensions": True,
    "bounded_sidecar_transport_and_compact_completion_are_required": True,
    "failure_policy": "fail_closed_no_report_if_source_replay_is_not_accurate",
}
V25C_PER_MIXED_BATCH_REPORTS = {
    "identifier": "ordered parent batch SHA-256 only",
    "never_persist": [
        "parent_sample_id",
        "track_id",
        "capture_id",
        "sample_id",
        "source_name",
    ],
    "required": [
        "raw_and_historical_final_prefix_and_total_component_gradients",
        "full_model_and_tr_rd_fusion_liquid_classifier_conflict_metrics",
        "negative_projection_and_cancellation_metrics",
        "known_subclass_clipped_rd_same_second_and_course_coverage_strata",
        "exact_gradient_reconstruction_and_component_additivity_residuals",
    ],
}
V25C_DATA_SCOPE = {
    "train_payload_access": "required_for_frozen_replay_only",
    "validation_payload_access_count": 0,
    "test_payload_access_count": 0,
    "unknown_payload_access_count": 0,
    "l3_payload_access_count": 0,
    "full_cascade_payload_access_count": 0,
}
V25C_INTERPRETATION_LIMITS = [
    "Negative parent-gradient cosine alone is not evidence of a harmful optimization conflict; a binary classifier head normally contains competing class gradients.",
    "A module-level association does not prove that class weighting, an RD/TR feature, a capture condition, or the module caused a generalization error.",
    "Only 33 train Clutter parents exist, so mixed-batch and observation strata remain sparse.",
    "V25c produces no checkpoint selection, validation, OOF, test, Unknown, L3, cascade, stable, rejection, or latency claim.",
]
V25C_TERMINAL_STATE = {
    "OOF": "not_executed",
    "test": "not_executed",
    "formal_rejection": "not_executed",
    "full_cascade": "not_executed",
    "stable": "not_executed",
    "P95": "not_executed",
    "deployment_eligible": False,
}
V25C_READINESS_CHECKS = {
    "completed_v25a_source_is_hash_bound": True,
    "all_20_epoch_checkpoints_and_telemetry_files_are_hash_bound": True,
    "prepared_source_hash_binding_is_rechecked_before_replay": True,
    "source_uses_historical_batch_weight_sum_semantics": True,
    "exact_replay_requires_per_step_batch_loss_norm_and_clip_verification": True,
    "historical_parent_gradient_terms_must_reconstruct_loss_backward_gradient": True,
    "final_and_prefix_component_additivity_must_be_checked": True,
    "module_partition_must_cover_every_trainable_parameter_once": True,
    "diagnostic_autograd_must_not_mutate_parameter_grad": True,
    "fresh_cuda_worker_is_required_for_each_frozen_epoch_replay": True,
    "source_gradient_bridge_preserves_v25b_parent_call_order_before_v25c_extensions": True,
    "bounded_worker_sidecar_and_compact_completion_are_required": True,
    "raw_parent_identifiers_are_forbidden_from_persisted_output": True,
    "validation_test_unknown_l3_and_cascade_payload_access_is_zero": True,
    "no_training_or_checkpoint_mutation_is_permitted": True,
}


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _write_new_or_identical(path: Path, payload: Mapping[str, Any]) -> str:
    content = json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(
                "refusing to overwrite a different V25c frozen preparation artifact: "
                f"{path}"
            )
        return "reused_identical"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)
    return "created"


def _read_json(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def v25c_implementation_binding(root: Path) -> dict[str, str]:
    """Hash every transitive file that defines a V25c result."""

    resolved_root = root.resolve()
    missing = [
        _relative(resolved_root, resolved_root / path)
        for path in V25C_IMPLEMENTATION_FILES
        if not (resolved_root / path).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"V25c implementation files are missing: {missing}")
    return {
        _relative(resolved_root, resolved_root / path): sha256_file(resolved_root / path)
        for path in V25C_IMPLEMENTATION_FILES
    }


def _frozen_training_semantics(source: Mapping[str, Any]) -> dict[str, Any]:
    config = source["config"]
    return {
        "model": "V25a self-developed CfCLiquidCausalEncoder",
        "optimizer": "torch.optim.AdamW(model.parameters(), lr=1e-4) defaults",
        "optimizer_weight_decay": 0.01,
        "loss": "parent_final_task_ce_with_prefix_auxiliary",
        "class_weight_normalization": "batch_weight_sum",
        "class_weights": [float(value) for value in config.train.class_weights],
        "prefix_auxiliary_weight": float(config.train.prefix_auxiliary_weight),
        "prefix_auxiliary_min_events": int(config.train.prefix_auxiliary_min_events),
        "gradient_clip_norm": float(config.train.gradient_clip_norm),
        "epochs": len(EXPECTED_EPOCHS),
        "batch_size": int(config.train.batch_size),
        "checkpoint_or_optimizer_not_written": True,
    }


def _diagnostic_protocol() -> dict[str, Any]:
    return {
        "mixed_batch_definition": "exactly one ClutterTarget and one KnownTarget parent in the frozen batch_size=2 source batch",
        "expected_mixed_known_clutter_batch_count": EXPECTED_MIXED_KNOWN_CLUTTER_BATCHES,
        "expected_clutter_only_batch_count": EXPECTED_CLUTTER_ONLY_BATCHES,
        "component_names": list(V25C_COMPONENT_NAMES),
        "primary_component": V25C_PRIMARY_COMPONENT,
        "module_partition": list(V25C_MODULE_NAMES),
        "shared_module_partition": list(V25C_SHARED_MODULE_NAMES),
        "course_coverage_semantics": "non-initial active-event valid fraction; first-event invalidity is expected and is not interpreted as missing course evidence",
        "conflict_metrics": [
            "Known and Clutter L2 norms",
            "dot product and cosine",
            "negative anti-aligned projection by role",
            "summed versus separate gradient L2 cancellation",
        ],
    }


def assert_prepared_v25c_binding(
    root: Path,
    source: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    readiness: Mapping[str, Any],
    contract_sha256: str,
) -> dict[str, Any]:
    """Fail closed unless V25c still binds the exact frozen V25a source."""

    resolved_root = root.resolve()
    expected_source = v25a_source_binding(resolved_root, source)
    expected_contract_path = _relative(resolved_root, resolved_root / EXPERIMENT_CONTRACT)
    expected_document_path = _relative(resolved_root, resolved_root / EXPERIMENT_DOCUMENT)
    expected_document_hash = sha256_file(resolved_root / EXPERIMENT_DOCUMENT)
    expected_implementation = v25c_implementation_binding(resolved_root)
    expected_semantics = _frozen_training_semantics(source)
    expected_protocol = _diagnostic_protocol()
    readiness_contract = dict(readiness.get("experiment_contract") or {})
    readiness_checks = dict(readiness.get("checks") or {})
    invalid = (
        contract.get("schema_version") != V25C_EXPERIMENT_SCHEMA
        or contract.get("internal_variant") != V25C_VARIANT
        or contract.get("experiment_document") != expected_document_path
        or contract.get("experiment_document_sha256") != expected_document_hash
        or dict(contract.get("implementation_sha256") or {}) != expected_implementation
        or contract.get("scope") != V25C_SCOPE
        or contract.get("sole_variable") != V25C_SOLE_VARIABLE
        or dict(contract.get("source_v25a") or {}) != expected_source
        or dict(contract.get("frozen_training_semantics") or {}) != expected_semantics
        or dict(contract.get("diagnostic_protocol") or {}) != expected_protocol
        or dict(contract.get("replay_verification") or {}) != V25C_REPLAY_VERIFICATION
        or dict(contract.get("execution_resource_safety_policy") or {})
        != v25c_execution_resource_safety_policy()
        or dict(contract.get("component_additivity_tolerance_policy") or {})
        != v25c_component_additivity_tolerance_policy()
        or dict(contract.get("per_mixed_batch_reports") or {}) != V25C_PER_MIXED_BATCH_REPORTS
        or dict(contract.get("data_scope") or {}) != V25C_DATA_SCOPE
        or list(contract.get("interpretation_limits") or ()) != V25C_INTERPRETATION_LIMITS
        or any(contract.get(key) != value for key, value in V25C_TERMINAL_STATE.items())
        or readiness.get("status") != "passed"
        or readiness.get("schema_version") != V25C_READINESS_SCHEMA
        or readiness.get("variant") != V25C_VARIANT
        or readiness_contract
        != {"path": expected_contract_path, "sha256": str(contract_sha256)}
        or readiness_checks != V25C_READINESS_CHECKS
        or dict(readiness.get("source_v25a") or {}) != expected_source
        or dict(readiness.get("diagnostic_protocol") or {}) != expected_protocol
        or dict(readiness.get("execution_resource_safety_policy") or {})
        != v25c_execution_resource_safety_policy()
        or list(readiness.get("expected_epochs") or ()) != list(EXPECTED_EPOCHS)
        or int(readiness.get("expected_steps_per_epoch", -1))
        != EXPECTED_STEPS_PER_EPOCH
        or int(readiness.get("expected_total_steps", -1)) != EXPECTED_TOTAL_STEPS
        or readiness.get("validation_used_for_readiness_gate") is not False
        or int(readiness.get("validation_payload_access_count", -1)) != 0
        or int(readiness.get("test_payload_access_count", -1)) != 0
        or int(readiness.get("unknown_payload_access_count", -1)) != 0
        or readiness.get("formal_training") != "not_executed"
        or any(readiness.get(key) != value for key, value in V25C_TERMINAL_STATE.items())
    )
    if invalid:
        raise ValueError(
            "V25c prepared contract/readiness no longer binds the exact frozen V25a source"
        )
    return expected_source


def build(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    source = load_v25a_source(root)
    contract = {
        "schema_version": V25C_EXPERIMENT_SCHEMA,
        "experiment_name": "V25c V25a exact replay module-gradient conflict localization",
        "internal_variant": V25C_VARIANT,
        "experiment_document": _relative(root, root / EXPERIMENT_DOCUMENT),
        "experiment_document_sha256": sha256_file(root / EXPERIMENT_DOCUMENT),
        "implementation_sha256": v25c_implementation_binding(root),
        "scope": V25C_SCOPE,
        "sole_variable": V25C_SOLE_VARIABLE,
        "source_v25a": v25a_source_binding(root, source),
        "frozen_training_semantics": _frozen_training_semantics(source),
        "diagnostic_protocol": _diagnostic_protocol(),
        "replay_verification": V25C_REPLAY_VERIFICATION,
        "execution_resource_safety_policy": v25c_execution_resource_safety_policy(),
        "component_additivity_tolerance_policy": v25c_component_additivity_tolerance_policy(),
        "per_mixed_batch_reports": V25C_PER_MIXED_BATCH_REPORTS,
        "data_scope": V25C_DATA_SCOPE,
        "interpretation_limits": V25C_INTERPRETATION_LIMITS,
        **V25C_TERMINAL_STATE,
    }
    _write_new_or_identical(root / EXPERIMENT_CONTRACT, contract)
    contract_hash = sha256_file(root / EXPERIMENT_CONTRACT)
    checks = dict(V25C_READINESS_CHECKS)
    readiness = {
        "status": "passed" if all(checks.values()) else "failed",
        "schema_version": V25C_READINESS_SCHEMA,
        "variant": V25C_VARIANT,
        "checks": checks,
        "experiment_contract": {
            "path": _relative(root, root / EXPERIMENT_CONTRACT),
            "sha256": contract_hash,
        },
        "source_v25a": v25a_source_binding(root, source),
        "diagnostic_protocol": _diagnostic_protocol(),
        "execution_resource_safety_policy": v25c_execution_resource_safety_policy(),
        "expected_epochs": list(EXPECTED_EPOCHS),
        "expected_steps_per_epoch": EXPECTED_STEPS_PER_EPOCH,
        "expected_total_steps": EXPECTED_TOTAL_STEPS,
        "validation_used_for_readiness_gate": False,
        "validation_payload_access_count": 0,
        "test_payload_access_count": 0,
        "unknown_payload_access_count": 0,
        "formal_training": "not_executed",
        **V25C_TERMINAL_STATE,
    }
    _write_new_or_identical(root / READINESS, readiness)
    return {
        "status": readiness["status"],
        "schema_version": V25C_READINESS_SCHEMA,
        "variant": V25C_VARIANT,
        "contract": str(root / EXPERIMENT_CONTRACT),
        "readiness": str(root / READINESS),
        "source_v25a_summary": str(source["paths"]["summary"]),
        "expected_total_steps": EXPECTED_TOTAL_STEPS,
        "validation": "not_accessed",
        "test": "not_executed",
    }


def check_prepared_binding(project_root: Path) -> dict[str, Any]:
    """Validate an existing V25c preparation without rewriting any artifact.

    The pipeline uses this narrow mode to distinguish a byte-stale pre-analysis
    contract from a current one before it decides whether a safe archival and
    regeneration is needed.  It intentionally neither creates nor updates a
    file.
    """

    root = project_root.resolve()
    contract_path = root / EXPERIMENT_CONTRACT
    readiness_path = root / READINESS
    if not contract_path.is_file() or not readiness_path.is_file():
        raise FileNotFoundError("V25c preparation artifacts are missing")
    source = load_v25a_source(root)
    assert_prepared_v25c_binding(
        root,
        source,
        contract=_read_json(contract_path),
        readiness=_read_json(readiness_path),
        contract_sha256=sha256_file(contract_path),
    )
    return {
        "status": "prepared_binding_current",
        "schema_version": V25C_READINESS_SCHEMA,
        "variant": V25C_VARIANT,
        "contract": str(contract_path),
        "readiness": str(readiness_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare V25c V25a exact-replay module-gradient conflict diagnostics."
    )
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument(
        "--check-prepared-binding",
        action="store_true",
        help="Validate existing preparation only; never create or rewrite an artifact.",
    )
    args = parser.parse_args()
    result = (
        check_prepared_binding(args.project_root.resolve())
        if args.check_prepared_binding
        else build(args.project_root.resolve())
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
