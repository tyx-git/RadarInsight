"""Bind V25b to the completed V25a exact-replay source artifact.

V25b deliberately has no training configuration or output checkpoint.  Its
sole executable stage is a development-only replay analysis of V25a's frozen
training updates.  Preparation writes a provenance contract and readiness
record; it neither creates a model artifact nor reads validation/test data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from radar_ln3.known_gate import load_known_gate_config, sha256_file
from radar_ln3.known_gate_gradient_telemetry import (
    V25A_GRADIENT_TELEMETRY_DIRECTORY,
    V25A_GRADIENT_TELEMETRY_SCHEMA,
)
from radar_ln3.known_gate_v25b_gradient_decomposition import (
    V25B_EXPERIMENT_SCHEMA,
    V25B_READINESS_SCHEMA,
    V25B_VARIANT,
    v25b_decomposition_tolerance_policy,
)
from train.known_gate_v25a import _require_v24_equivalent_config


SOURCE_CONFIG = Path(
    "config/RADAR-LN3/RADAR-LN3-V25a-cq08-liquid-known-gate-course-preclip-"
    "gradient-telemetry-seed43-config.json"
)
SOURCE_CONTRACT = Path(
    "data/processed/RADAR-LN3-V25a-cq08-known-gate-course-preclip-gradient-"
    "telemetry-contract.json"
)
SOURCE_READINESS = Path(
    "src-v3/processed/report/RADAR-LN3-V25a-known-gate-course-preclip-gradient-"
    "telemetry-training-readiness.json"
)
SOURCE_ARTIFACT = Path(
    "artifacts/RADAR-LN3-V25a-cq08-liquid-known-gate-course-preclip-gradient-"
    "telemetry-seed43-formal"
)
SOURCE_SUMMARY = SOURCE_ARTIFACT / "training-summary.json"

V25B_IMPLEMENTATION_FILES = (
    Path("src-v3/radar_ln3/features.py"),
    Path("src-v3/radar_ln3/micro_doppler.py"),
    Path("src-v3/radar_ln3/rd_physical.py"),
    Path("src-v3/radar_ln3/losses.py"),
    Path("src-v3/radar_ln3/model.py"),
    Path("src-v3/radar_ln3/known_gate.py"),
    Path("src-v3/radar_ln3/known_gate_features.py"),
    Path("src-v3/radar_ln3/known_gate_gradient_telemetry.py"),
    Path("src-v3/radar_ln3/data.py"),
    Path("src-v3/radar_ln3/open_set_architecture.py"),
    Path("src-v3/radar_ln3/logging_utils.py"),
    Path("src-v3/radar_ln3/known_gate_v25b_gradient_decomposition.py"),
    Path("experiments/V25/V25b/train/known_gate_common.py"),
    Path("experiments/V25/V25b/train/known_gate_v25a.py"),
    Path("experiments/V25/V25b/build/known_gate_v25b.py"),
    Path("experiments/V25/V25b/report/known_gate_v25b.py"),
    Path("experiments/V25/V25b/scripts/v25b_known_gate_gradient_decomposition_pipeline.ps1"),
    Path("src-v3/docs/radar_ln3_v25b_exact_replay_parent_gradient_decomposition.md"),
)
EXPERIMENT_DOCUMENT = V25B_IMPLEMENTATION_FILES[-1]
EXPERIMENT_CONTRACT = Path(
    "data/processed/RADAR-LN3-V25b-cq08-known-gate-exact-replay-parent-gradient-"
    "decomposition-contract.json"
)
READINESS = Path(
    "src-v3/processed/report/RADAR-LN3-V25b-known-gate-exact-replay-parent-"
    "gradient-decomposition-readiness.json"
)
REPORT = Path(
    "src-v3/processed/report/RADAR-LN3-V25b-known-gate-exact-replay-parent-"
    "gradient-decomposition.json"
)

EXPECTED_EPOCHS = tuple(range(1, 21))
EXPECTED_STEPS_PER_EPOCH = 493
EXPECTED_TOTAL_STEPS = len(EXPECTED_EPOCHS) * EXPECTED_STEPS_PER_EPOCH
V25B_SCOPE = (
    "development-only offline exact-gradient replay of frozen V25a train "
    "batches; no training intervention or product-performance claim"
)
V25B_SOLE_VARIABLE = (
    "read-only per-parent final/prefix gradient decomposition after the "
    "historical V25a batch loss has already been backpropagated"
)
V25B_REPLAY_VERIFICATION = {
    "prepared_source_hash_binding_must_match_exactly_before_replay": True,
    "parent_batch_hash_must_match_exactly": True,
    "loss_preclip_norm_and_clip_decision_must_match_with_cuda_tolerances": True,
    "per_parent_historical_gradient_terms_must_reconstruct_loss_backward_gradient": True,
    "fused_loss_to_explicit_proxy_and_proxy_to_parent_sum_bridges_must_each_pass": True,
    "cuda_fused_reduction_reconstruction_uses_registered_absolute_relative_and_component_tolerances": True,
    "autograd_diagnostic_path_must_not_mutate_parameter_grad": True,
    "source_epoch_model_state_must_match_with_cuda_tolerances": True,
    "each_epoch_is_replayed_in_a_fresh_cuda_worker_context": True,
    "failure_policy": "fail_closed_no_report_if_source_replay_is_not_accurate",
}
V25B_PER_CLUTTER_BATCH_REPORTS = {
    "identifier": "ordered parent batch SHA-256 only",
    "never_persist": [
        "parent_sample_id",
        "track_id",
        "capture_id",
        "sample_id",
        "source_name",
    ],
    "required": [
        "raw_known_and_clutter_gradient_norms",
        "historical_weighted_gradient_norms",
        "dot_product_and_cosine",
        "final_and_prefix_contributions",
        "module_breakdown",
        "exact_gradient_reconstruction_residual",
        "batch_observation_metadata",
        "clipped_vs_not_clipped_and_base_label_strata",
    ],
}
V25B_DATA_SCOPE = {
    "train_payload_access": "required_for_frozen_replay_only",
    "validation_payload_access_count": 0,
    "test_payload_access_count": 0,
    "unknown_payload_access_count": 0,
    "l3_payload_access_count": 0,
    "full_cascade_payload_access_count": 0,
}
V25B_INTERPRETATION_LIMITS = [
    "A directional gradient association does not establish a causal source of a spike.",
    "The diagnostic preserves V25a historical class weighting; it does not test a remedy.",
    "Only 33 train Clutter parents exist, so Clutter-containing strata remain sparse.",
    "V25b produces no checkpoint selection, validation, OOF, test, Unknown, L3, cascade, stable, rejection, or latency claim.",
]
V25B_TERMINAL_STATE = {
    "OOF": "not_executed",
    "test": "not_executed",
    "formal_rejection": "not_executed",
    "full_cascade": "not_executed",
    "stable": "not_executed",
    "P95": "not_executed",
    "deployment_eligible": False,
}
V25B_READINESS_CHECKS = {
    "completed_v25a_source_is_hash_bound": True,
    "all_20_epoch_checkpoints_and_telemetry_files_are_hash_bound": True,
    "prepared_source_hash_binding_is_rechecked_before_replay": True,
    "source_uses_historical_batch_weight_sum_semantics": True,
    "exact_replay_requires_per_step_batch_loss_norm_and_clip_verification": True,
    "diagnostic_gradient_terms_must_reconstruct_loss_backward_gradient": True,
    "all_fused_proxy_parent_gradient_bridges_are_independently_verified": True,
    "cuda_fused_reduction_reconstruction_tolerance_is_explicit_and_bounded": True,
    "diagnostic_autograd_must_not_mutate_parameter_grad": True,
    "fresh_cuda_worker_is_required_for_each_frozen_epoch_replay": True,
    "raw_parent_identifiers_are_forbidden_from_persisted_output": True,
    "validation_test_unknown_l3_and_cascade_payload_access_is_zero": True,
    "no_training_or_checkpoint_mutation_is_permitted": True,
}


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _read_json(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _same_serialized_path(
    value: Any,
    expected: Path,
    *,
    root: Path,
) -> bool:
    """Compare a serialized absolute or root-relative path safely.

    V25a persists some records as absolute Windows paths and its readiness
    contract path as a project-root-relative POSIX path.  Resolving the latter
    against the process CWD would turn it into ``src-v3/data/...`` during the
    module invocation, so explicitly anchor relative values at the project
    root instead.
    """

    try:
        candidate = Path(str(value))
        if not candidate.is_absolute():
            candidate = root.resolve() / candidate
        return candidate.resolve() == expected.resolve()
    except (OSError, ValueError, TypeError):
        return False


def _write_new_or_identical(path: Path, payload: Mapping[str, Any]) -> str:
    content = json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(
                "refusing to overwrite a different V25b frozen preparation artifact: "
                f"{path}"
            )
        return "reused_identical"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)
    return "created"


def _source_epoch_paths(root: Path) -> dict[str, Path]:
    artifact = root / SOURCE_ARTIFACT
    result: dict[str, Path] = {
        "config": root / SOURCE_CONFIG,
        "contract": root / SOURCE_CONTRACT,
        "readiness": root / SOURCE_READINESS,
        "summary": root / SOURCE_SUMMARY,
        "telemetry_metadata": artifact / V25A_GRADIENT_TELEMETRY_DIRECTORY / "metadata.json",
    }
    for epoch in EXPECTED_EPOCHS:
        result[f"checkpoint_epoch_{epoch:03d}"] = (
            artifact / "epochs" / f"epoch-{epoch:03d}.pt"
        )
        result[f"telemetry_epoch_{epoch:03d}"] = (
            artifact
            / V25A_GRADIENT_TELEMETRY_DIRECTORY
            / f"epoch-{epoch:03d}.json"
        )
    return result


def v25b_implementation_binding(root: Path) -> dict[str, str]:
    """Hash all code and protocol files that define a V25b replay result."""

    resolved_root = root.resolve()
    missing = [
        _relative(resolved_root, resolved_root / path)
        for path in V25B_IMPLEMENTATION_FILES
        if not (resolved_root / path).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"V25b implementation files are missing: {missing}")
    return {
        _relative(resolved_root, resolved_root / path): sha256_file(resolved_root / path)
        for path in V25B_IMPLEMENTATION_FILES
    }


def v25a_source_binding(root: Path, source: Mapping[str, Any]) -> dict[str, Any]:
    """Return the complete immutable V25a identity consumed by V25b.

    This is deliberately a value object, rather than a collection of partial
    checks at the call site.  Preparation serializes it into both V25b
    artifacts and the analysis stage compares it again before it reads a
    single train batch.  Thus an internally valid, but later replaced, V25a
    run cannot silently become the source of a V25b diagnosis.
    """

    resolved_root = root.resolve()
    paths = dict(source["paths"])
    hashes = dict(source["hashes"])
    summary = dict(source["summary"])
    return {
        "config": _relative(resolved_root, paths["config"]),
        "config_sha256": str(hashes["config"]),
        "contract": _relative(resolved_root, paths["contract"]),
        "contract_sha256": str(hashes["contract"]),
        "readiness": _relative(resolved_root, paths["readiness"]),
        "readiness_sha256": str(hashes["readiness"]),
        "training_summary": _relative(resolved_root, paths["summary"]),
        "training_summary_sha256": str(hashes["summary"]),
        "telemetry_metadata": _relative(resolved_root, paths["telemetry_metadata"]),
        "telemetry_metadata_sha256": str(hashes["telemetry_metadata"]),
        "checkpoint_epoch_sha256": {
            str(epoch): str(value)
            for epoch, value in dict(source["checkpoint_hashes"]).items()
        },
        "telemetry_epoch_sha256": {
            str(epoch): str(value)
            for epoch, value in dict(source["telemetry_hashes"]).items()
        },
        "parent_order_sha256": {
            str(epoch): str(value)
            for epoch, value in dict(source["parent_order_hashes"]).items()
        },
        "initial_model_state_sha256": str(summary.get("initial_model_state_sha256", "")),
        "best_checkpoint_epoch": int(summary["best_epoch"]),
    }


def assert_prepared_v25b_binding(
    root: Path,
    source: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    readiness: Mapping[str, Any],
    contract_sha256: str,
) -> dict[str, Any]:
    """Fail closed unless V25b preparation still binds this exact V25a source.

    ``load_v25a_source`` proves that the *current* source is a self-consistent
    completed V25a artifact.  That alone is insufficient for a frozen V25b
    diagnostic: a later V25a replacement could also be self-consistent.  This
    check ties the source to the hashes captured during V25b preparation.
    """

    resolved_root = root.resolve()
    expected_source = v25a_source_binding(resolved_root, source)
    expected_contract_path = _relative(resolved_root, resolved_root / EXPERIMENT_CONTRACT)
    expected_document_path = _relative(resolved_root, resolved_root / EXPERIMENT_DOCUMENT)
    expected_document_hash = sha256_file(resolved_root / EXPERIMENT_DOCUMENT)
    expected_implementation = v25b_implementation_binding(resolved_root)
    config = source["config"]
    expected_semantics = {
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
    readiness_contract = dict(readiness.get("experiment_contract") or {})
    readiness_checks = dict(readiness.get("checks") or {})
    invalid = (
        contract.get("schema_version") != V25B_EXPERIMENT_SCHEMA
        or contract.get("internal_variant") != V25B_VARIANT
        or contract.get("experiment_document") != expected_document_path
        or contract.get("experiment_document_sha256") != expected_document_hash
        or dict(contract.get("implementation_sha256") or {}) != expected_implementation
        or contract.get("scope") != V25B_SCOPE
        or contract.get("sole_variable") != V25B_SOLE_VARIABLE
        or dict(contract.get("source_v25a") or {}) != expected_source
        or dict(contract.get("frozen_training_semantics") or {}) != expected_semantics
        or dict(contract.get("replay_verification") or {}) != V25B_REPLAY_VERIFICATION
        or dict(contract.get("gradient_reconstruction_tolerance_policy") or {})
        != v25b_decomposition_tolerance_policy()
        or dict(contract.get("per_clutter_batch_reports") or {})
        != V25B_PER_CLUTTER_BATCH_REPORTS
        or dict(contract.get("data_scope") or {}) != V25B_DATA_SCOPE
        or list(contract.get("interpretation_limits") or ()) != V25B_INTERPRETATION_LIMITS
        or any(contract.get(key) != value for key, value in V25B_TERMINAL_STATE.items())
        or readiness.get("status") != "passed"
        or readiness.get("schema_version") != V25B_READINESS_SCHEMA
        or readiness.get("variant") != V25B_VARIANT
        or readiness_contract
        != {"path": expected_contract_path, "sha256": str(contract_sha256)}
        or readiness_checks != V25B_READINESS_CHECKS
        or dict(readiness.get("source_v25a") or {}) != expected_source
        or list(readiness.get("expected_epochs") or ()) != list(EXPECTED_EPOCHS)
        or int(readiness.get("expected_steps_per_epoch", -1))
        != EXPECTED_STEPS_PER_EPOCH
        or int(readiness.get("expected_total_steps", -1)) != EXPECTED_TOTAL_STEPS
        or readiness.get("validation_used_for_readiness_gate") is not False
        or int(readiness.get("validation_payload_access_count", -1)) != 0
        or int(readiness.get("test_payload_access_count", -1)) != 0
        or int(readiness.get("unknown_payload_access_count", -1)) != 0
        or readiness.get("formal_training") != "not_executed"
        or any(readiness.get(key) != value for key, value in V25B_TERMINAL_STATE.items())
    )
    if invalid:
        raise ValueError(
            "V25b prepared contract/readiness no longer binds the exact frozen V25a source"
        )
    return expected_source


def load_v25a_source(root: Path) -> dict[str, Any]:
    """Validate only immutable V25a source objects needed for V25b replay.

    No validation Dataset, test Dataset, Unknown parent, L3 object, or product
    evaluator is constructed here.  The source V25a summary is read only to
    bind the update history and telemetry records.
    """

    root = root.resolve()
    paths = _source_epoch_paths(root)
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"V25b frozen V25a source files are missing: {missing}")
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    source_config = load_known_gate_config(paths["config"])
    _require_v24_equivalent_config(source_config)
    source_contract = _read_json(paths["contract"])
    source_readiness = _read_json(paths["readiness"])
    summary = _read_json(paths["summary"])
    metadata = _read_json(paths["telemetry_metadata"])
    if (
        source_config.schema_version != "RADAR-LN3-V25a"
        or source_config.variant != "v25a_course_preclip_gradient_telemetry"
        or source_config.train.epochs != 20
        or source_config.train.batch_size != 2
        or source_config.train.seed != 43
        or source_config.train.dataloader_seed != 43
        or source_config.train.class_weight_normalization != "batch_weight_sum"
        or tuple(float(value) for value in source_config.train.class_weights)
        != (0.5173137460650578, 14.93939393939394)
        or source_config.train.prefix_auxiliary_weight != 0.1
        or source_config.train.prefix_auxiliary_min_events != 3
        or source_readiness.get("status") != "passed"
        or source_readiness.get("schema_version")
        != "RADAR-LN3-V25a-known-gate-course-preclip-gradient-telemetry-training-readiness-v1"
        or source_readiness.get("variant") != source_config.variant
        or dict(source_readiness.get("config") or {}).get("path")
        != SOURCE_CONFIG.as_posix()
        or dict(source_readiness.get("config") or {}).get("sha256")
        != hashes["config"]
        or not _same_serialized_path(
            source_readiness.get("contract_path"), paths["contract"], root=root
        )
        or source_readiness.get("contract_sha256") != hashes["contract"]
        or source_contract.get("internal_variant") != source_config.variant
        or source_contract.get("schema_version")
        != "RADAR-LN3-V25a-known-gate-course-preclip-gradient-telemetry-experiment-contract-v1"
        or summary.get("status") != "completed"
        or summary.get("variant") != source_config.variant
        or int(summary.get("global_step", -1)) != EXPECTED_TOTAL_STEPS
        or int(summary.get("best_epoch", -1)) not in EXPECTED_EPOCHS
        or len(list(summary.get("history") or ())) != len(EXPECTED_EPOCHS)
        or not _same_serialized_path(
            dict(summary.get("training_readiness") or {}).get("path"),
            paths["readiness"],
            root=root,
        )
        or dict(summary.get("training_readiness") or {}).get("sha256")
        != hashes["readiness"]
        or dict(summary.get("training_readiness") or {}).get("status") != "passed"
        or summary.get("test_evaluation") != "not_executed"
        or summary.get("oof_threshold_calibration") != "not_executed"
        or summary.get("full_cascade_evaluation") != "not_executed"
        or summary.get("formal_rejection") != "not_executed"
        or summary.get("stable_evaluation") != "not_executed"
        or summary.get("p95_latency") != "not_executed"
        or summary.get("deployment_eligible") is not False
        or metadata.get("schema_version") != V25A_GRADIENT_TELEMETRY_SCHEMA
        or metadata.get("config_file_sha256") != hashes["config"]
        or metadata.get("initial_model_state_sha256")
        != summary.get("initial_model_state_sha256")
        or int(metadata.get("steps_per_epoch", -1)) != EXPECTED_STEPS_PER_EPOCH
        or metadata.get("raw_parent_sample_ids_stored") is True
        or dict(metadata.get("gradient_semantics") or {}).get("raw_parent_sample_ids_stored")
        is not False
        or any(
            dict(metadata.get("gradient_semantics") or {}).get(field) is not False
            for field in (
                "telemetry_changes_loss",
                "telemetry_changes_gradients",
                "telemetry_changes_optimizer",
                "telemetry_changes_data_order",
            )
        )
        or dict(summary.get("gradient_telemetry") or {}).get("schema_version")
        != V25A_GRADIENT_TELEMETRY_SCHEMA
        or dict(summary.get("gradient_telemetry") or {}).get("metadata_sha256")
        != hashes["telemetry_metadata"]
        or not _same_serialized_path(
            dict(summary.get("gradient_telemetry") or {}).get("metadata_path"),
            paths["telemetry_metadata"],
            root=root,
        )
        or int(dict(summary.get("gradient_telemetry") or {}).get("epoch_file_count", -1))
        != len(EXPECTED_EPOCHS)
        or int(dict(summary.get("gradient_telemetry") or {}).get("record_count", -1))
        != EXPECTED_TOTAL_STEPS
        or int(dict(summary.get("gradient_telemetry") or {}).get("steps_per_epoch", -1))
        != EXPECTED_STEPS_PER_EPOCH
    ):
        raise ValueError("V25b V25a source contract is stale or outside sealed scope")
    history = [dict(row) for row in list(summary["history"])]
    if [int(row.get("epoch", -1)) for row in history] != list(EXPECTED_EPOCHS):
        raise ValueError("V25b V25a source history lacks the full ordered 20 epochs")
    telemetry_hashes: dict[str, str] = {}
    checkpoint_hashes: dict[str, str] = {}
    parent_order_hashes: dict[str, str] = {}
    for epoch, history_row in zip(EXPECTED_EPOCHS, history, strict=True):
        telemetry_key = f"telemetry_epoch_{epoch:03d}"
        checkpoint_key = f"checkpoint_epoch_{epoch:03d}"
        descriptor = dict(history_row.get("gradient_telemetry") or {})
        telemetry_path = paths[telemetry_key]
        telemetry = _read_json(telemetry_path)
        expected_global_end = epoch * EXPECTED_STEPS_PER_EPOCH
        if (
            int(history_row.get("global_step", -1)) != expected_global_end
            or int(history_row.get("train_steps", -1)) != EXPECTED_STEPS_PER_EPOCH
            or str(descriptor.get("path", "")) != str(telemetry_path)
            or str(descriptor.get("sha256", "")) != hashes[telemetry_key]
            or int(descriptor.get("epoch", -1)) != epoch
            or int(descriptor.get("record_count", -1)) != EXPECTED_STEPS_PER_EPOCH
            or telemetry.get("schema_version") != V25A_GRADIENT_TELEMETRY_SCHEMA
            or int(telemetry.get("epoch", -1)) != epoch
            or int(telemetry.get("step_count", -1)) != EXPECTED_STEPS_PER_EPOCH
            or int(telemetry.get("global_step_start", -1))
            != (epoch - 1) * EXPECTED_STEPS_PER_EPOCH + 1
            or int(telemetry.get("global_step_end", -1)) != expected_global_end
            or len(list(telemetry.get("records") or ())) != EXPECTED_STEPS_PER_EPOCH
            or not str(history_row.get("parent_order_sha256", ""))
            or str(telemetry.get("parent_order_sha256", ""))
            != str(history_row.get("parent_order_sha256", ""))
        ):
            raise ValueError(f"V25b V25a source epoch binding is invalid: {epoch}")
        telemetry_hashes[str(epoch)] = hashes[telemetry_key]
        checkpoint_hashes[str(epoch)] = hashes[checkpoint_key]
        parent_order_hashes[str(epoch)] = str(history_row["parent_order_sha256"])
    if dict(metadata.get("expected_parent_order_hashes") or {}) != parent_order_hashes:
        raise ValueError("V25b V25a telemetry metadata parent order differs from history")
    return {
        "paths": paths,
        "hashes": hashes,
        "config": source_config,
        "contract": source_contract,
        "readiness": source_readiness,
        "summary": summary,
        "metadata": metadata,
        "telemetry_hashes": telemetry_hashes,
        "checkpoint_hashes": checkpoint_hashes,
        "parent_order_hashes": parent_order_hashes,
    }


def build(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    source = load_v25a_source(root)
    paths = dict(source["paths"])
    hashes = dict(source["hashes"])
    config = source["config"]
    contract = {
        "schema_version": V25B_EXPERIMENT_SCHEMA,
        "experiment_name": "V25b V25a exact replay parent-gradient decomposition",
        "internal_variant": V25B_VARIANT,
        "experiment_document": _relative(root, root / EXPERIMENT_DOCUMENT),
        "experiment_document_sha256": sha256_file(root / EXPERIMENT_DOCUMENT),
        "implementation_sha256": v25b_implementation_binding(root),
        "scope": V25B_SCOPE,
        "sole_variable": V25B_SOLE_VARIABLE,
        "source_v25a": v25a_source_binding(root, source),
        "frozen_training_semantics": {
            "model": "V25a self-developed CfCLiquidCausalEncoder",
            "optimizer": "torch.optim.AdamW(model.parameters(), lr=1e-4) defaults",
            "optimizer_weight_decay": 0.01,
            "loss": "parent_final_task_ce_with_prefix_auxiliary",
            "class_weight_normalization": "batch_weight_sum",
            "class_weights": [float(value) for value in config.train.class_weights],
            "prefix_auxiliary_weight": float(config.train.prefix_auxiliary_weight),
            "prefix_auxiliary_min_events": int(config.train.prefix_auxiliary_min_events),
            "gradient_clip_norm": float(config.train.gradient_clip_norm),
            "epochs": 20,
            "batch_size": 2,
            "checkpoint_or_optimizer_not_written": True,
        },
        "replay_verification": V25B_REPLAY_VERIFICATION,
        "gradient_reconstruction_tolerance_policy": v25b_decomposition_tolerance_policy(),
        "per_clutter_batch_reports": V25B_PER_CLUTTER_BATCH_REPORTS,
        "data_scope": V25B_DATA_SCOPE,
        "interpretation_limits": V25B_INTERPRETATION_LIMITS,
        **V25B_TERMINAL_STATE,
    }
    _write_new_or_identical(root / EXPERIMENT_CONTRACT, contract)
    contract_hash = sha256_file(root / EXPERIMENT_CONTRACT)
    checks = dict(V25B_READINESS_CHECKS)
    readiness = {
        "status": "passed" if all(checks.values()) else "failed",
        "schema_version": V25B_READINESS_SCHEMA,
        "variant": V25B_VARIANT,
        "checks": checks,
        "experiment_contract": {
            "path": _relative(root, root / EXPERIMENT_CONTRACT),
            "sha256": contract_hash,
        },
        "source_v25a": v25a_source_binding(root, source),
        "expected_epochs": list(EXPECTED_EPOCHS),
        "expected_steps_per_epoch": EXPECTED_STEPS_PER_EPOCH,
        "expected_total_steps": EXPECTED_TOTAL_STEPS,
        "validation_used_for_readiness_gate": False,
        "validation_payload_access_count": 0,
        "test_payload_access_count": 0,
        "unknown_payload_access_count": 0,
        "formal_training": "not_executed",
        **V25B_TERMINAL_STATE,
    }
    _write_new_or_identical(root / READINESS, readiness)
    return {
        "status": readiness["status"],
        "schema_version": V25B_READINESS_SCHEMA,
        "variant": V25B_VARIANT,
        "contract": str(root / EXPERIMENT_CONTRACT),
        "readiness": str(root / READINESS),
        "source_v25a_summary": str(paths["summary"]),
        "expected_total_steps": EXPECTED_TOTAL_STEPS,
        "validation": "not_accessed",
        "test": "not_executed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare V25b's sealed V25a exact-replay gradient diagnostic."
    )
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    args = parser.parse_args()
    print(json.dumps(build(args.project_root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
