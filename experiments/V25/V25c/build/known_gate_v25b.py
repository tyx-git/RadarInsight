"""Bind V25b to the completed V25a exact-replay source artifact.

V25b deliberately has no training configuration or output checkpoint.  Its
sole executable stage is a development-only replay analysis of V25a's frozen
training updates.  Preparation writes a provenance contract and readiness
record; it neither creates a model artifact nor reads validation/test data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from radar_ln3.known_gate import load_known_gate_config, sha256_file
from radar_ln3.known_gate_gradient_telemetry import (
    V25A_GRADIENT_TELEMETRY_DIRECTORY,
    V25A_GRADIENT_TELEMETRY_SCHEMA,
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
    Path("experiments/V25/V25c/train/known_gate_common.py"),
    Path("experiments/V25/V25c/train/known_gate_v25a.py"),
    Path("experiments/V25/V25c/build/known_gate_v25b.py"),
    Path(),
    Path(),
    Path("src-v3/docs/radar_ln3_v25b_exact_replay_parent_gradient_decomposition.md"),
)

EXPECTED_EPOCHS = tuple(range(1, 21))
EXPECTED_STEPS_PER_EPOCH = 493
EXPECTED_TOTAL_STEPS = len(EXPECTED_EPOCHS) * EXPECTED_STEPS_PER_EPOCH


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





