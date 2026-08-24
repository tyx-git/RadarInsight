"""Development-only audit report for V25a pre-clip gradient telemetry.

The report deliberately aggregates only the immutable training telemetry.  It
does not tune a checkpoint, access a test split, or make a product decision.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from build.known_gate_v25a import ARTIFACT, CONFIG, EXPERIMENT_CONTRACT, READINESS
from radar_ln3.known_gate import (
    KNOWN_GATE_V25A_READINESS_SCHEMA,
    V25A_GRADIENT_TELEMETRY_VARIANT,
    load_known_gate_config,
    sha256_file,
)
from radar_ln3.known_gate_gradient_telemetry import (
    V25A_BASE_LABELS,
    V25A_GRADIENT_MODULE_NAMES,
    V25A_GRADIENT_TELEMETRY_DIRECTORY,
    V25A_GRADIENT_TELEMETRY_SCHEMA,
    V25A_RECORD_FIELDS,
    V25A_TASK_LABELS,
)
from train.known_gate_v21 import _verify_contract, _verify_readiness


OUTPUT = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V25a-known-gate-course-preclip-gradient-telemetry-audit.json"
)
EXPECTED_EPOCHS = tuple(range(1, 21))
STEPS_PER_EPOCH = 493
EXPECTED_RECORD_COUNT = len(EXPECTED_EPOCHS) * STEPS_PER_EPOCH


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    text = json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _finite(values: Iterable[Any], *, field: str) -> np.ndarray:
    result = np.asarray(list(values), dtype=np.float64)
    if result.size == 0 or not np.isfinite(result).all():
        raise ValueError(f"V25a report {field} is empty or non-finite")
    return result


def _distribution(values: Iterable[Any]) -> dict[str, float | int]:
    data = _finite(values, field="distribution")
    return {
        "count": int(data.size),
        "min": float(np.min(data)),
        "median": float(np.median(data)),
        "mean": float(np.mean(data)),
        "p90": float(np.quantile(data, 0.90)),
        "p95": float(np.quantile(data, 0.95)),
        "p99": float(np.quantile(data, 0.99)),
        "max": float(np.max(data)),
    }


def _clip_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("V25a report cannot summarize an empty stratum")
    clipped = [row for row in records if bool(row["was_globally_clipped"])]
    scales = [float(row["global_clip_scale"]) for row in clipped]
    return {
        "step_count": len(records),
        "clip_count": len(clipped),
        "clip_rate": float(len(clipped) / len(records)),
        "pre_clip_global_l2": _distribution(
            row["pre_clip_global_l2"] for row in records
        ),
        "clip_scale_when_clipped": (
            _distribution(scales) if scales else {"count": 0, "status": "no_clipped_steps"}
        ),
    }


def _module_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for module in V25A_GRADIENT_MODULE_NAMES:
        result[module] = {
            "pre_clip_l2": _distribution(
                row["pre_clip_module_l2"][module] for row in records
            ),
            "squared_gradient_fraction": _distribution(
                row["pre_clip_module_squared_fraction"][module] for row in records
            ),
            "nonzero_gradient_step_count": int(
                sum(bool(row["pre_clip_module_nonzero_gradient"][module]) for row in records)
            ),
        }
    return result


def _signature_summary(records: Sequence[Mapping[str, Any]], field: str) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[str(row[field])].append(row)
    result: dict[str, Any] = {}
    for key in sorted(grouped):
        rows = grouped[key]
        result[key] = {
            **_clip_summary(rows),
            "task_label_counts_total": {
                label: int(sum(int(row["task_label_counts"][label]) for row in rows))
                for label in V25A_TASK_LABELS
            },
            "base_label_counts_total": {
                label: int(sum(int(row["base_label_counts"][label]) for row in rows))
                for label in V25A_BASE_LABELS
            },
        }
    return result


def _boolean_stratum_summary(
    records: Sequence[Mapping[str, Any]],
    *,
    name: str,
    predicate: Any,
) -> dict[str, Any]:
    groups = {
        "true": [row for row in records if bool(predicate(row))],
        "false": [row for row in records if not bool(predicate(row))],
    }
    return {
        name: {
            key: (_clip_summary(rows) if rows else {"step_count": 0, "status": "empty"})
            for key, rows in groups.items()
        }
    }


def _observation_strata(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    definitions = {
        "has_same_second_event": lambda row: int(
            row["batch_observation_counts"]["zero_delta_event_count"]
        )
        > 0,
        "all_active_events_have_rd": lambda row: int(
            row["batch_observation_counts"]["rd_present_event_count"]
        )
        == int(row["batch_observation_counts"]["active_event_count"]),
        "all_active_events_have_valid_course": lambda row: int(
            row["batch_observation_counts"]["course_feature_valid_event_count"]
        )
        == int(row["batch_observation_counts"]["active_event_count"]),
    }
    result: dict[str, Any] = {}
    for name, predicate in definitions.items():
        result.update(
            _boolean_stratum_summary(records, name=name, predicate=predicate)
        )
    return result


def _epoch_summaries(
    records: Sequence[Mapping[str, Any]],
    *,
    selected_epoch: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    by_epoch: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        by_epoch[int(row["epoch"])].append(row)
    if tuple(sorted(by_epoch)) != EXPECTED_EPOCHS:
        raise ValueError("V25a telemetry lacks one or more epochs")
    epoch_summary = {
        str(epoch): {
            **_clip_summary(by_epoch[epoch]),
            "module_gradients": _module_summary(by_epoch[epoch]),
        }
        for epoch in EXPECTED_EPOCHS
    }
    if selected_epoch not in EXPECTED_EPOCHS:
        raise ValueError("V25a selected checkpoint epoch is outside the full run")
    phase_epochs = {
        "early_epochs_1_to_3": {1, 2, 3},
        "selected_checkpoint_epoch": {selected_epoch},
        "late_epochs_18_to_20": {18, 19, 20},
    }
    phase_summary = {
        name: {
            "epochs": sorted(epochs),
            **_clip_summary(
                [row for row in records if int(row["epoch"]) in epochs]
            ),
            "module_gradients": _module_summary(
                [row for row in records if int(row["epoch"]) in epochs]
            ),
        }
        for name, epochs in phase_epochs.items()
    }
    return epoch_summary, phase_summary


def _load_telemetry(
    artifact: Path,
    *,
    summary: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    telemetry = dict(summary.get("gradient_telemetry") or {})
    directory = artifact / V25A_GRADIENT_TELEMETRY_DIRECTORY
    metadata_path = directory / "metadata.json"
    if (
        telemetry.get("schema_version") != V25A_GRADIENT_TELEMETRY_SCHEMA
        or Path(str(telemetry.get("directory", ""))).resolve() != directory.resolve()
        or Path(str(telemetry.get("metadata_path", ""))).resolve()
        != metadata_path.resolve()
        or telemetry.get("metadata_sha256") != sha256_file(metadata_path)
        or int(telemetry.get("epoch_file_count", -1)) != 20
        or int(telemetry.get("record_count", -1)) != EXPECTED_RECORD_COUNT
        or int(telemetry.get("steps_per_epoch", -1)) != STEPS_PER_EPOCH
        or telemetry.get("gradient_telemetry_changes_training_semantics") is not False
        or telemetry.get("source_v24_parent_order_hashes_exactly_reproduced") is not True
    ):
        raise ValueError("V25a summary telemetry binding is invalid")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        metadata.get("schema_version") != V25A_GRADIENT_TELEMETRY_SCHEMA
        or int(metadata.get("steps_per_epoch", -1)) != STEPS_PER_EPOCH
        or metadata.get("capture_timing")
        != "after_loss_backward_before_existing_global_norm_clip"
        or dict(metadata.get("gradient_semantics") or {}).get("telemetry_changes_gradients")
        is not False
        or dict(metadata.get("gradient_semantics") or {}).get("raw_parent_sample_ids_stored")
        is not False
    ):
        raise ValueError("V25a telemetry metadata violates audit-only semantics")
    history = [dict(row) for row in list(summary.get("history") or ())]
    if [int(row.get("epoch", -1)) for row in history] != list(EXPECTED_EPOCHS):
        raise ValueError("V25a summary history is incomplete")
    records: list[dict[str, Any]] = []
    epoch_payloads: list[dict[str, Any]] = []
    expected_orders = dict(metadata.get("expected_parent_order_hashes") or {})
    for epoch, history_row in enumerate(history, start=1):
        descriptor = dict(history_row.get("gradient_telemetry") or {})
        path = directory / f"epoch-{epoch:03d}.json"
        if (
            Path(str(descriptor.get("path", ""))).resolve() != path.resolve()
            or descriptor.get("sha256") != sha256_file(path)
            or int(descriptor.get("epoch", -1)) != epoch
            or int(descriptor.get("record_count", -1)) != STEPS_PER_EPOCH
            or str(history_row.get("parent_order_sha256", ""))
            != str(expected_orders.get(str(epoch), ""))
        ):
            raise ValueError("V25a summary history has stale telemetry descriptor")
        payload = json.loads(path.read_text(encoding="utf-8"))
        step_records = [dict(row) for row in list(payload.get("records") or ())]
        if (
            payload.get("schema_version") != V25A_GRADIENT_TELEMETRY_SCHEMA
            or int(payload.get("epoch", -1)) != epoch
            or int(payload.get("step_count", -1)) != STEPS_PER_EPOCH
            or int(payload.get("global_step_start", -1)) != (epoch - 1) * STEPS_PER_EPOCH + 1
            or int(payload.get("global_step_end", -1)) != epoch * STEPS_PER_EPOCH
            or str(payload.get("parent_order_sha256", ""))
            != str(history_row.get("parent_order_sha256", ""))
            or len(step_records) != STEPS_PER_EPOCH
        ):
            raise ValueError("V25a epoch telemetry file is incomplete or stale")
        if any(set(record) != V25A_RECORD_FIELDS for record in step_records):
            raise ValueError("V25a telemetry record fields violate the sealed no-identifier schema")
        epoch_payloads.append(payload)
        records.extend(step_records)
    if len(records) != EXPECTED_RECORD_COUNT:
        raise ValueError("V25a telemetry record count differs from full training")
    raw_blob = json.dumps(records, ensure_ascii=False)
    forbidden_identifier_fields = {
        "parent_sample_id",
        "parent_sample_ids",
        "track_id",
        "capture_id",
        "sample_id",
        "source_name",
    }
    if any(
        field in forbidden_identifier_fields
        for record in records
        for field in record
    ) or any(
        field in raw_blob
        for field in forbidden_identifier_fields
    ):
        raise ValueError("V25a telemetry leaked a raw parent identifier")
    return metadata, epoch_payloads, records


def build_report(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    config_path = root / CONFIG
    readiness_path = root / READINESS
    contract_path = root / EXPERIMENT_CONTRACT
    artifact = root / ARTIFACT
    summary_path = artifact / "training-summary.json"
    if not all(path.is_file() for path in (config_path, readiness_path, contract_path, summary_path)):
        raise FileNotFoundError("V25a completed training artifact is incomplete")
    config = load_known_gate_config(config_path)
    contract = _verify_contract(root, config)
    readiness = _verify_readiness(root, config_path, config, contract)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        config.variant != V25A_GRADIENT_TELEMETRY_VARIANT
        or readiness.get("schema_version") != KNOWN_GATE_V25A_READINESS_SCHEMA
        or summary.get("status") != "completed"
        or summary.get("variant") != V25A_GRADIENT_TELEMETRY_VARIANT
        or int(summary.get("global_step", -1)) != EXPECTED_RECORD_COUNT
        or int(summary.get("best_epoch", -1)) < 1
        or summary.get("test_evaluation") != "not_executed"
        or summary.get("oof_threshold_calibration") != "not_executed"
        or summary.get("full_cascade_evaluation") != "not_executed"
        or summary.get("formal_rejection") != "not_executed"
        or summary.get("stable_evaluation") != "not_executed"
        or summary.get("p95_latency") != "not_executed"
        or summary.get("deployment_eligible") is not False
        or dict(summary.get("training_semantics_audit") or {}).get("telemetry_only_change")
        is not True
        or summary.get("initial_model_state_sha256")
        != readiness.get("source_v24_initial_model_state_sha256")
    ):
        raise ValueError("V25a summary violates the sealed telemetry-only contract")
    semantics = dict(summary.get("training_semantics_audit") or {})
    if (
        semantics.get("optimizer")
        != "torch.optim.AdamW(model.parameters(), lr=1e-4) defaults"
        or float(semantics.get("optimizer_weight_decay", float("nan"))) != 0.01
        or float(semantics.get("global_gradient_clip_norm", float("nan"))) != 100.0
        or semantics.get("class_weight_normalization") != "batch_weight_sum"
        or semantics.get("telemetry_capture_timing")
        != "after_loss_backward_before_existing_global_norm_clip"
        or any(
            semantics.get(key) is not False
            for key in (
                "telemetry_changes_gradients",
                "telemetry_changes_optimizer",
                "telemetry_changes_loss",
                "telemetry_changes_data_order",
            )
        )
    ):
        raise ValueError("V25a summary training semantics are stale")
    metadata, _, records = _load_telemetry(artifact, summary=summary)
    epoch_summary, phase_summary = _epoch_summaries(
        records, selected_epoch=int(summary["best_epoch"])
    )
    clipped = [row for row in records if bool(row["was_globally_clipped"])]
    non_clipped = [row for row in records if not bool(row["was_globally_clipped"])]
    report = {
        "status": "completed_development_gradient_telemetry_audit_only",
        "schema_version": "RADAR-LN3-V25a-known-gate-course-preclip-gradient-telemetry-audit-v1",
        "scope": "V25a is descriptive V24 training telemetry only; no checkpoint tuning, OOF, test, Unknown, L3, full cascade, rejection, stable or P95 claim.",
        "training_summary_path": str(summary_path),
        "training_summary_sha256": sha256_file(summary_path),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "readiness_path": str(readiness_path),
        "readiness_sha256": readiness["sha256"],
        "experiment_contract_path": str(contract_path),
        "experiment_contract_sha256": sha256_file(contract_path),
        "telemetry_metadata": {
            "path": str(artifact / V25A_GRADIENT_TELEMETRY_DIRECTORY / "metadata.json"),
            "sha256": sha256_file(artifact / V25A_GRADIENT_TELEMETRY_DIRECTORY / "metadata.json"),
            "payload": metadata,
        },
        "integrity": {
            "epoch_count": 20,
            "steps_per_epoch": STEPS_PER_EPOCH,
            "record_count": len(records),
            "expected_record_count": EXPECTED_RECORD_COUNT,
            "all_source_v24_parent_orders_reproduced": True,
            "initial_model_state_sha256": summary.get("initial_model_state_sha256"),
            "sealed_v24_initial_model_state_sha256": readiness.get(
                "source_v24_initial_model_state_sha256"
            ),
            "raw_parent_identifiers_stored": False,
            "telemetry_changes_training_semantics": False,
        },
        "run_context_not_used_for_tuning": {
            "best_checkpoint_epoch": int(summary["best_epoch"]),
            "performance_status": str(summary.get("performance_status", "")),
            "validation_was_used_only_for_the_frozen_v24_checkpoint_rule": True,
        },
        "whole_run": {
            **_clip_summary(records),
            "loss": {
                name: _distribution(row["loss"][name] for row in records)
                for name in ("total", "final_state", "prefix_auxiliary")
            },
            "module_gradients": _module_summary(records),
        },
        "by_epoch": epoch_summary,
        "phase_comparison": phase_summary,
        "by_task_label_signature": _signature_summary(records, "task_label_signature"),
        "by_base_label_signature": _signature_summary(records, "base_label_signature"),
        "by_contains_clutter_parent": _boolean_stratum_summary(
            records,
            name="contains_clutter_parent",
            predicate=lambda row: row["contains_clutter_parent"],
        )["contains_clutter_parent"],
        "observation_condition_strata": _observation_strata(records),
        "module_gradients_by_clipping": {
            "clipped": _module_summary(clipped) if clipped else {"status": "no_clipped_steps"},
            "not_clipped": _module_summary(non_clipped)
            if non_clipped
            else {"status": "no_non_clipped_steps"},
        },
        "interpretation_limits": [
            "Telemetry is observational: a batch-label association cannot establish a label, source, feature, or module as the cause of a spike.",
            "Batch size is two and only 33 train Clutter parents exist; contains-Clutter strata are sparse and should not be generalized.",
            "The run intentionally preserves V24 batch_weight_sum class weighting and global clip norm; it does not evaluate a remedy.",
            "Validation is retained only because frozen V24 checkpoint selection requires it; this report does not use it for tuning.",
        ],
        "OOF": "not_executed",
        "test": "not_executed",
        "formal_rejection": "not_executed",
        "full_cascade": "not_executed",
        "stable": "not_executed",
        "P95": "not_executed",
        "deployment_eligible": False,
    }
    output = root / OUTPUT
    _write_json(output, report)
    return {**report, "report_path": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report V25a pre-clip gradient telemetry without changing training semantics."
    )
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    args = parser.parse_args()
    print(json.dumps(build_report(args.project_root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
