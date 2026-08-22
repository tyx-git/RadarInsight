"""Read-only V23e scan of the frozen V23a 20-epoch checkpoint trajectory."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from radar_ln3.known_gate import (
    build_known_gate_dataset,
    evaluate_known_gate,
    known_gate_device,
)
from radar_ln3.known_gate_official_cfc import (
    build_official_cfc_model_from_frozen_source,
)
from radar_ln3.known_gate_v23 import sha256_file
from radar_ln3.known_gate_v23_adapters import (
    V23A_VARIANT,
    artifact_relative,
    config_identity_sha256,
    config_relative,
    load_adapter_config,
    readiness_relative,
    source_runtime_report,
)
from radar_ln3.known_gate_v23e_selection import (
    V23E_MINIMUM_CLUTTER_PARENT_SUPPORT,
    V23E_SELECTOR_SCHEMA,
    assess_checkpoint,
    clutter_prefix_action_trajectories,
    frozen_v23a_legacy_reference,
    is_better_candidate,
)
from report.known_gate_v21e import _ci


OUTPUT = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V23e-known-gate-parent-balanced-checkpoint-selection.json"
)
REPORT_SCHEMA = "RADAR-LN3-V23e-known-gate-parent-balanced-checkpoint-selection-report-v1"
BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_SEED = 23051


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _load_json(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _v23a_context(root: Path) -> tuple[Any, Any, Path, dict[str, Any], dict[str, Any]]:
    config, source_config, _source_path = load_adapter_config(root, V23A_VARIANT)
    config_path = (root / config_relative(V23A_VARIANT)).resolve()
    readiness_path = (root / readiness_relative(V23A_VARIANT)).resolve()
    artifact = (root / artifact_relative(V23A_VARIANT)).resolve()
    summary_path = artifact / "training-summary.json"
    if not all(path.is_file() for path in (config_path, readiness_path, summary_path)):
        raise FileNotFoundError("V23e requires completed V23a config/readiness/summary")
    readiness = _load_json(readiness_path)
    summary = _load_json(summary_path)
    if (
        readiness.get("status") != "passed"
        or readiness.get("variant") != V23A_VARIANT
        or int(readiness.get("validation_parent_count", -1)) != 232
        or int(readiness.get("validation_clutter_parent_support", -1))
        != V23E_MINIMUM_CLUTTER_PARENT_SUPPORT
        or summary.get("status") != "completed"
        or summary.get("variant") != V23A_VARIANT
        or len(list(summary.get("history") or ())) != 20
        or summary.get("config_sha256") != sha256_file(config_path)
        or summary.get("config_identity_sha256") != config_identity_sha256(config)
        or summary.get("test_evaluation") != "not_executed"
        or summary.get("deployment_eligible") is not False
    ):
        raise ValueError("V23e completed V23a source binding is stale")
    return config, source_config, artifact, readiness, summary


def _verify_epoch_payload(
    path: Path,
    *,
    epoch: int,
    config: Any,
    config_path: Path,
    readiness_sha256: str,
    runtime_path: Path,
    runtime_sha256: str,
) -> dict[str, Any]:
    payload = dict(torch.load(path, map_location="cpu", weights_only=False))
    if (
        payload.get("checkpoint_schema") != config.train.checkpoint_schema
        or payload.get("task_name") != "known_gate"
        or payload.get("variant") != V23A_VARIANT
        or payload.get("architecture") != "official_ncps_cfc_external_state_adapter"
        or int(payload.get("epoch", -1)) != epoch
        or payload.get("config_identity_sha256") != config_identity_sha256(config)
        or payload.get("config_file_sha256") != sha256_file(config_path)
        or payload.get("readiness_sha256") != readiness_sha256
        or dict(payload.get("runtime_compatibility") or {})
        != {"path": str(runtime_path), "sha256": runtime_sha256}
        or payload.get("full_parameter_training") is not True
        or payload.get("unknown_is_model_class") is not False
        or not isinstance(payload.get("model_state"), Mapping)
    ):
        raise ValueError(f"V23e epoch checkpoint is stale or invalid: {path}")
    return payload


def _records_from_evaluation(evaluation: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    records = dict(evaluation.get("records") or {})
    result = {
        scope: [dict(row) for row in list(records.get(scope) or ())]
        for scope in ("parent", "eligible_prefix", "window")
    }
    if (
        len(result["parent"]) != 232
        or len(result["eligible_prefix"]) != 3756
        or len(result["window"]) != 4452
    ):
        raise ValueError("V23e evaluation record counts differ from V23a contract")
    return result


def _selection_bootstrap(records: Mapping[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Parent bootstrap for the selected state, retaining all its prefixes."""

    parents = {str(row["parent_sample_id"]): dict(row) for row in records["parent"]}
    prefixes: dict[str, list[dict[str, Any]]] = {}
    for row in records["eligible_prefix"]:
        prefixes.setdefault(str(row["parent_sample_id"]), []).append(dict(row))
    labels = ("DroneTarget", "BirdTarget", "BalloonTarget", "ClutterTarget")
    strata = {
        label: np.asarray(
            [
                parent_id
                for parent_id, row in parents.items()
                if str(row["label_name"]) == label
            ],
            dtype=object,
        )
        for label in labels
    }
    expected = {"DroneTarget": 113, "BirdTarget": 56, "BalloonTarget": 54, "ClutterTarget": 9}
    if {label: len(values) for label, values in strata.items()} != expected:
        raise ValueError("V23e selected-state bootstrap support changed")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    parent_clutter_precision: list[float] = []
    parent_clutter_recall: list[float] = []
    parent_clutter_f1: list[float] = []
    parent_macro_f1: list[float] = []
    prefix_clutter_recall: list[float] = []
    prefix_clutter_f1: list[float] = []
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled_ids = [
            str(parent_id)
            for values in strata.values()
            for parent_id in rng.choice(values, size=len(values), replace=True)
        ]
        sampled_parent = [parents[parent_id] for parent_id in sampled_ids]
        sampled_prefix = [
            {**row, "parent_sample_id": f"{parent_id}#bootstrap{index}"}
            for index, parent_id in enumerate(sampled_ids)
            for row in prefixes.get(parent_id, ())
        ]
        # Distinct synthetic IDs retain the intended multiplicity when a
        # bootstrap draw contains the same original parent more than once.
        from radar_ln3.known_gate import (
            summarize_known_gate_records,
            summarize_parent_balanced_eligible_prefix_records,
        )

        parent_metrics = summarize_known_gate_records(
            sampled_parent, scope="v23e_selected_parent_bootstrap"
        )
        prefix_metrics = summarize_parent_balanced_eligible_prefix_records(
            sampled_prefix, scope="v23e_selected_prefix_bootstrap"
        )
        parent_rows = {row["class"]: row for row in parent_metrics["per_class"]}
        prefix_rows = {row["class"]: row for row in prefix_metrics["per_class"]}
        parent_clutter_precision.append(float(parent_rows["KNOWN_CLUTTER_EVIDENCE"]["precision"]))
        parent_clutter_recall.append(float(parent_rows["KNOWN_CLUTTER_EVIDENCE"]["recall"]))
        parent_clutter_f1.append(float(parent_rows["KNOWN_CLUTTER_EVIDENCE"]["f1"]))
        parent_macro_f1.append(float(parent_metrics["macro_f1"]))
        prefix_clutter_recall.append(float(prefix_rows["KNOWN_CLUTTER_EVIDENCE"]["recall"]))
        prefix_clutter_f1.append(float(prefix_rows["KNOWN_CLUTTER_EVIDENCE"]["f1"]))
    return {
        "method": "stratified parent bootstrap; each sampled parent retains all causal prefixes with internal 1/n weight",
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "confidence_level": 0.95,
        "stratum_support": expected,
        "parent_final": {
            "macro_f1": _ci(parent_macro_f1),
            "clutter_precision": _ci(parent_clutter_precision),
            "clutter_recall": _ci(parent_clutter_recall),
            "clutter_f1": _ci(parent_clutter_f1),
        },
        "parent_balanced_eligible_prefix": {
            "clutter_recall": _ci(prefix_clutter_recall),
            "clutter_f1": _ci(prefix_clutter_f1),
        },
    }


def run(
    project_root: Path,
    *,
    output: Path | None = None,
    start_epoch: int = 1,
    end_epoch: int = 20,
) -> dict[str, Any]:
    root = project_root.resolve()
    if not 1 <= int(start_epoch) <= int(end_epoch) <= 20:
        raise ValueError("V23e epoch range must be within [1, 20]")
    config, source_config, artifact, readiness, summary = _v23a_context(root)
    config_path = (root / config_relative(V23A_VARIANT)).resolve()
    runtime_path, _runtime, runtime_sha = source_runtime_report(root)
    validation = build_known_gate_dataset(root, source_config, split_name="val")
    if validation.label_counts() != {
        "KNOWN_TARGET_EVIDENCE": 223,
        "KNOWN_CLUTTER_EVIDENCE": V23E_MINIMUM_CLUTTER_PARENT_SUPPORT,
    }:
        raise ValueError("V23e validation support differs from the sealed V23a source")
    device = known_gate_device(source_config)
    reference = frozen_v23a_legacy_reference(root)
    model, initialization_audit = build_official_cfc_model_from_frozen_source(
        root, source_config, state_adapter_spec=config.adapter
    )
    if model.external_state_adapter_specification() != config.adapter.as_dict():
        raise ValueError("V23e model adapter differs from frozen V23a")
    model.to(device)
    candidates: list[dict[str, Any]] = []
    epoch_records: dict[int, dict[str, list[dict[str, Any]]]] = {}
    readiness_sha = sha256_file(root / readiness_relative(V23A_VARIANT))
    for epoch in range(int(start_epoch), int(end_epoch) + 1):
        checkpoint_path = artifact / "epochs" / f"epoch-{epoch:03d}.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        checkpoint = _verify_epoch_payload(
            checkpoint_path,
            epoch=epoch,
            config=config,
            config_path=config_path,
            readiness_sha256=readiness_sha,
            runtime_path=runtime_path,
            runtime_sha256=runtime_sha,
        )
        model.load_state_dict(checkpoint["model_state"], strict=True)
        evaluation = evaluate_known_gate(model, validation, config=source_config, device=device)
        records = _records_from_evaluation(evaluation)
        assessment = assess_checkpoint(evaluation, reference=reference, epoch=epoch)
        assessment["checkpoint"] = {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "saved_epoch_metrics_match_checkpoint_epoch": int(
                dict(checkpoint.get("metrics") or {}).get("epoch", -1)
            )
            == epoch,
        }
        if not assessment["checkpoint"]["saved_epoch_metrics_match_checkpoint_epoch"]:
            raise ValueError("V23e epoch checkpoint metric metadata differs")
        candidates.append(assessment)
        epoch_records[epoch] = records
    scanned_all_epochs = int(start_epoch) == 1 and int(end_epoch) == 20
    selected: dict[str, Any] | None = None
    for candidate in candidates:
        if is_better_candidate(candidate, selected):
            selected = candidate
    if selected is None:
        selected_payload: dict[str, Any] | None = None
        selected_bootstrap: dict[str, Any] | None = None
        selected_trajectories: dict[str, Any] | None = None
        conclusion = (
            "V23e selector-only scan found no feasible V23a epoch; checkpoint "
            "selection alone cannot satisfy the frozen parent-final and causal-prefix constraints."
            if scanned_all_epochs
            else "V23e partial read-only scan completed; no selector conclusion is permitted until every frozen epoch 1-20 is scanned."
        )
    else:
        selected_epoch = int(selected["epoch"])
        selected_payload = selected
        selected_bootstrap = _selection_bootstrap(epoch_records[selected_epoch])
        selected_trajectories = clutter_prefix_action_trajectories(epoch_records[selected_epoch])
        conclusion = (
            "V23e selector-only scan found a feasible V23a epoch under the frozen "
            "parent-final and parent-balanced causal-prefix constraints."
        )
    report = {
        "status": (
            "completed_development_selector_only"
            if scanned_all_epochs
            else "completed_development_selector_partial_only"
        ),
        "schema_version": REPORT_SCHEMA,
        "selector_schema_version": V23E_SELECTOR_SCHEMA,
        "experiment": "V23e checkpoint-selection-only scan of frozen V23a training states",
        "scope": "V21 development KnownGate only; no retraining, OOF, test, Unknown, L3, full cascade, rejection, stable or P95",
        "single_factor": {
            "changed_component": "checkpoint selection rule only",
            "frozen": [
                "V23a loss", "official-CfC model and V23a positive-dt adapter",
                "features", "dt semantics", "initialization", "DataLoader parent order",
                "optimizer", "batch weighting", "20 saved epochs",
            ],
            "training_step_count": 0,
            "optimizer_step_count": 0,
        },
        "source_v23a": {
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
            "config_identity_sha256": config_identity_sha256(config),
            "readiness_path": str(root / readiness_relative(V23A_VARIANT)),
            "readiness_sha256": readiness_sha,
            "training_summary_path": str(artifact / "training-summary.json"),
            "training_summary_sha256": sha256_file(artifact / "training-summary.json"),
            "legacy_best_epoch": int(summary["best_epoch"]),
            "initialization_audit": initialization_audit,
        },
        "legacy_reference": reference,
        "selector_protocol": {
            "clutter_parent_support": V23E_MINIMUM_CLUTTER_PARENT_SUPPORT,
            "parent_final_clutter_true_positive_count_minimum": 8,
            "parent_final_clutter_recall_minimum": 8 / 9,
            "parent_final_clutter_precision_minimum": 0.90,
            "prefix_aggregation": "hard causal actions; each eligible parent total weight is one, each prefix is 1/n_parent",
            "pending_events": "events 1-2 are PENDING_ABSTAIN and excluded from prefix classification denominators",
            "reference": "frozen V23a legacy parent-final-selected epoch only",
            "ranking": candidates[0]["ranking_definition"],
        },
        "scanned_epoch_range": [int(start_epoch), int(end_epoch)],
        "scanned_all_frozen_epochs": scanned_all_epochs,
        "epoch_candidates": candidates,
        "feasible_epoch_count": sum(bool(candidate["feasible"]) for candidate in candidates),
        "selected_checkpoint": selected_payload,
        "selected_checkpoint_parent_bootstrap": selected_bootstrap,
        "selected_checkpoint_clutter_prefix_action_trajectories": selected_trajectories,
        "conclusion": conclusion,
        "not_final_product_performance": True,
        "OOF": "未执行",
        "test": "未执行",
        "formal_rejection": "未执行",
        "full_cascade": "未执行",
        "stable": "未执行",
        "P95": "未执行",
        "deployment_eligible": False,
    }
    destination = (root / output) if output is not None else (root / OUTPUT)
    _write_json(destination, report)
    report["report_path"] = str(destination)
    return report


def merge_parts(project_root: Path, *, output: Path | None = None) -> dict[str, Any]:
    """Merge four complete V23e partial scans without re-running inference.

    The frozen V23a trajectory is deterministic and every partial file embeds
    checkpoint hashes, metrics and the common legacy reference.  Merging those
    sealed slices is therefore read-only bookkeeping, not post-hoc training or
    selection.  It keeps the scan usable on hosts that enforce short command
    timeouts.
    """

    root = project_root.resolve()
    parts: list[dict[str, Any]] = []
    for start, end in ((1, 5), (6, 10), (11, 15), (16, 20)):
        path = root / OUTPUT.with_name(
            f"RADAR-LN3-V23e-known-gate-parent-balanced-checkpoint-selection-part-{start:02d}-{end:02d}.json"
        )
        if not path.is_file():
            raise FileNotFoundError(f"V23e required partial scan is missing: {path}")
        payload = _load_json(path)
        if (
            payload.get("status") != "completed_development_selector_partial_only"
            or list(payload.get("scanned_epoch_range") or ()) != [start, end]
            or payload.get("scanned_all_frozen_epochs") is not False
            or payload.get("schema_version") != REPORT_SCHEMA
            or payload.get("selector_schema_version") != V23E_SELECTOR_SCHEMA
            or len(list(payload.get("epoch_candidates") or ())) != end - start + 1
        ):
            raise ValueError(f"V23e partial scan violates its contract: {path}")
        parts.append(payload)
    reference = dict(parts[0]["legacy_reference"])
    source = dict(parts[0]["source_v23a"])
    protocol = dict(parts[0]["selector_protocol"])
    if any(
        dict(part["legacy_reference"]) != reference
        or dict(part["source_v23a"]) != source
        or dict(part["selector_protocol"]) != protocol
        for part in parts[1:]
    ):
        raise ValueError("V23e partial scans do not share the same frozen source/reference/protocol")
    candidates = [
        dict(candidate)
        for part in parts
        for candidate in list(part["epoch_candidates"])
    ]
    if [int(candidate["epoch"]) for candidate in candidates] != list(range(1, 21)):
        raise ValueError("V23e partial scan epochs are incomplete, repeated or unordered")
    selected: dict[str, Any] | None = None
    for candidate in candidates:
        if is_better_candidate(candidate, selected):
            selected = candidate
    if selected is not None:
        # A feasible state is unexpected given the current V23a history but
        # must still be represented correctly rather than silently discarded.
        selected_payload = selected
        selected_bootstrap = None
        selected_trajectories = None
        conclusion = "V23e merged all frozen partial scans and found a feasible checkpoint; re-run the selected epoch report for action traces/bootstrap."
    else:
        selected_payload = None
        selected_bootstrap = None
        selected_trajectories = None
        conclusion = "V23e selector-only scan found no feasible V23a epoch; checkpoint selection alone cannot satisfy the frozen parent-final and causal-prefix constraints."
    report = {
        "status": "completed_development_selector_only",
        "schema_version": REPORT_SCHEMA,
        "selector_schema_version": V23E_SELECTOR_SCHEMA,
        "experiment": "V23e checkpoint-selection-only scan of frozen V23a training states",
        "scope": "V21 development KnownGate only; no retraining, OOF, test, Unknown, L3, full cascade, rejection, stable or P95",
        "single_factor": {
            "changed_component": "checkpoint selection rule only",
            "frozen": list(parts[0]["single_factor"]["frozen"]),
            "training_step_count": 0,
            "optimizer_step_count": 0,
        },
        "source_v23a": source,
        "legacy_reference": reference,
        "selector_protocol": protocol,
        "partial_scan_sources": [
            {
                "path": str(
                    OUTPUT.with_name(
                        f"RADAR-LN3-V23e-known-gate-parent-balanced-checkpoint-selection-part-{start:02d}-{end:02d}.json"
                    )
                ),
                "sha256": sha256_file(
                    root / OUTPUT.with_name(
                        f"RADAR-LN3-V23e-known-gate-parent-balanced-checkpoint-selection-part-{start:02d}-{end:02d}.json"
                    )
                ),
                "epoch_range": [start, end],
            }
            for start, end in ((1, 5), (6, 10), (11, 15), (16, 20))
        ],
        "scanned_epoch_range": [1, 20],
        "scanned_all_frozen_epochs": True,
        "epoch_candidates": candidates,
        "feasible_epoch_count": sum(bool(candidate["feasible"]) for candidate in candidates),
        "selected_checkpoint": selected_payload,
        "selected_checkpoint_parent_bootstrap": selected_bootstrap,
        "selected_checkpoint_clutter_prefix_action_trajectories": selected_trajectories,
        "conclusion": conclusion,
        "not_final_product_performance": True,
        "OOF": "未执行",
        "test": "未执行",
        "formal_rejection": "未执行",
        "full_cascade": "未执行",
        "stable": "未执行",
        "P95": "未执行",
        "deployment_eligible": False,
    }
    destination = (root / output) if output is not None else (root / OUTPUT)
    _write_json(destination, report)
    report["report_path"] = str(destination)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only V23e checkpoint-selection scan for frozen V23a."
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--start-epoch", type=int, default=1)
    parser.add_argument("--end-epoch", type=int, default=20)
    parser.add_argument("--merge-parts", action="store_true")
    args = parser.parse_args()
    result = (
        merge_parts(args.project_root, output=args.output)
        if args.merge_parts
        else run(
            args.project_root,
            output=args.output,
            start_epoch=args.start_epoch,
            end_epoch=args.end_epoch,
        )
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
