"""Evaluate V26b frozen L1+L2 raw routing without calibrated L3 decisions."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from build.known_gate_v26b_l1_l2_raw_routing import (
    CONTRACT_SCHEMA,
    EXPECTED_PARENT_COUNTS,
    READINESS_SCHEMA,
    V18R_CHECKPOINT,
    V18R_CONFIG,
    V18R_READINESS,
    V18R_SUMMARY,
    V26B_CACHE_MANIFEST,
    V26B_CONTRACT,
    V26B_L3_REGISTRY,
    V26B_READINESS,
    V26B_TEST_MICRO_CACHE,
    V26B_TEST_PHYSICAL_CACHE,
    V26B_VALIDATION_MICRO_CACHE,
    V26B_VALIDATION_PHYSICAL_CACHE,
    V26_AUDIT_VALIDATION_CACHE,
    V26_CHECKPOINT,
    V26_CONFIG,
    V26_READINESS,
    V26_SUMMARY,
    implementation_sha256,
    materialize_test_l2_mixed_cache,
)
from build.known_gate_v26 import materialize_test_audit_cache
from radar_ln3.known_gate import (
    KNOWN_LABELS,
    UNKNOWN_LABEL,
    build_known_gate_model_for_config,
    known_gate_device,
    load_known_gate_config,
    sha256_file,
)
from radar_ln3.known_gate_v26_audit import (
    V26OpenSetAuditDataset,
    build_v26_open_set_audit_dataloader,
)
from radar_ln3.l1_l2_raw_routing import (
    FrozenL2MixedAuditDataset,
    build_frozen_l2_mixed_audit_dataloader,
    l2_model_inputs,
    verify_l1_l2_event_alignment,
)
from radar_ln3.open_set_architecture import ProductAction
from radar_ln3.semantic_decision_registry import (
    UNFITTED_CALIBRATION_STATUS,
    validate_semantic_decision_registry,
)
from radar_ln3.task_models import (
    build_paired_model_for_config,
    load_task_checkpoint,
    load_task_config,
)
from train.known_gate_common import (
    _load_checkpoint as load_known_gate_checkpoint,
    _model_inputs as l1_model_inputs,
    _verify_contract,
    _verify_readiness,
)


OUTPUT = Path(
    "src-v3/processed/report/RADAR-LN3-V26b-frozen-l1-l2-raw-routing-audit.json"
)
PARENT_DETAIL_CSV = Path(
    "src-v3/processed/report/RADAR-LN3-V26b-frozen-l1-l2-raw-routing-parent-details.csv"
)
MAX_PERSISTED_REPORT_BYTES = 2 * 1024 * 1024
BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_SEED = 26061
DBB_NAMES = ("DroneTarget", "BirdTarget", "BalloonTarget")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    text = json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n"
    if len(text.encode("utf-8")) > MAX_PERSISTED_REPORT_BYTES:
        raise MemoryError(f"V26b report exceeds bounded size: {path}")
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _write_parent_details(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    columns = (
        "split",
        "parent_sample_id",
        "label_name",
        "capture_id",
        "provenance_component_id",
        "event_count",
        "l1_route",
        "l1_target_probability",
        "l1_clutter_probability",
        "l1_confidence",
        "l2_top_subclass",
        "l2_top_probability",
        "l2_margin",
        "raw_route_action",
        "raw_route_correct_for_known_truth",
        "raw_route_unknown_false_accept",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _require_preparation(root: Path) -> dict[str, Any]:
    paths = {
        "contract": root / V26B_CONTRACT,
        "registry": root / V26B_L3_REGISTRY,
        "cache_manifest": root / V26B_CACHE_MANIFEST,
        "readiness": root / V26B_READINESS,
        "l1_config": root / V26_CONFIG,
        "l1_checkpoint": root / V26_CHECKPOINT,
        "l1_readiness": root / V26_READINESS,
        "l1_summary": root / V26_SUMMARY,
        "l2_config": root / V18R_CONFIG,
        "l2_checkpoint": root / V18R_CHECKPOINT,
        "l2_readiness": root / V18R_READINESS,
        "l2_summary": root / V18R_SUMMARY,
        "validation_l1_cache": root / V26_AUDIT_VALIDATION_CACHE,
        "validation_l2_physical": root / V26B_VALIDATION_PHYSICAL_CACHE,
        "validation_l2_micro": root / V26B_VALIDATION_MICRO_CACHE,
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("V26b preparation is incomplete: " + ", ".join(missing))
    contract = _read_json(paths["contract"])
    registry = validate_semantic_decision_registry(_read_json(paths["registry"]))
    cache_manifest = _read_json(paths["cache_manifest"])
    readiness = _read_json(paths["readiness"])
    checks = dict(readiness.get("checks") or {})
    bindings = dict(registry["component_bindings"])
    expected_bindings = {
        "l1_known_gate": {
            "task_name": "known_gate",
            "config": paths["l1_config"],
            "checkpoint": paths["l1_checkpoint"],
            "training_summary": paths["l1_summary"],
            "training_readiness": paths["l1_readiness"],
        },
        "l2_dbb_specialist": {
            "task_name": "dbb_specialist",
            "config": paths["l2_config"],
            "checkpoint": paths["l2_checkpoint"],
            "training_summary": paths["l2_summary"],
            "training_readiness": paths["l2_readiness"],
        },
    }
    binding_matches = True
    for name, expected in expected_bindings.items():
        binding = dict(bindings.get(name) or {})
        if binding.get("task_name") != expected["task_name"]:
            binding_matches = False
            break
        for field in ("config", "checkpoint", "training_summary", "training_readiness"):
            path = Path(expected[field]).resolve()
            if (
                Path(str(binding.get(f"{field}_path", ""))).as_posix()
                != path.relative_to(root).as_posix()
                or binding.get(f"{field}_sha256") != sha256_file(path)
            ):
                binding_matches = False
                break
        if not binding_matches:
            break
    implementation_matches = contract.get("implementation_sha256") == implementation_sha256(root)
    if (
        contract.get("schema_version") != CONTRACT_SCHEMA
        or readiness.get("schema_version") != READINESS_SCHEMA
        or readiness.get("status") != "passed"
        or not checks
        or not all(value is True for value in checks.values())
        or readiness.get("contract_sha256") != sha256_file(paths["contract"])
        or readiness.get("l3_registry_sha256") != sha256_file(paths["registry"])
        or contract.get("l3_registry", {}).get("sha256")
        != sha256_file(paths["registry"])
        or contract.get("data_contract", {}).get("cache_manifest", {}).get("sha256")
        != sha256_file(paths["cache_manifest"])
        or cache_manifest.get("schema_version")
        != "RADAR-LN3-V26b-l2-mixed-open-set-feature-cache-v1"
        or cache_manifest.get("validation", {}).get("physical_cache_sha256")
        != sha256_file(paths["validation_l2_physical"])
        or cache_manifest.get("validation", {}).get("micro_doppler_cache_sha256")
        != sha256_file(paths["validation_l2_micro"])
        or registry["calibration"]["status"] != UNFITTED_CALIBRATION_STATUS
        or registry["final_actions_enabled"] is not False
        or not binding_matches
        or not implementation_matches
        or contract.get("unknown_semantics", {}).get("unknown_is_model_class") is not False
        or contract.get("unknown_semantics", {}).get("pending_is_unknown_rejection_success")
        is not False
    ):
        raise ValueError("V26b preparation/registry contract is stale or not audit-only")
    return {
        "paths": paths,
        "contract": contract,
        "registry": registry,
        "cache_manifest": cache_manifest,
        "readiness": readiness,
    }


def _load_models(root: Path, prepared: Mapping[str, Any]) -> tuple[Any, Any, torch.device, dict[str, Any]]:
    paths = dict(prepared["paths"])
    l1_config = load_known_gate_config(paths["l1_config"])
    l1_contract = _verify_contract(root, l1_config)
    l1_readiness = _verify_readiness(root, paths["l1_config"], l1_config, l1_contract)
    device = known_gate_device(l1_config)
    l1, reference, pairing = build_known_gate_model_for_config(root, l1_config)
    del reference, pairing
    l1.to(device)
    load_known_gate_checkpoint(
        paths["l1_checkpoint"],
        l1,
        None,
        config=l1_config,
        config_path=paths["l1_config"],
        readiness_sha256=l1_readiness["sha256"],
        map_location=device,
    )

    l2_config = load_task_config(paths["l2_config"])
    l2, reference_l2, pairing_l2 = build_paired_model_for_config(root, l2_config)
    del reference_l2, pairing_l2
    l2.to(device)
    load_task_checkpoint(
        paths["l2_checkpoint"],
        l2,
        expected_schema=l2_config.train.checkpoint_schema,
        map_location=device,
    )
    l1.eval()
    l2.eval()
    return l1, l2, device, {"l1_config": l1_config, "l2_config": l2_config}


def _base_metadata(dataset: Any, parent_id: str) -> dict[str, str]:
    first = dataset.base._groups[str(parent_id)].iloc[0]
    return {
        "capture_id": str(first.get("capture_id", "")),
        "provenance_component_id": str(first.get("provenance_component_id", "")),
    }


def _route_action(l1_id: int, l2_id: int) -> str:
    if l1_id == 1:
        return ProductAction.CLUTTER.value
    return DBB_NAMES[l2_id]


def _known_route_correct(label: str, action: str) -> bool | None:
    if label == UNKNOWN_LABEL:
        return None
    return action == label


def _run_split(
    root: Path,
    *,
    split: str,
    l1: Any,
    l2: Any,
    device: torch.device,
    configs: Mapping[str, Any],
    l1_cache: Path,
    l2_physical_cache: Path,
    l2_micro_cache: Path,
) -> dict[str, Any]:
    l1_dataset = V26OpenSetAuditDataset(root, configs["l1_config"], split_name=split, cache_path=l1_cache)
    l2_dataset = FrozenL2MixedAuditDataset(
        root,
        configs["l2_config"],
        split_name=split,
        rd_physical_cache_path=l2_physical_cache,
        micro_doppler_cache_path=l2_micro_cache,
    )
    if l1_dataset.label_counts() != EXPECTED_PARENT_COUNTS[split]:
        raise ValueError(f"V26b L1 {split} support differs from frozen mixed contract")
    if l2_dataset.label_counts() != EXPECTED_PARENT_COUNTS[split]:
        raise ValueError(f"V26b L2 {split} support differs from frozen mixed contract")
    alignment = verify_l1_l2_event_alignment(l1_dataset.base, l2_dataset.base)
    l1_loader = build_v26_open_set_audit_dataloader(l1_dataset, batch_size=1)
    l2_loader = build_frozen_l2_mixed_audit_dataloader(l2_dataset, batch_size=1)
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for l1_batch, l2_batch in zip(l1_loader, l2_loader, strict=True):
            l1_parent = str(l1_batch["parent_sample_ids"][0])
            l2_parent = str(l2_batch["parent_sample_ids"][0])
            if l1_parent != l2_parent:
                raise ValueError("V26b paired L1/L2 loader parent order differs")
            l1_tensors = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in l1_batch.items()
            }
            l2_tensors = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in l2_batch.items()
            }
            l1_output = l1(**l1_model_inputs(l1_tensors))
            l2_output = l2(**l2_model_inputs(l2_tensors))
            l1_probability = torch.softmax(l1_output["window_logits"], dim=-1)[0]
            l2_probability = torch.softmax(l2_output["window_logits"], dim=-1)[0]
            if not (torch.isfinite(l1_probability).all() and torch.isfinite(l2_probability).all()):
                raise ValueError("V26b L1/L2 raw routing emitted a non-finite probability")
            l1_id = int(l1_probability.argmax().item())
            l2_id = int(l2_probability.argmax().item())
            label = str(l1_batch["base_label_names"][0])
            if label != str(l2_batch["label_names"][0]):
                raise ValueError("V26b L1/L2 labels differ after key alignment")
            l2_values, _ = torch.topk(l2_probability, k=2)
            action = _route_action(l1_id, l2_id)
            row = {
                "split": split,
                "parent_sample_id": l1_parent,
                "label_name": label,
                "event_count": int(l1_batch["lengths"][0].item()),
                "l1_route": "KNOWN_TARGET_EVIDENCE" if l1_id == 0 else "KNOWN_CLUTTER_EVIDENCE",
                "l1_target_probability": float(l1_probability[0].item()),
                "l1_clutter_probability": float(l1_probability[1].item()),
                "l1_confidence": float(l1_probability[l1_id].item()),
                "l2_top_subclass": DBB_NAMES[l2_id],
                "l2_top_probability": float(l2_values[0].item()),
                "l2_margin": float((l2_values[0] - l2_values[1]).item()),
                "raw_route_action": action,
                "raw_route_correct_for_known_truth": _known_route_correct(label, action),
                "raw_route_unknown_false_accept": label == UNKNOWN_LABEL,
                **_base_metadata(l1_dataset, l1_parent),
            }
            rows.append(row)
    if len(rows) != len(l1_dataset) or len({row["parent_sample_id"] for row in rows}) != len(rows):
        raise ValueError("V26b raw routing does not cover each parent exactly once")
    return {"alignment": alignment, "rows": rows}


def _safe_divide(numerator: float, denominator: float) -> float | None:
    return float(numerator / denominator) if denominator else None


def _known_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    known = [row for row in rows if row["label_name"] != UNKNOWN_LABEL]
    if not known:
        raise ValueError("V26b cannot summarize empty Known route set")
    labels = (*DBB_NAMES, "ClutterTarget")
    matrix = np.zeros((4, 4), dtype=np.int64)
    index = {label: value for value, label in enumerate(labels)}
    for row in known:
        matrix[index[str(row["label_name"])] , index[str(row["raw_route_action"])]] += 1
    per_class: list[dict[str, Any]] = []
    for class_id, label in enumerate(labels):
        tp = int(matrix[class_id, class_id])
        precision = _safe_divide(tp, int(matrix[:, class_id].sum()))
        recall = _safe_divide(tp, int(matrix[class_id].sum()))
        f1 = _safe_divide(2.0 * precision * recall, precision + recall) if precision is not None and recall is not None and precision + recall else 0.0
        per_class.append({"class": label, "precision": precision, "recall": recall, "f1": f1, "support": int(matrix[class_id].sum())})
    correct = int(np.trace(matrix))
    return {
        "support": len(known),
        "accuracy": float(correct / len(known)),
        "macro_f1": float(np.mean([float(row["f1"]) for row in per_class])),
        "per_class": per_class,
        "confusion_matrix_rows_true_columns_raw_route_action": matrix.tolist(),
    }


def _unknown_routes(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    unknown = [row for row in rows if row["label_name"] == UNKNOWN_LABEL]
    counts = Counter(str(row["raw_route_action"]) for row in unknown)
    l1_routes = Counter(str(row["l1_route"]) for row in unknown)
    l2_top = Counter(str(row["l2_top_subclass"]) for row in unknown)
    return {
        "parent_support": len(unknown),
        "unknown_rejection_available": False,
        "unknown_rejected_parent_count": 0,
        "unknown_false_accept_rate": 1.0 if unknown else None,
        "raw_action_routes": {label: int(counts.get(label, 0)) for label in (*DBB_NAMES, "ClutterTarget")},
        "l1_raw_routes": {label: int(l1_routes.get(label, 0)) for label in ("KNOWN_TARGET_EVIDENCE", "KNOWN_CLUTTER_EVIDENCE")},
        "l2_raw_top_subclasses": {label: int(l2_top.get(label, 0)) for label in DBB_NAMES},
        "stage_specific_unknown_rejection_claim": "not_available_without human-reviewed Unknown first-action registry and cross-fitted L3 thresholds",
    }


def _bootstrap(rows: Sequence[Mapping[str, Any]], *, seed: int) -> dict[str, Any]:
    known = [row for row in rows if row["label_name"] != UNKNOWN_LABEL]
    strata = {
        label: np.asarray([index for index, row in enumerate(known) if row["label_name"] == label], dtype=np.int64)
        for label in (*KNOWN_LABELS,)
    }
    if any(indices.size == 0 for indices in strata.values()):
        raise ValueError("V26b bootstrap lacks a known-label stratum")
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = defaultdict(list)
    for _ in range(BOOTSTRAP_REPLICATES):
        selected = np.concatenate([rng.choice(indices, size=indices.size, replace=True) for indices in strata.values()])
        metrics = _known_metrics([known[int(index)] for index in selected])
        per_class = {row["class"]: row for row in metrics["per_class"]}
        for name in ("accuracy", "macro_f1"):
            values[name].append(float(metrics[name]))
        for name in ("precision", "recall", "f1"):
            values[f"clutter_{name}"].append(float(per_class["ClutterTarget"][name]))
    return {
        "method": "stratified parent bootstrap by known base label",
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": seed,
        "confidence_level": 0.95,
        "stratum_support": {label: int(indices.size) for label, indices in strata.items()},
        "metrics": {name: {"median": float(np.median(value)), "lower_95": float(np.quantile(value, 0.025)), "upper_95": float(np.quantile(value, 0.975))} for name, value in values.items()},
    }


def _capture_strata(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    capture = defaultdict(lambda: {"known": 0, "unknown": 0, "unknown_l1_target": 0, "unknown_l1_clutter": 0})
    for row in rows:
        value = capture[str(row["capture_id"])]
        if row["label_name"] == UNKNOWN_LABEL:
            value["unknown"] += 1
            value["unknown_l1_target"] += int(row["l1_route"] == "KNOWN_TARGET_EVIDENCE")
            value["unknown_l1_clutter"] += int(row["l1_route"] == "KNOWN_CLUTTER_EVIDENCE")
        else:
            value["known"] += 1
    return {"capture_count": len(capture), "rows": [{"capture_id": key, **value} for key, value in sorted(capture.items()) if value["unknown"]]}


def run(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    prepared = _require_preparation(root)
    l1, l2, device, configs = _load_models(root, prepared)
    validation = _run_split(
        root,
        split="val",
        l1=l1,
        l2=l2,
        device=device,
        configs=configs,
        l1_cache=root / V26_AUDIT_VALIDATION_CACHE,
        l2_physical_cache=root / V26B_VALIDATION_PHYSICAL_CACHE,
        l2_micro_cache=root / V26B_VALIDATION_MICRO_CACHE,
    )
    # Test cache materialization/revalidation is deliberately deferred until
    # both frozen checkpoints have been hash-verified and loaded.  The V26
    # course cache may already exist from its L1-only audit; this call still
    # proves equality against the frozen post-checkpoint cache after the L2
    # endpoint is loaded, without using any test label to tune a threshold.
    l1_test_cache, l1_test_cache_statistics = materialize_test_audit_cache(root)
    test_cache = materialize_test_l2_mixed_cache(root)
    test = _run_split(
        root,
        split="test",
        l1=l1,
        l2=l2,
        device=device,
        configs=configs,
        l1_cache=l1_test_cache,
        l2_physical_cache=root / V26B_TEST_PHYSICAL_CACHE,
        l2_micro_cache=root / V26B_TEST_MICRO_CACHE,
    )
    all_rows = [*validation["rows"], *test["rows"]]
    _write_parent_details(root / PARENT_DETAIL_CSV, all_rows)
    report = {
        "status": "completed_frozen_l1_l2_raw_routing_audit",
        "schema_version": "RADAR-LN3-V26b-frozen-l1-l2-raw-routing-audit-v1",
        "scope": "frozen raw routing only; no calibrated L3 decision, open-set rejection or deployment claim",
        "component_bindings": prepared["registry"]["component_bindings"],
        "l3_registry": {
            "path": str(root / V26B_L3_REGISTRY),
            "sha256": sha256_file(root / V26B_L3_REGISTRY),
            "calibration_status": prepared["registry"]["calibration"]["status"],
            "final_actions_enabled": prepared["registry"]["final_actions_enabled"],
        },
        "test_used_for_model_or_threshold_selection": False,
        "validation": {
            "alignment": validation["alignment"],
            "known_raw_route_metrics": _known_metrics(validation["rows"]),
            "unknown_raw_route_diagnostics": _unknown_routes(validation["rows"]),
            "parent_bootstrap": _bootstrap(validation["rows"], seed=BOOTSTRAP_SEED),
            "capture_strata": _capture_strata(validation["rows"]),
        },
        "test": {
            "alignment": test["alignment"],
            "l1_feature_cache": {
                "path": str(l1_test_cache),
                "sha256": sha256_file(l1_test_cache),
                "statistics": l1_test_cache_statistics,
            },
            "l2_feature_cache": test_cache,
            "known_raw_route_metrics": _known_metrics(test["rows"]),
            "unknown_raw_route_diagnostics": _unknown_routes(test["rows"]),
            "parent_bootstrap": _bootstrap(test["rows"], seed=BOOTSTRAP_SEED + 1),
            "capture_strata": _capture_strata(test["rows"]),
            "used_for_model_or_threshold_selection": False,
        },
        "records": {"parent_detail_csv": str(root / PARENT_DETAIL_CSV), "parent_detail_csv_sha256": sha256_file(root / PARENT_DETAIL_CSV), "row_count": len(all_rows)},
        "unknown_semantics": {
            "unknown_is_model_class": False,
            "raw_l1_or_l2_route_is_not_unknown_rejection": True,
            "pending_abstain_available": False,
            "pending_is_unknown_rejection_success": False,
            "stage_resolved_unknown_metrics": "not_available_without human-reviewed first-action registry and cross-fitted L3 calibration",
        },
        "OOF": "not_executed",
        "L3_calibration": "not_executed",
        "formal_rejection": "not_executed",
        "stable": "not_executed",
        "P95": "not_executed",
        "deployment_eligible": False,
    }
    _write_json(root / OUTPUT, report)
    return {"status": report["status"], "report": str(root / OUTPUT), "report_sha256": sha256_file(root / OUTPUT), "parent_detail_csv": str(root / PARENT_DETAIL_CSV), "test": "completed_frozen_raw_routing_audit_only"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate V26b frozen L1/L2 raw routing.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    print(json.dumps(run(args.project_root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()