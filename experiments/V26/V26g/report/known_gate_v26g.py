"""Run the frozen V26g quality-aware Pending/Abstain audit.

This module reruns only the immutable V26f inference path and verifies the
replay against V26f's sealed report before applying V26g's availability-only
action view.  No train step, gradient, cache write, normalization, score fit,
or threshold fit exists in this module.
"""

from __future__ import annotations

import argparse
import csv
import gc
import io
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from radar_ln3.known_gate_v26f import (
    INDEX,
    L1_TEST_CACHE,
    L1_VALIDATION_CACHE,
    L2_TEST_MICRO_CACHE,
    L2_TEST_PHYSICAL_CACHE,
    L2_VALIDATION_MICRO_CACHE,
    L2_VALIDATION_PHYSICAL_CACHE,
    V26C_CONTRACT,
    V26E_CONTRACT,
    selected_event_sample_ids,
)
from radar_ln3.known_gate_v26g import BOOTSTRAP_SEED, CONTRACT, EXPECTED_PARENT_COUNTS, FOLD_COUNT, MAX_DETAIL_BYTES, MAX_REPORT_BYTES, MIXED_LABELS, PARENT_DETAIL_CSV, QUALITY_POLICY, READINESS, REPORT, REPORT_SCHEMA, V26F_PARENT_DETAIL_CSV, V26F_REPORT, quality_pending_reasons, read_json, relative_path, require_bound_file, sha256_file, validate_prepared_contract, write_immutable_text
from radar_ln3.open_set_architecture import ProductAction, resolve_semantic_action
from report.known_gate_v26f import (
    _bootstrap,
    _capture_strata,
    _compact_split,
    _fold_summary_metrics,
    _l1_diagnostics,
    _l2_diagnostics,
    _load_fold_models,
    _metrics,
    _read_thresholds,
    _run_split,
)
from train.known_gate_v26c_oof import _runtime_l2_config
from train.known_gate_v26e_oof import _runtime_l1_config


UNKNOWN_LABEL = "UnknownTarget"
FLOAT_DETAIL_FIELDS = {
    "l1_target_probability",
    "l1_clutter_probability",
    "l2_subclass_probability",
}
BOOLEAN_DETAIL_FIELDS = {
    "known_target_accepted",
    "known_clutter_accepted",
    "l2_evidence_used_for_action",
    "l2_subclass_accepted",
}
SOURCE_DETAIL_FIELDS = (
    "split",
    "fold_id",
    "parent_sample_id",
    "label_name",
    "track_id",
    "capture_id",
    "provenance_component_id",
    "parent_length",
    "l1_route",
    "l1_target_probability",
    "l1_clutter_probability",
    "known_target_accepted",
    "known_clutter_accepted",
    "l2_evidence_used_for_action",
    "l2_top_subclass",
    "l2_subclass_probability",
    "l2_subclass_accepted",
    "semantic_action",
    "semantic_reason",
)


def _parse_bool(value: str) -> bool:
    text = str(value).strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    raise ValueError(f"V26g source detail boolean is invalid: {value!r}")


def _detail_key(row: Mapping[str, Any]) -> tuple[str, int, str]:
    return (str(row["split"]), int(row["fold_id"]), str(row["parent_sample_id"]))


def _read_source_parent_details(root: Path) -> dict[tuple[str, int, str], dict[str, str]]:
    path = require_bound_file(root, V26F_PARENT_DETAIL_CSV, "V26f source parent detail")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != SOURCE_DETAIL_FIELDS:
            raise ValueError("V26g V26f parent detail columns differ")
        details: dict[tuple[str, int, str], dict[str, str]] = {}
        for raw in reader:
            row = {str(key): str(value) for key, value in dict(raw).items()}
            key = _detail_key(row)
            if key in details:
                raise ValueError("V26g V26f parent detail repeats a parent key")
            if row["split"] not in {"val", "test"} or int(row["fold_id"]) not in range(FOLD_COUNT):
                raise ValueError("V26g V26f parent detail has invalid split/fold")
            details[key] = row
    expected = sum(sum(counts.values()) for counts in EXPECTED_PARENT_COUNTS.values()) * FOLD_COUNT
    if len(details) != expected:
        raise ValueError("V26g V26f parent detail support differs")
    return details


def _quality_lookup(root: Path, *, split: str) -> dict[tuple[str, int], dict[str, Any]]:
    """Build an exact parent/event lookup from V26f-bound frozen caches."""

    if split not in {"val", "test"}:
        raise ValueError(f"V26g unsupported split: {split}")
    cache_paths = (
        L1_VALIDATION_CACHE if split == "val" else L1_TEST_CACHE,
        L2_VALIDATION_PHYSICAL_CACHE if split == "val" else L2_TEST_PHYSICAL_CACHE,
        L2_VALIDATION_MICRO_CACHE if split == "val" else L2_TEST_MICRO_CACHE,
    )
    records: list[tuple[str, int, str, str]] = []
    with (root / INDEX).open("r", encoding="utf-8-sig", newline="") as handle:
        for raw in csv.DictReader(handle):
            if str(raw.get("split_name")) != split or str(raw.get("label_name")) not in MIXED_LABELS:
                continue
            parent_id = str(raw.get("parent_sample_id") or "")
            sample_id = str(raw.get("sample_id") or "")
            label_name = str(raw.get("label_name") or "")
            try:
                event_index = int(str(raw.get("causal_event_index")))
            except (TypeError, ValueError) as exc:
                raise ValueError("V26g source index causal event index is invalid") from exc
            if not parent_id or not sample_id or event_index < 0:
                raise ValueError("V26g source index identity is invalid")
            records.append((parent_id, event_index, sample_id, label_name))
    records.sort(key=lambda item: (item[0], item[1]))
    expected_ids = selected_event_sample_ids(root, split=split)
    if [item[2] for item in records] != expected_ids:
        raise ValueError("V26g source index/cache event order differs from V26f")
    if len({(parent, event) for parent, event, _, _ in records}) != len(records):
        raise ValueError("V26g source index repeats a parent/event key")

    loaded: list[tuple[np.ndarray, np.ndarray]] = []
    for ordinal, cache_path in enumerate(cache_paths):
        with np.load(root / cache_path, allow_pickle=False) as archive:
            identifiers = np.asarray(archive["sample_ids"])
            valid = np.asarray(archive["valid"])
        if identifiers.ndim != 1 or [str(value) for value in identifiers.tolist()] != expected_ids:
            raise ValueError("V26g frozen quality cache identifiers differ")
        if valid.dtype != np.bool_:
            raise ValueError("V26g frozen quality cache valid dtype differs")
        if ordinal == 0:
            if valid.shape != (len(expected_ids), 1):
                raise ValueError("V26g L1 course validity shape differs")
        elif valid.shape != (len(expected_ids),):
            raise ValueError("V26g L2 validity shape differs")
        loaded.append((identifiers, valid))
    course_valid = loaded[0][1][:, 0]
    physical_valid = loaded[1][1]
    micro_valid = loaded[2][1]
    if not np.array_equal(physical_valid, micro_valid):
        raise ValueError("V26g RD physical and micro-Doppler validity masks differ")
    return {
        (parent_id, event_index): {
            "sample_id": sample_id,
            "label_name": label_name,
            "known_gate_feature_valid": bool(course_valid[index]),
            "rd_physical_valid": bool(physical_valid[index]),
            "micro_doppler_valid": bool(micro_valid[index]),
        }
        for index, (parent_id, event_index, sample_id, label_name) in enumerate(records)
    }


def _attach_quality_view(
    replay_rows: Sequence[Mapping[str, Any]],
    *,
    lookup: Mapping[tuple[str, int], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for raw in replay_rows:
        row = dict(raw)
        key = (str(row["parent_sample_id"]), int(row["event_index"]))
        validity = dict(lookup.get(key) or {})
        if not validity:
            raise ValueError(f"V26g lacks frozen validity for parent/event: {key}")
        if str(validity["label_name"]) != str(row["label_name"]):
            raise ValueError("V26g frozen validity label alignment differs")
        row["sample_id"] = str(validity["sample_id"])
        row["known_gate_feature_valid"] = bool(validity["known_gate_feature_valid"])
        row["rd_physical_valid"] = bool(validity["rd_physical_valid"])
        row["micro_doppler_valid"] = bool(validity["micro_doppler_valid"])
        reasons = quality_pending_reasons(
            window_length=int(row["window_length"]),
            known_gate_feature_valid=bool(row["known_gate_feature_valid"]),
            known_target_accepted=bool(row["known_target_accepted"]),
            rd_physical_valid=bool(row["rd_physical_valid"]),
            micro_doppler_valid=bool(row["micro_doppler_valid"]),
        )
        quality_eligible = not reasons
        dbb_subclass = (
            ProductAction(str(row["l2_top_subclass"]))
            if bool(row["known_target_accepted"])
            else None
        )
        decision = resolve_semantic_action(
            observation_eligible=quality_eligible,
            known_target_accepted=bool(row["known_target_accepted"]),
            known_clutter_accepted=bool(row["known_clutter_accepted"]),
            dbb_subclass=dbb_subclass,
            dbb_subclass_accepted=bool(row["l2_subclass_accepted"]),
        )
        if reasons and decision.action is not ProductAction.PENDING_ABSTAIN:
            raise AssertionError("V26g quality-ineligible action is not Pending/Abstain")
        if not reasons and (
            decision.action.value != str(row["semantic_action"])
            or decision.reason != str(row["semantic_reason"])
        ):
            raise AssertionError("V26g quality-eligible action no longer matches V26f semantic resolver")
        row["replay_semantic_action"] = str(row["semantic_action"])
        row["replay_semantic_reason"] = str(row["semantic_reason"])
        row["quality_policy_eligible"] = bool(quality_eligible)
        row["quality_pending_reasons"] = list(reasons)
        row["quality_semantic_action"] = decision.action.value
        row["quality_semantic_reason"] = (
            "quality_evidence_unavailable:" + "+".join(reasons)
            if reasons
            else decision.reason
        )
        row["quality_forced_pending"] = bool(reasons)
        row["quality_action_changed"] = (
            row["replay_semantic_action"] != row["quality_semantic_action"]
        )
        row["quality_action_changed_to_pending"] = bool(
            row["quality_action_changed"]
            and row["quality_semantic_action"] == ProductAction.PENDING_ABSTAIN.value
        )
        result.append(row)
    if len(result) != len(lookup):
        raise ValueError("V26g replay/quality lookup event coverage differs")
    return result


def _scopes(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in rows:
        grouped[str(raw["parent_sample_id"])].append(dict(raw))
    parent_rows: list[dict[str, Any]] = []
    eligible_prefix_rows: list[dict[str, Any]] = []
    for parent_id, group in grouped.items():
        group.sort(key=lambda row: int(row["event_index"]))
        if [int(row["event_index"]) for row in group] != list(range(len(group))):
            raise ValueError(f"V26g causal order is not contiguous: {parent_id}")
        parent_rows.append(dict(group[-1]))
        eligible_prefix_rows.extend(
            dict(row)
            for row in group
            if bool(row["product_action_eligible"]) and row is not group[-1]
        )
    parent_rows.sort(key=lambda row: str(row["parent_sample_id"]))
    eligible_prefix_rows.sort(
        key=lambda row: (str(row["parent_sample_id"]), int(row["event_index"]))
    )
    return {
        "window": [dict(row) for row in rows],
        "eligible_prefix": eligible_prefix_rows,
        "parent_final": parent_rows,
    }


def _action_view(rows: Sequence[Mapping[str, Any]], *, action: str, reason: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        row["semantic_action"] = str(row[action])
        row["semantic_reason"] = str(row[reason])
        result.append(row)
    return result


def _l1_acceptance_route(row: Mapping[str, Any]) -> str:
    target = bool(row["known_target_accepted"])
    clutter = bool(row["known_clutter_accepted"])
    if target and clutter:
        return "L1_ACCEPTANCE_CONFLICT"
    if target:
        return "L1_KNOWN_TARGET_ACCEPTED"
    if clutter:
        return "L1_KNOWN_CLUTTER_ACCEPTED"
    return "L1_NEITHER_ACCEPTED"


def _quality_diagnostics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    support = len(rows)
    by_label: dict[str, dict[str, Any]] = {}
    by_route: dict[str, dict[str, Any]] = {}
    transitions: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    for label in MIXED_LABELS:
        label_rows = [row for row in rows if str(row["label_name"]) == label]
        by_label[label] = {
            "support": len(label_rows),
            "quality_policy_eligible_count": sum(bool(row["quality_policy_eligible"]) for row in label_rows),
            "quality_forced_pending_count": sum(bool(row["quality_forced_pending"]) for row in label_rows),
            "replay_pending_count": sum(
                str(row["replay_semantic_action"]) == ProductAction.PENDING_ABSTAIN.value
                for row in label_rows
            ),
            "quality_pending_count": sum(
                str(row["quality_semantic_action"]) == ProductAction.PENDING_ABSTAIN.value
                for row in label_rows
            ),
            "action_changed_to_pending_count": sum(
                bool(row["quality_action_changed_to_pending"]) for row in label_rows
            ),
        }
    for row in rows:
        route = _l1_acceptance_route(row)
        item = by_route.setdefault(
            route,
            {
                "support": 0,
                "quality_policy_eligible_count": 0,
                "quality_forced_pending_count": 0,
                "action_changed_to_pending_count": 0,
            },
        )
        item["support"] += 1
        item["quality_policy_eligible_count"] += int(bool(row["quality_policy_eligible"]))
        item["quality_forced_pending_count"] += int(bool(row["quality_forced_pending"]))
        item["action_changed_to_pending_count"] += int(bool(row["quality_action_changed_to_pending"]))
        transitions[
            f"{row['replay_semantic_action']}->{row['quality_semantic_action']}"
        ] += 1
        for reason in row["quality_pending_reasons"]:
            reasons[str(reason)] += 1
    return {
        "support": support,
        "quality_policy_eligible_count": sum(bool(row["quality_policy_eligible"]) for row in rows),
        "quality_forced_pending_count": sum(bool(row["quality_forced_pending"]) for row in rows),
        "replay_pending_count": sum(
            str(row["replay_semantic_action"]) == ProductAction.PENDING_ABSTAIN.value
            for row in rows
        ),
        "quality_pending_count": sum(
            str(row["quality_semantic_action"]) == ProductAction.PENDING_ABSTAIN.value
            for row in rows
        ),
        "action_changed_count": sum(bool(row["quality_action_changed"]) for row in rows),
        "action_changed_to_pending_count": sum(
            bool(row["quality_action_changed_to_pending"]) for row in rows
        ),
        "reason_counts": dict(sorted(reasons.items())),
        "transition_counts": dict(sorted(transitions.items())),
        "by_truth_label_reporting_only": by_label,
        "by_l1_acceptance_route": dict(sorted(by_route.items())),
        "validity_counts": {
            "known_gate_course_invalid": sum(not bool(row["known_gate_feature_valid"]) for row in rows),
            "l2_rd_pair_invalid": sum(not bool(row["rd_physical_valid"]) for row in rows),
            "target_route_l2_invalid": sum(
                bool(row["known_target_accepted"]) and not bool(row["rd_physical_valid"])
                for row in rows
            ),
            "rd_physical_micro_mask_mismatch": sum(
                bool(row["rd_physical_valid"]) != bool(row["micro_doppler_valid"])
                for row in rows
            ),
        },
    }


def _quality_compact(rows: Sequence[Mapping[str, Any]], *, split: str) -> dict[str, Any]:
    scopes = _scopes(rows)
    window = _action_view(scopes["window"], action="quality_semantic_action", reason="quality_semantic_reason")
    eligible = _action_view(
        scopes["eligible_prefix"], action="quality_semantic_action", reason="quality_semantic_reason"
    )
    parent = _action_view(scopes["parent_final"], action="quality_semantic_action", reason="quality_semantic_reason")
    seed = BOOTSTRAP_SEED + (0 if split == "val" else 1)
    return {
        "event_count": len(window),
        "parent_count": len(parent),
        "eligible_prefix_event_count": len(eligible),
        "metrics": {
            "window_mixed": _metrics(window, scope="window"),
            "window_known_only": _metrics(
                [row for row in window if str(row["label_name"]) != UNKNOWN_LABEL],
                scope="window_known_only",
            ),
            "eligible_prefix_mixed": _metrics(eligible, scope="eligible_prefix"),
            "eligible_prefix_parent_balanced_mixed": _metrics(
                eligible, scope="eligible_prefix_parent_balanced", parent_balanced=True
            ),
            "parent_final_mixed": _metrics(parent, scope="parent_final"),
            "parent_final_known_only": _metrics(
                [row for row in parent if str(row["label_name"]) != UNKNOWN_LABEL],
                scope="parent_final_known_only",
            ),
        },
        "quality_diagnostics": {
            "window": _quality_diagnostics(scopes["window"]),
            "eligible_prefix": _quality_diagnostics(scopes["eligible_prefix"]),
            "parent_final": _quality_diagnostics(scopes["parent_final"]),
        },
        "model_forward_diagnostics_replayed": {
            "l1_window": _l1_diagnostics(window),
            "l1_parent_final": _l1_diagnostics(parent),
            "l2_window": _l2_diagnostics(window),
            "l2_parent_final": _l2_diagnostics(parent),
        },
        "capture_strata_parent_final": _capture_strata(parent),
        "parent_bootstrap": _bootstrap(parent, seed=seed),
    }


def _assert_replay_parent_detail(
    rows: Sequence[Mapping[str, Any]], *, source_details: Mapping[tuple[str, int, str], Mapping[str, str]]
) -> set[tuple[str, int, str]]:
    """Check one fold/split subset against V26f's all-fold source CSV."""

    seen: set[tuple[str, int, str]] = set()
    for row in rows:
        key = _detail_key(row)
        if key in seen:
            raise ValueError("V26g replay repeats a V26f parent-detail key")
        seen.add(key)
        source = source_details.get(key)
        if source is None:
            raise ValueError("V26g replay parent is absent from V26f source detail")
        for field in SOURCE_DETAIL_FIELDS:
            actual = row.get(field)
            expected = source[field]
            if field in FLOAT_DETAIL_FIELDS:
                if not np.isclose(float(actual), float(expected), rtol=0.0, atol=1.0e-7):
                    raise AssertionError(f"V26g V26f parent-detail float mismatch: {key}; {field}")
            elif field in BOOLEAN_DETAIL_FIELDS:
                if bool(actual) != _parse_bool(expected):
                    raise AssertionError(f"V26g V26f parent-detail boolean mismatch: {key}; {field}")
            elif str(actual) != str(expected):
                raise AssertionError(f"V26g V26f parent-detail mismatch: {key}; {field}")
    return seen


def _parent_detail_csv(root: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    fields = (
        "split",
        "fold_id",
        "parent_sample_id",
        "label_name",
        "track_id",
        "capture_id",
        "provenance_component_id",
        "parent_length",
        "sample_id",
        "known_gate_feature_valid",
        "rd_physical_valid",
        "micro_doppler_valid",
        "quality_policy_eligible",
        "quality_pending_reasons",
        "quality_forced_pending",
        "l1_route",
        "l1_target_probability",
        "l1_clutter_probability",
        "known_target_accepted",
        "known_clutter_accepted",
        "l2_evidence_used_for_action",
        "l2_top_subclass",
        "l2_subclass_probability",
        "l2_subclass_accepted",
        "replay_semantic_action",
        "replay_semantic_reason",
        "quality_semantic_action",
        "quality_semantic_reason",
        "quality_action_changed",
        "quality_action_changed_to_pending",
    )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for raw in sorted(rows, key=lambda row: (str(row["split"]), int(row["fold_id"]), str(row["parent_sample_id"]))):
        row = dict(raw)
        row["quality_pending_reasons"] = ";".join(str(item) for item in row["quality_pending_reasons"])
        writer.writerow({field: row.get(field) for field in fields})
    return write_immutable_text(
        root / PARENT_DETAIL_CSV,
        stream.getvalue(),
        max_bytes=MAX_DETAIL_BYTES,
        description="parent detail CSV",
    )


def _quality_fold_summary(
    folds: Sequence[Mapping[str, Any]], *, split: str, scope: str
) -> dict[str, Any]:
    source_folds: list[dict[str, Any]] = []
    for fold in folds:
        entry = dict(dict(fold[split])["quality_aware"])
        source_folds.append({"fold_id": fold["fold_id"], split: entry})
    return _fold_summary_metrics(source_folds, split=split, scope=scope)


def run(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    prepared_contract, _prepared_readiness = validate_prepared_contract(root)
    source_report = read_json(root / V26F_REPORT, description="V26f source report")
    thresholds_by_fold = _read_thresholds(root)
    source_details = _read_source_parent_details(root)
    quality_lookups = {split: _quality_lookup(root, split=split) for split in ("val", "test")}
    all_folds: list[dict[str, Any]] = []
    parent_details: list[dict[str, Any]] = []
    replay_parent_keys: set[tuple[str, int, str]] = set()
    try:
        for fold_id in range(FOLD_COUNT):
            l1_model, l2_model, device, model_binding = _load_fold_models(
                root, fold_id=fold_id, prepared_contract=read_json(
                    root / "data/processed/RADAR-LN3-V26f-cq08-frozen-mixed-open-set-audit-contract.json",
                    description="V26f source contract",
                )
            )
            source_fold = next(
                dict(row) for row in list(source_report.get("folds") or ()) if int(row["fold_id"]) == fold_id
            )
            if dict(source_fold.get("thresholds") or {}) != dict(thresholds_by_fold[fold_id]):
                raise AssertionError("V26g replay thresholds differ from V26f source report")
            if dict(source_fold.get("model_bindings") or {}) != dict(model_binding):
                raise AssertionError("V26g replay checkpoint/config binding differs from V26f source report")
            v26e_contract = read_json(root / V26E_CONTRACT, description="V26e source contract")
            v26c_contract = read_json(root / V26C_CONTRACT, description="V26c source contract")
            e_fold = next(row for row in v26e_contract["folds"] if int(row["fold_id"]) == fold_id)
            c_fold = next(row for row in v26c_contract["folds"] if int(row["fold_id"]) == fold_id)
            l1_config, _, _ = _runtime_l1_config(root, fold_id, dict(e_fold["l1"]))
            l2_config, _, _ = _runtime_l2_config(root, fold_id, dict(c_fold["l2"]))
            split_entries: dict[str, Any] = {}
            for short_split, source_split in (("val", "validation"), ("test", "test")):
                raw_result = _run_split(
                    root,
                    split=short_split,
                    fold_id=fold_id,
                    l1_model=l1_model,
                    l2_model=l2_model,
                    device=device,
                    l1_config=l1_config,
                    l2_config=l2_config,
                    thresholds=thresholds_by_fold[fold_id],
                )
                replay = _compact_split(raw_result)
                if replay != dict(source_fold[source_split]):
                    raise AssertionError(
                        f"V26g V26f replay summary differs: fold={fold_id}; split={short_split}"
                    )
                quality_rows = _attach_quality_view(
                    raw_result["event_rows"], lookup=quality_lookups[short_split]
                )
                scopes = _scopes(quality_rows)
                current_parent_keys = _assert_replay_parent_detail(
                    scopes["parent_final"], source_details=source_details
                )
                if replay_parent_keys.intersection(current_parent_keys):
                    raise AssertionError("V26g V26f parent detail was replayed more than once")
                replay_parent_keys.update(current_parent_keys)
                parent_details.extend(scopes["parent_final"])
                split_entries[source_split] = {
                    "v26f_replay": replay,
                    "quality_aware": _quality_compact(quality_rows, split=short_split),
                }
            all_folds.append(
                {
                    "fold_id": int(fold_id),
                    "thresholds": dict(thresholds_by_fold[fold_id]),
                    "model_bindings": model_binding,
                    "device": str(device),
                    "validation": split_entries["validation"],
                    "test": split_entries["test"],
                }
            )
            del l1_model, l2_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if replay_parent_keys != set(source_details):
        raise AssertionError("V26g V26f parent detail was not replayed exactly once")
    detail_status = _parent_detail_csv(root, parent_details)
    aggregate = {
        "validation": {
            "v26f_replay_reference": dict(source_report["aggregate_fold_diagnostics"]["validation"]),
            "quality_aware": {
                "parent_final_mixed": _quality_fold_summary(
                    all_folds, split="validation", scope="parent_final_mixed"
                ),
                "eligible_prefix_parent_balanced_mixed": _quality_fold_summary(
                    all_folds,
                    split="validation",
                    scope="eligible_prefix_parent_balanced_mixed",
                ),
            },
        },
        "test": {
            "v26f_replay_reference": dict(source_report["aggregate_fold_diagnostics"]["test"]),
            "quality_aware": {
                "parent_final_mixed": _quality_fold_summary(
                    all_folds, split="test", scope="parent_final_mixed"
                ),
                "eligible_prefix_parent_balanced_mixed": _quality_fold_summary(
                    all_folds,
                    split="test",
                    scope="eligible_prefix_parent_balanced_mixed",
                ),
            },
        },
        "interpretation": "fold-level diagnostic mean/std/range; no pooled deployable model is created",
    }
    report = {
        "status": "completed_v26g_frozen_quality_aware_pending_audit",
        "schema_version": REPORT_SCHEMA,
        "experiment": "V26g frozen quality-aware Pending/Abstain audit",
        "scope": "CQ08 mixed validation/test frozen V26f replay plus availability-only action view",
        "contract_path": relative_path(root, root / CONTRACT),
        "contract_sha256": sha256_file(root / CONTRACT),
        "readiness_path": relative_path(root, root / READINESS),
        "readiness_sha256": sha256_file(root / READINESS),
        "source_v26f": {
            "report_path": relative_path(root, root / V26F_REPORT),
            "report_sha256": sha256_file(root / V26F_REPORT),
            "parent_detail_path": relative_path(root, root / V26F_PARENT_DETAIL_CSV),
            "parent_detail_sha256": sha256_file(root / V26F_PARENT_DETAIL_CSV),
            "parent_final_replay_rows_verified": len(replay_parent_keys),
            "summary_replay_verification": "exact V26f compact split equality for every fold/split",
        },
        "quality_policy": QUALITY_POLICY,
        "data": {
            "source": "CQ08",
            "mixed_labels": list(MIXED_LABELS),
            "validation_parent_counts": EXPECTED_PARENT_COUNTS["val"],
            "test_parent_counts": EXPECTED_PARENT_COUNTS["test"],
        },
        "fold_count": FOLD_COUNT,
        "folds": all_folds,
        "aggregate_fold_diagnostics": aggregate,
        "parent_detail_csv": relative_path(root, root / PARENT_DETAIL_CSV),
        "parent_detail_csv_sha256": sha256_file(root / PARENT_DETAIL_CSV),
        "parent_detail_csv_write": detail_status,
        "routing_semantics": {
            "l1_l2_l3_replay_is_identical_to_v26f_before_quality_view": True,
            "l1_known_target_route_requires_l2_validity_for_quality_aware_action": True,
            "l1_clutter_action_does_not_consume_l2_validity": True,
            "quality_rule_uses_no_truth_label": True,
            "pending_is_not_unknown_rejection_success": True,
            "tr_dt_valid_is_not_quality_gate": True,
            "rd_present_is_not_quality_gate": True,
        },
        "isolation": {
            "unknown_training_exposure": 0,
            "unknown_normalization_exposure": 0,
            "unknown_checkpoint_selection_exposure": 0,
            "unknown_threshold_fit_exposure": 0,
            "new_training": False,
            "new_threshold_fit": False,
            "new_feature_cache": False,
        },
        "not_executed": {
            "training": True,
            "oof_retraining": True,
            "normalization": True,
            "threshold_fitting": True,
            "feature_cache_rebuild": True,
            "deployment_registry": True,
            "cross_source": True,
            "full_production_stable": True,
            "p95": True,
        },
        "deployment_eligible": False,
    }
    write_immutable_text(
        root / REPORT,
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        max_bytes=MAX_REPORT_BYTES,
        description="report",
    )
    return {
        "status": report["status"],
        "report": str(root / REPORT),
        "report_sha256": sha256_file(root / REPORT),
        "parent_detail_csv": str(root / PARENT_DETAIL_CSV),
        "parent_detail_csv_sha256": sha256_file(root / PARENT_DETAIL_CSV),
        "fold_count": FOLD_COUNT,
        "training": "not_executed",
        "threshold_fitting": "not_executed",
        "deployment_eligible": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the V26g quality-aware Pending audit.")
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.project_root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()