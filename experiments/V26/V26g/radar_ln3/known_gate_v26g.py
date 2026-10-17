"""Frozen V26g quality-aware Pending/Abstain audit contract.

V26g is deliberately an inference-only action-policy audit over the frozen
V26f L1 -> L2 -> L3 replay.  It does not train, fit thresholds, alter feature
caches, or use truth labels to decide an action.  It adds only an explicit
evidence-availability gate before the existing strict-open-set resolver:

* at least three causal events must be available;
* the one-column L1 course feature for the current event must be valid; and
* only for an L1 KnownTarget-accepted route, both frozen L2 RD feature masks
  must be valid.

``tr_dt_valid`` is intentionally excluded: first events and same-second
observations have valid temporal semantics even when their delta-time flag is
false.  ``rd_present`` is also not promoted to a new policy variable here;
the already frozen L2 validity masks are the sole RD availability evidence.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .known_gate_v26f import BOOTSTRAP_SEED, EXPECTED_PARENT_COUNTS, FOLD_COUNT, L1_TEST_CACHE, L1_VALIDATION_CACHE, L2_TEST_MICRO_CACHE, L2_TEST_PHYSICAL_CACHE, L2_VALIDATION_MICRO_CACHE, L2_VALIDATION_PHYSICAL_CACHE, MAX_DETAIL_BYTES as V26F_MAX_DETAIL_BYTES, MAX_REPORT_BYTES as V26F_MAX_REPORT_BYTES, MIXED_LABELS, MINIMUM_EVIDENCE_EVENTS, PARENT_DETAIL_CSV as V26F_PARENT_DETAIL_CSV, CONTRACT as V26F_CONTRACT, READINESS as V26F_READINESS, REPORT as V26F_REPORT, sha256_file, validate_prepared_contract as validate_v26f_prepared_contract


CONTRACT = Path(
    "data/processed/RADAR-LN3-V26g-cq08-frozen-quality-aware-pending-audit-contract.json"
)
READINESS = Path(
    "src-v3/processed/report/RADAR-LN3-V26g-frozen-quality-aware-pending-audit-readiness.json"
)
REPORT = Path(
    "src-v3/processed/report/RADAR-LN3-V26g-frozen-quality-aware-pending-audit.json"
)
PARENT_DETAIL_CSV = Path(
    "src-v3/processed/report/RADAR-LN3-V26g-frozen-quality-aware-pending-audit-parent-details.csv"
)
DOCUMENT = Path("src-v3/docs/radar_ln3_v26g_quality_aware_pending_audit.md")

CONTRACT_SCHEMA = "RADAR-LN3-V26g-frozen-quality-aware-pending-audit-contract-v1"
READINESS_SCHEMA = "RADAR-LN3-V26g-frozen-quality-aware-pending-audit-readiness-v1"
REPORT_SCHEMA = "RADAR-LN3-V26g-frozen-quality-aware-pending-audit-v1"

MAX_REPORT_BYTES = max(8 * 1024 * 1024, int(V26F_MAX_REPORT_BYTES))
MAX_DETAIL_BYTES = max(4 * 1024 * 1024, int(V26F_MAX_DETAIL_BYTES))
MAX_CONTRACT_BYTES = 2 * 1024 * 1024
MAX_READINESS_BYTES = 512 * 1024

QUALITY_PENDING_REASONS = (
    "insufficient_event_count",
    "known_gate_course_feature_invalid",
    "target_route_l2_evidence_invalid",
)

# This dictionary is copied into the immutable contract and compared verbatim
# at every run.  It intentionally contains no threshold, label, score, or
# selected-result-dependent field.
QUALITY_POLICY: dict[str, Any] = {
    "policy_name": "availability_only_v1",
    "minimum_evidence_events": int(MINIMUM_EVIDENCE_EVENTS),
    "known_gate_course_feature_valid_required": True,
    "l2_validity_required_only_when_l1_known_target_accepted": True,
    "l2_validity_pair": "rd_physical_valid_and_micro_doppler_valid",
    "rd_present_used_as_quality_gate": False,
    "tr_dt_valid_used_as_quality_gate": False,
    "continuous_quality_scores_used_as_quality_gate": False,
    "truth_labels_used_for_policy": False,
    "threshold_fitting": "not_executed",
}

IMPLEMENTATION_FILES = (
    Path("experiments/V26/V26g/radar_ln3/known_gate_v26g.py"),
    Path("experiments/V26/V26g/build/known_gate_v26g.py"),
    Path("experiments/V26/V26g/report/known_gate_v26g.py"),
    Path("experiments/V26/V26g/scripts/v26g_known_gate_quality_pending_audit_pipeline.ps1"),
)


def resolve_path(root: Path, value: str | Path) -> Path:
    raw_text = str(value).strip()
    raw = Path(raw_text.replace("\\", "/"))
    return (raw if raw.is_absolute() else root / raw).resolve()


def relative_path(root: Path, value: str | Path) -> str:
    root_path = Path(root).resolve()
    path = resolve_path(root_path, value)
    try:
        return path.relative_to(root_path).as_posix()
    except ValueError as exc:
        raise ValueError(f"V26g path escapes the project root: {value}") from exc


def require_bound_file(
    root: Path,
    value: str | Path,
    description: str,
    *,
    expected_sha256: str | None = None,
    max_bytes: int = 64 * 1024 * 1024,
) -> Path:
    path = resolve_path(root, value)
    try:
        path.relative_to(Path(root).resolve())
    except ValueError as exc:
        raise ValueError(f"V26g {description} escapes the project root: {value}") from exc
    if not path.is_file():
        raise FileNotFoundError(f"V26g {description} is missing: {path}")
    if int(path.stat().st_size) > int(max_bytes):
        raise MemoryError(f"V26g {description} exceeds bounded size: {path}")
    if expected_sha256 is not None:
        expected = str(expected_sha256).strip().lower()
        if len(expected) != 64 or sha256_file(path) != expected:
            raise ValueError(f"V26g {description} hash differs: {path}")
    return path


def read_json(path: Path, *, description: str) -> dict[str, Any]:
    checked = Path(path)
    if not checked.is_file():
        raise FileNotFoundError(f"V26g {description} is missing: {checked}")
    if int(checked.stat().st_size) > 64 * 1024 * 1024:
        raise MemoryError(f"V26g {description} exceeds bounded size: {checked}")
    payload = json.loads(checked.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"V26g {description} must be a JSON object: {checked}")
    return payload


def write_immutable_text(path: Path, text: str, *, max_bytes: int, description: str) -> str:
    if len(text.encode("utf-8")) > int(max_bytes):
        raise MemoryError(f"V26g {description} exceeds bounded size: {path}")
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"refusing to overwrite a different V26g {description}: {path}")
        return "reused_identical"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists() and temporary.read_text(encoding="utf-8") != text:
        raise FileExistsError(
            f"refusing to replace a different interrupted V26g {description}: {temporary}"
        )
    if not temporary.exists():
        temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)
    return "created"


def write_immutable_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    max_bytes: int,
    description: str,
) -> str:
    return write_immutable_text(
        path,
        json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
        max_bytes=max_bytes,
        description=description,
    )


def implementation_sha256(root: Path) -> dict[str, str]:
    return {
        relative_path(root, path): sha256_file(
            require_bound_file(root, path, f"implementation {path}")
        )
        for path in IMPLEMENTATION_FILES
    }


def source_binding(root: Path, value: str | Path, *, description: str) -> dict[str, Any]:
    path = require_bound_file(root, value, description)
    return {
        "path": relative_path(root, path),
        "sha256": sha256_file(path),
        "size_bytes": int(path.stat().st_size),
    }


def quality_pending_reasons(
    *,
    window_length: int,
    known_gate_feature_valid: bool,
    known_target_accepted: bool,
    rd_physical_valid: bool,
    micro_doppler_valid: bool,
) -> tuple[str, ...]:
    """Return label-free evidence-availability failures in frozen order.

    The L2 validity pair is intentionally checked only on the actual model
    KnownTarget acceptance route.  L1 Clutter decisions therefore remain an
    L1-only action, as required by the strict-open-set architecture.
    """

    if int(window_length) < 1:
        raise ValueError("V26g window length must be positive")
    physical = bool(rd_physical_valid)
    micro = bool(micro_doppler_valid)
    if physical != micro:
        raise ValueError("V26g frozen L2 RD validity masks differ")
    reasons: list[str] = []
    if int(window_length) < int(MINIMUM_EVIDENCE_EVENTS):
        reasons.append("insufficient_event_count")
    if not bool(known_gate_feature_valid):
        reasons.append("known_gate_course_feature_invalid")
    if bool(known_target_accepted) and not (physical and micro):
        reasons.append("target_route_l2_evidence_invalid")
    return tuple(reasons)


def validate_quality_cache_relationship(root: Path) -> dict[str, Any]:
    """Verify the frozen availability masks before quality-gating them.

    V26f already validates cache hashes, schema, range and sample order.  This
    V26g-specific check establishes the extra policy fact that the two L2
    feature families are available together for each current event.
    """

    v26f_contract, _readiness = validate_v26f_prepared_contract(root)
    bindings = {
        str(dict(row).get("split", "")): dict(row)
        for row in list(dict(v26f_contract.get("data") or {}).get("cache_bindings") or ())
    }
    if set(bindings) != {"val", "test"}:
        raise ValueError("V26g V26f cache bindings must cover val and test")
    expected_paths = {
        "val": (L1_VALIDATION_CACHE, L2_VALIDATION_PHYSICAL_CACHE, L2_VALIDATION_MICRO_CACHE),
        "test": (L1_TEST_CACHE, L2_TEST_PHYSICAL_CACHE, L2_TEST_MICRO_CACHE),
    }
    summary: dict[str, Any] = {}
    for split, (expected_course, expected_physical, expected_micro) in expected_paths.items():
        entry = bindings[split]
        parts = (
            ("l1_course", expected_course, 2),
            ("l2_rd_physical4", expected_physical, 1),
            ("l2_micro_doppler8", expected_micro, 1),
        )
        loaded: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name, expected_path, rank in parts:
            binding = dict(entry.get(name) or {})
            if str(binding.get("path", "")) != relative_path(root, root / expected_path):
                raise ValueError(f"V26g {split} {name} path differs from V26f source")
            path = require_bound_file(
                root,
                str(binding.get("path", "")),
                f"{split} {name} cache",
                expected_sha256=str(binding.get("sha256", "")),
            )
            with np.load(path, allow_pickle=False) as archive:
                identifiers = np.asarray(archive["sample_ids"])
                valid = np.asarray(archive["valid"])
            if identifiers.ndim != 1 or valid.dtype != np.bool_:
                raise ValueError(f"V26g {split} {name} cache validity type differs")
            if rank == 2 and valid.shape != (len(identifiers), 1):
                raise ValueError(f"V26g {split} course validity shape differs")
            if rank == 1 and valid.shape != (len(identifiers),):
                raise ValueError(f"V26g {split} L2 validity shape differs")
            loaded[name] = (identifiers, valid)
        course_ids, course_valid = loaded["l1_course"]
        physical_ids, physical_valid = loaded["l2_rd_physical4"]
        micro_ids, micro_valid = loaded["l2_micro_doppler8"]
        if not (np.array_equal(course_ids, physical_ids) and np.array_equal(course_ids, micro_ids)):
            raise ValueError(f"V26g {split} quality-cache sample order differs")
        if not np.array_equal(physical_valid, micro_valid):
            raise ValueError(f"V26g {split} RD physical/micro validity masks differ")
        summary[split] = {
            "event_count": int(len(course_ids)),
            "known_gate_course_invalid_event_count": int((~course_valid[:, 0]).sum()),
            "l2_rd_pair_invalid_event_count": int((~physical_valid).sum()),
            "rd_physical_micro_validity_masks_equal": True,
            "sample_ids_sha256": hashlib.sha256(
                "\n".join(str(value) for value in course_ids.tolist()).encode("utf-8")
            ).hexdigest(),
        }
    return summary


def validate_v26f_source_report(root: Path) -> dict[str, Any]:
    """Validate V26f's immutable output before V26g replays it."""

    # This also verifies the V26f implementation binding, cache hashes and all
    # five frozen L1/L2 checkpoint bindings.
    contract, readiness = validate_v26f_prepared_contract(root)
    report_path = require_bound_file(root, V26F_REPORT, "V26f source report")
    detail_path = require_bound_file(root, V26F_PARENT_DETAIL_CSV, "V26f source parent detail")
    report = read_json(report_path, description="V26f source report")
    if (
        report.get("schema_version") != "RADAR-LN3-V26f-frozen-mixed-open-set-audit-v1"
        or report.get("status") != "completed_v26f_frozen_mixed_open_set_audit"
        or report.get("contract_sha256") != sha256_file(root / V26F_CONTRACT)
        or report.get("readiness_sha256") != sha256_file(root / V26F_READINESS)
        or report.get("parent_detail_csv_sha256") != sha256_file(detail_path)
        or report.get("deployment_eligible") is not False
        or dict(report.get("isolation") or {}).get("unknown_training_exposure") != 0
        or dict(report.get("isolation") or {}).get("unknown_threshold_fit_exposure") != 0
        or dict(report.get("routing_semantics") or {}).get("pending_is_not_unknown_rejection_success")
        is not True
    ):
        raise ValueError("V26g V26f source report is stale or outside frozen-audit scope")
    return {
        "v26f_contract_sha256": sha256_file(
            root / V26F_CONTRACT
        ),
        "v26f_readiness_sha256": sha256_file(root / V26F_READINESS),
        "v26f_report_sha256": sha256_file(report_path),
        "v26f_parent_detail_sha256": sha256_file(detail_path),
        "v26f_contract_schema": contract.get("schema_version"),
        "v26f_readiness_schema": readiness.get("schema_version"),
        "v26f_report_schema": report.get("schema_version"),
    }


def validate_prepared_contract(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate V26g and its complete immutable V26f source chain."""

    source = validate_v26f_source_report(root)
    quality_caches = validate_quality_cache_relationship(root)
    contract_path = root / CONTRACT
    readiness_path = root / READINESS
    contract = read_json(contract_path, description="V26g contract")
    readiness = read_json(readiness_path, description="V26g readiness")
    checks = dict(readiness.get("checks") or {})
    if (
        contract.get("schema_version") != CONTRACT_SCHEMA
        or readiness.get("schema_version") != READINESS_SCHEMA
        or readiness.get("status") != "passed"
        or readiness.get("contract_sha256") != sha256_file(contract_path)
        or not checks
        or not all(value is True for value in checks.values())
        or dict(contract.get("quality_policy") or {}) != QUALITY_POLICY
        or dict(contract.get("source_v26f") or {}) != source
        or dict(contract.get("quality_cache_relationship") or {}) != quality_caches
        or dict(contract.get("implementation_sha256") or {}) != implementation_sha256(root)
        or int(contract.get("fold_count", -1)) != int(FOLD_COUNT)
        or int(contract.get("unknown_training_exposure", -1)) != 0
        or int(contract.get("unknown_threshold_fit_exposure", -1)) != 0
        or contract.get("deployment_eligible") is not False
        or readiness.get("deployment_eligible") is not False
    ):
        raise ValueError("V26g prepared contract/readiness is invalid or stale")
    for raw in list(contract.get("source_bindings") or ()):
        item = dict(raw)
        path = require_bound_file(
            root,
            str(item.get("path", "")),
            f"source binding {item.get('name', '')}",
            expected_sha256=str(item.get("sha256", "")),
        )
        if int(item.get("size_bytes", -1)) != int(path.stat().st_size):
            raise ValueError("V26g source binding size differs")
    return contract, readiness


__all__ = ["BOOTSTRAP_SEED", "CONTRACT", "CONTRACT_SCHEMA", "DOCUMENT", "EXPECTED_PARENT_COUNTS", "FOLD_COUNT", "IMPLEMENTATION_FILES", "MAX_CONTRACT_BYTES", "MAX_DETAIL_BYTES", "MAX_READINESS_BYTES", "MAX_REPORT_BYTES", "MIXED_LABELS", "PARENT_DETAIL_CSV", "QUALITY_PENDING_REASONS", "QUALITY_POLICY", "READINESS", "READINESS_SCHEMA", "REPORT", "REPORT_SCHEMA", "V26F_PARENT_DETAIL_CSV", "V26F_CONTRACT", "V26F_READINESS", "V26F_REPORT", "implementation_sha256", "quality_pending_reasons", "read_json", "relative_path", "require_bound_file", "resolve_path", "sha256_file", "source_binding", "validate_prepared_contract", "validate_quality_cache_relationship", "validate_v26f_source_report", "write_immutable_json", "write_immutable_text"]