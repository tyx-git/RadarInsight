from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.core.data.rd_pairing_contract import (
    RDPairingPolicy,
    evaluate_rd_tr_pairing,
    pairing_contract_payload,
)


def _as_bool(value: object) -> bool:
    return str(value if value is not None else "").strip().lower() in {"1", "true", "yes"}


def _quality_flags(value: object) -> list[str]:
    text = str(value if value is not None else "").strip()
    if not text or text.lower() == "nan":
        return []
    return [item.strip() for item in text.replace("[", "").replace("]", "").replace('"', "").split(";") if item.strip()]


def audit_expert1_rd_pairing(
    project_root: Path,
    *,
    manifest_name: str = "expert1_rd_segment_manifest.csv",
    output_name: str = "expert1_rd_pairing_audit.json",
    records_name: str = "expert1_rd_pairing_audit_records.csv",
    min_window_coverage_ratio: float = 0.80,
    min_continuous_rd_seconds: float = 1.00,
    max_frame_gap_seconds: float = 0.10,
) -> dict[str, object]:
    processed_root = project_root / "data" / "processed"
    manifest_path = processed_root / manifest_name
    manifest = pd.read_csv(manifest_path, low_memory=False)
    policy = RDPairingPolicy(
        min_window_coverage_ratio=min_window_coverage_ratio,
        min_continuous_rd_seconds=min_continuous_rd_seconds,
        max_frame_gap_seconds=max_frame_gap_seconds,
    )
    records: list[dict[str, object]] = []
    decision_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    for row in manifest.to_dict(orient="records"):
        parent_id = str(row.get("parent_sample_id", "") or "")
        sync_status = str(row.get("rd_time_sync_status", "") or "")
        decision = evaluate_rd_tr_pairing(
            {
                **row,
                "target_association_verified": bool(parent_id and sync_status in {"exact_time_sync", "within_tolerance"}),
                "rd_target_association_id": parent_id,
                "tr_target_association_id": parent_id,
            },
            policy,
        )
        quality_flags = _quality_flags(row.get("rd_quality_flags"))
        output = {
            "segment_id": str(row.get("segment_id", "")),
            "parent_sample_id": parent_id,
            "source_name": str(row.get("source_name", "")),
            "label_name": str(row.get("label_name", "")),
            "rd_time_sync_status": sync_status,
            "segment_duration_seconds": row.get("segment_duration_seconds"),
            "rd_time_coverage_ratio": row.get("rd_time_coverage_ratio"),
            "rd_quality_flags": ";".join(quality_flags),
            "pairing_decision": decision.decision,
            "rd_fusion_eligible_contract": decision.fusion_eligible,
            "pairing_reasons": ";".join(decision.reasons),
        }
        records.append(output)
        decision_counts[decision.decision] += 1
        reason_counts.update(decision.reasons)
    records_frame = pd.DataFrame(records)
    output_path = project_root / output_name
    records_path = project_root / records_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records_path.parent.mkdir(parents=True, exist_ok=True)
    records_frame.to_csv(records_path, index=False, encoding="utf-8")
    by_source_label = (
        records_frame.groupby(["source_name", "label_name", "pairing_decision"], dropna=False)
        .size()
        .sort_index()
        .to_dict()
    )
    parent_counts = (
        records_frame.groupby("pairing_decision", dropna=False)["parent_sample_id"]
        .nunique()
        .to_dict()
    )
    payload = {
        "audit_scope": "manifest-only; no model training, selection, calibration, or test evaluation",
        "contract": pairing_contract_payload(policy),
        "manifest_name": manifest_name,
        "segment_count": int(len(records_frame)),
        "decision_counts": dict(decision_counts),
        "decision_parent_counts": {str(key): int(value) for key, value in parent_counts.items()},
        "reason_counts": dict(reason_counts),
        "source_label_decision_counts": {"|".join(map(str, key)): int(value) for key, value in by_source_label.items()},
        "records_path": records_path.relative_to(project_root).as_posix(),
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Audit Expert1 RD/TR pairing eligibility without rebuilding data.")
    parser.add_argument("--manifest-name", default="expert1_rd_segment_manifest.csv")
    parser.add_argument("--output", default="tmp/expert1_rd_pairing_audit.json")
    parser.add_argument("--records-output", default="tmp/expert1_rd_pairing_audit_records.csv")
    parser.add_argument("--min-window-coverage", type=float, default=0.80)
    parser.add_argument("--min-continuous-rd-seconds", type=float, default=1.00)
    parser.add_argument("--max-frame-gap-seconds", type=float, default=0.10)
    args = parser.parse_args(list(argv) if argv is not None else None)
    project_root = Path(__file__).resolve().parents[2]
    payload = audit_expert1_rd_pairing(
        project_root,
        manifest_name=args.manifest_name,
        output_name=args.output,
        records_name=args.records_output,
        min_window_coverage_ratio=args.min_window_coverage,
        min_continuous_rd_seconds=args.min_continuous_rd_seconds,
        max_frame_gap_seconds=args.max_frame_gap_seconds,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
