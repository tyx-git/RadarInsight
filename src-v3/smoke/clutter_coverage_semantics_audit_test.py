from __future__ import annotations

import argparse
import csv
import json
import tempfile
from pathlib import Path
from typing import Any

from report.clutter_coverage_semantics_audit import (
    AUDIT_CLUSTER_PROFILES,
    BURST_REGISTRY,
    CAPTURE_REGISTRY,
    PARENT_REGISTRY,
    REPORT,
    SEMANTIC_ADJUDICATION_TEMPLATE,
    build,
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def run(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    with tempfile.TemporaryDirectory(prefix="clutter-audit-smoke-") as temporary:
        output = Path(temporary).resolve()
        first = build(root, output_root=output)
        first_payloads = {
            relative.as_posix(): (output / relative).read_bytes()
            for relative in (
                PARENT_REGISTRY,
                CAPTURE_REGISTRY,
                BURST_REGISTRY,
                AUDIT_CLUSTER_PROFILES,
                SEMANTIC_ADJUDICATION_TEMPLATE,
                REPORT,
            )
        }
        second = build(root, output_root=output)
        second_payloads = {
            relative.as_posix(): (output / relative).read_bytes()
            for relative in (
                PARENT_REGISTRY,
                CAPTURE_REGISTRY,
                BURST_REGISTRY,
                AUDIT_CLUSTER_PROFILES,
                SEMANTIC_ADJUDICATION_TEMPLATE,
                REPORT,
            )
        }
        if first_payloads != second_payloads:
            raise AssertionError("Clutter coverage audit is not byte deterministic")

        parents = _read_csv(output / PARENT_REGISTRY)
        captures = _read_csv(output / CAPTURE_REGISTRY)
        bursts = _read_csv(output / BURST_REGISTRY)
        clusters = _read_csv(output / AUDIT_CLUSTER_PROFILES)
        template = _read_csv(output / SEMANTIC_ADJUDICATION_TEMPLATE)
        report = json.loads((output / REPORT).read_text(encoding="utf-8"))

        split_counts = {
            split: sum(row["split_name"] == split for row in parents)
            for split in ("train", "val", "test")
        }
        test_rows = [row for row in parents if row["split_name"] == "test"]
        train_rows = [row for row in parents if row["split_name"] == "train"]
        val_rows = [row for row in parents if row["split_name"] == "val"]
        if (
            len(parents) != 50
            or split_counts != {"train": 33, "val": 9, "test": 8}
            or len(captures) != 50
            or len({row["capture_id"] for row in captures}) != 50
            or len(bursts) == 0
            or len(clusters) != 4
            or len(template) != 42
        ):
            raise AssertionError("Clutter audit output coverage is incomplete")
        if any(row["split_name"] == "test" for row in template):
            raise AssertionError("sealed test parent entered semantic adjudication template")
        if any(
            row["audit_cluster_id"] != "SEALED_UNASSIGNED"
            or row["payload_audit_scope"] != "metadata_only_sealed"
            or any(row[field] for field in ("range_median_m", "radial_speed_std_mps"))
            for row in test_rows
        ):
            raise AssertionError("sealed test parent received a physical descriptor or cluster")
        if any(
            row["audit_cluster_assignment_scope"] != "train_fit_member"
            for row in train_rows
        ) or any(
            row["audit_cluster_assignment_scope"]
            != "validation_nearest_train_cluster"
            for row in val_rows
        ):
            raise AssertionError("Clutter audit cluster population scope is invalid")
        if any(
            row["semantic_subtype"] != "UNADJUDICATED"
            or row["current_label_name"] != "ClutterTarget"
            or row["source_numeric_label_id"] != "5"
            for row in parents
        ):
            raise AssertionError("Clutter audit assigned a semantic subtype or changed a label")
        if any(row["current_label_name"] != "ClutterTarget" for row in template):
            raise AssertionError("Unknown or another class entered Clutter adjudication")
        access = report["access_and_leakage"]
        if (
            access["test_parent_count_registered_without_physical_parse"] != 8
            or access["test_tr_feature_json_parse_count"] != 0
            or access["test_raw_file_open_count"] != 0
            or access["test_mat_file_open_count"] != 0
            or access["test_rd_tensor_open_count"] != 0
            or report["inventory"]["new_legal_clutter_parent_recovered_count"] != 0
            or report["semantic_adjudication"]["automatic_label_changes"] != 0
        ):
            raise AssertionError("Clutter audit leakage or automatic relabel invariant failed")
        if first["checks"] != second["checks"] or not all(first["checks"].values()):
            raise AssertionError("Clutter audit checks are not stable and passing")

        result = {
            "status": "passed",
            "schema_version": "RADAR-LN3-V21-clutter-coverage-semantics-audit-smoke-v1",
            "parent_count": len(parents),
            "split_parent_counts": split_counts,
            "capture_count": len(captures),
            "audit_cluster_profile_count": len(clusters),
            "semantic_template_parent_count": len(template),
            "test_metadata_only_parent_count": len(test_rows),
            "byte_deterministic": True,
            "automatic_label_changes": 0,
            "unknown_in_cluster_or_template": False,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Focused contract test for the V21 Clutter coverage/semantics audit."
    )
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    args = parser.parse_args()
    run(args.project_root.resolve())


if __name__ == "__main__":
    main()
