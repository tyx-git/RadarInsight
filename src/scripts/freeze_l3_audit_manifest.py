"""Create a provenance-bound manifest for the held-out L3 audit role."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any


SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _atomic_json(path: Path, payload: dict[str, Any], *, overwrite: bool) -> None:
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode(
        "utf-8"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        if path.read_bytes() == encoded:
            return
        raise FileExistsError(f"refusing to overwrite different file: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Freeze the independent L3 audit manifest")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument(
        "--source-manifest-path",
        type=Path,
        default=Path("src/config/manifests.json"),
    )
    parser.add_argument(
        "--calibration-path",
        type=Path,
        default=Path("artifacts/radar_ln3/l3/radar_ln3_calibration.json"),
    )
    parser.add_argument(
        "--output-manifest-path",
        type=Path,
        default=Path("src/config/radar_ln3_audit_manifest.json"),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip-data-validation",
        action="store_true",
        help="复用前置 data_index 校验结果，不重复扫描完整 TR/RD 文件",
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    from radar_ln3.data.manifest import (
        canonical_json_hash,
        data_manifest_hash,
        load_manifest,
        resolve_manifest_view,
    )
    from radar_ln3.data.paths import resolve_project_path
    from radar_ln3.data.contracts import MissingDataFileError
    from radar_ln3.decision import load_l3_calibration

    root = args.project_root.resolve()
    source_path = resolve_project_path(root, args.source_manifest_path, must_exist=True)
    output_path = resolve_project_path(root, args.output_manifest_path, must_exist=False)
    if source_path == output_path:
        raise ValueError("audit manifest must be a different file from source manifest")
    try:
        calibration = load_l3_calibration(root, args.calibration_path)
    except MissingDataFileError as exc:
        calibration_path = resolve_project_path(
            root,
            args.calibration_path,
            must_exist=False,
        )
        raise ValueError(
            "L3 calibration artifact 不存在，不能冻结 audit manifest："
            f" {calibration_path}。请先在当前 L1/L2 checkpoint 下运行：\n"
            "python src/scripts/calibrate_l3.py "
            "--project-root $root "
            "--manifest-path src/config/manifests.json "
            "--feature-cache-manifest-path "
            "data/cq08/cache/features/l3/validation/feature-cache-manifest.json "
            "--device cuda --batch-size 2 --overwrite"
        ) from exc
    source = load_manifest(
        args.source_manifest_path,
        root,
        verify_files=not bool(args.skip_data_validation),
        allow_unmaterialized=False,
    )
    source_hash = data_manifest_hash(source)
    if calibration["source_manifest_sha256"] != source_hash:
        raise ValueError("calibration source hash does not match source manifest")
    if calibration["status"] != "exploratory" or calibration["deployment_eligible"] is not False:
        raise ValueError("audit manifest must be frozen from exploratory calibration")

    view = resolve_manifest_view(
        source,
        split_name="test",
        include_unknown=True,
        training_view=False,
        task_name="l3_audit",
    )
    from radar_ln3.data.canonical_rd import canonical_rd_binding

    binding = canonical_rd_binding(root)
    if binding is not None:
        view = replace(view, data_binding=f"canonical-rd:{binding}")
    parent_ids = [str(value) for value in view.parent_ids]
    calibration_parent_ids = {str(value) for value in calibration["parent_ids"]}
    overlap = sorted(set(parent_ids).intersection(calibration_parent_ids))
    if overlap:
        raise ValueError(f"audit parents overlap calibration parents: {overlap[:3]}")
    audit_parent_set = set(parent_ids)
    label_by_parent: dict[str, str] = {}
    for row in source.tr_records:
        parent_id = str(row["parent_sample_id"])
        if parent_id in audit_parent_set:
            label = str(row["label_name"])
            previous = label_by_parent.setdefault(parent_id, label)
            if previous != label:
                raise ValueError(f"audit parent has conflicting labels: {parent_id}")
    if set(label_by_parent) != audit_parent_set:
        raise ValueError("audit view parent labels are incomplete")
    unknown_support = sum(
        label == "UnknownTarget" for label in label_by_parent.values()
    )
    minimum_unknown_support = int(calibration["policy"]["minimum_audit_unknown_support"])
    if unknown_support < minimum_unknown_support:
        raise ValueError(
            f"audit Unknown parent support {unknown_support} < calibration policy requirement "
            f"{minimum_unknown_support}。请先重新封存 calibration policy：\n"
            "python src/scripts/calibrate_l3.py "
            "--project-root $root "
            "--manifest-path src/config/manifests.json "
            "--feature-cache-manifest-path "
            "data/cq08/cache/features/l3/validation/feature-cache-manifest.json "
            "--device cuda --batch-size 2 "
            "--minimum-audit-unknown-support 16 --overwrite"
        )

    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("source manifest root must be an object")
    audit = dict(payload.get("audit") or {})
    audit["l3_audit"] = {
        "role": "post_selection_audit",
        "source_manifest_sha256": source_hash,
        "calibration_artifact_sha256": calibration["artifact_sha256"],
        "split_name": "test",
        "parent_ids": parent_ids,
        "parent_ids_sha256": canonical_json_hash(sorted(parent_ids)),
        "parent_count": len(parent_ids),
        "unknown_parent_support": unknown_support,
        "threshold_fitting_allowed": False,
        "thresholds_frozen_before_access": True,
    }
    payload["audit"] = audit
    payload.pop("manifest_sha256", None)
    payload["manifest_sha256"] = canonical_json_hash(payload)
    _atomic_json(output_path, payload, overwrite=bool(args.overwrite))
    audit_manifest = load_manifest(
        output_path,
        root,
        verify_files=not bool(args.skip_data_validation),
        allow_unmaterialized=False,
    )
    return {
        "path": str(output_path.relative_to(root).as_posix()),
        "source_data_manifest_sha256": source_hash,
        "audit_data_manifest_sha256": data_manifest_hash(audit_manifest),
        "manifest_sha256": audit_manifest.manifest_sha256,
        "parent_count": len(parent_ids),
        "unknown_parent_support": unknown_support,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        print(json.dumps(run(args), ensure_ascii=False, indent=2), flush=True)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"L3 audit manifest failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
