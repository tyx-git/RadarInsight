"""Build the raw-backed feature cache used by one L3 evidence/head role.

The normal training cache intentionally excludes Unknown and test parents.
The L3 head-fit, validation, and post-selection audit roles use explicit views
so their support and provenance can be checked independently.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _progress(event: dict[str, Any]) -> None:
    stage = str(event.get("stage", ""))
    if stage in {
        "validation_start",
        "validation_complete",
        "feature_cache_start",
        "feature_cache_progress",
        "feature_cache_complete",
        "feature_cache_split_complete",
    }:
        print(json.dumps(event, ensure_ascii=False, sort_keys=True), flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build an L3 validation or audit feature cache")
    parser.add_argument(
        "--project-root",
        type=Path,
        required=True,
        help="RadarInsight project root containing src/ and data/",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("src/config/manifests.json"),
    )
    parser.add_argument(
        "--role",
        choices=("head-fit", "validation", "audit"),
        required=True,
        help="head-fit uses train parents including Unknown; validation uses validation; audit uses test",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Output directory under data/; defaults to data/cq08/cache/features/l3/<role>",
    )
    parser.add_argument("--progress-interval", type=int, default=25)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip-data-validation",
        action="store_true",
        help="复用前置 data_index 校验结果，不重复扫描完整 TR/RD 文件",
    )
    return parser


def build(args: argparse.Namespace) -> dict[str, Any]:
    from radar_ln3.data.cq08 import CQ08Adapter, CQ08RawEchoProvider
    from radar_ln3.model import load_model_config
    from core.training.cache import build_feature_training_cache

    root = args.project_root.resolve()
    manifest_path = args.manifest_path
    role = str(args.role)
    split_name = "train" if role == "head-fit" else ("validation" if role == "validation" else "test")
    training_view = role != "audit"
    task_name = "l3_head_fit" if role == "head-fit" else f"l3_{role}"
    output_root = args.output_root or Path(
        f"data/cq08/cache/features/l3/{'head_fit' if role == 'head-fit' else role}"
    )
    if isinstance(args.progress_interval, bool) or int(args.progress_interval) <= 0:
        raise ValueError("--progress-interval must be a positive integer")

    config = load_model_config(root)
    adapter = CQ08Adapter(
        root,
        registered_index=True,
        manifest_path=manifest_path,
        verify_files=not bool(args.skip_data_validation),
    )
    provider = CQ08RawEchoProvider(
        root,
        source_root=adapter.source_root,
        cache_size=2,
    )
    result = build_feature_training_cache(
        root,
        manifest_path=manifest_path,
        output_root=output_root,
        split_name=split_name,
        include_unknown=True,
        training_view=training_view,
        task_name=task_name,
        allow_unknown_head_training=(role == "head-fit"),
        raw_echo_provider=provider,
        stable_median=config.data.l2.stable8_median,
        stable_scale=config.data.l2.stable8_scale,
        feature_config={
            "purpose": "l3_evidence",
            "role": role,
            "split_name": split_name,
            "include_unknown": True,
            "training_view": training_view,
            "task_name": task_name,
            "allow_unknown_head_training": role == "head-fit",
        },
        require_stable_evidence=True,
        overwrite=bool(args.overwrite),
        register=False,
        verify_files=not bool(args.skip_data_validation),
        validate_event_cache=not bool(args.skip_data_validation),
        progress_callback=_progress,
        progress_interval=int(args.progress_interval),
    )
    return {
        "role": role,
        "split_name": split_name,
        "training_view": training_view,
        "feature_cache_manifest": result["feature_cache_manifest"],
        "view_name": result["view_name"],
        "view_hash": result["view_hash"],
        "record_count": result["record_count"],
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        print(json.dumps(build(args), ensure_ascii=False, indent=2), flush=True)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"L3 feature cache failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
