"""Materialize the canonical CQ08 RD store under ``data/cq08``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Materialize the standalone radar_ln3 canonical CQ08 RD store")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path, default=Path("src/config/manifests.json"))
    parser.add_argument(
        "--source-frame-index-path",
        type=Path,
        default=Path("data/processed/cq08_rd_physical_canonical_frame_index_v3_1.csv"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/cq08/rd/canonical"),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip-data-validation",
        action="store_true",
        help="跳过数据文件 size/hash 校验（verify_files=False）；manifest 结构与事件路径边界校验仍会执行",
    )
    args = parser.parse_args(argv)
    try:
        from radar_ln3.data import materialize_canonical_rd_store

        result = materialize_canonical_rd_store(
            args.project_root,
            manifest_path=args.manifest_path,
            source_frame_index_path=args.source_frame_index_path,
            output_root=args.output_root,
            overwrite=bool(args.overwrite),
            verify_files=not bool(args.skip_data_validation),
        )
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"canonical RD materialization failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
