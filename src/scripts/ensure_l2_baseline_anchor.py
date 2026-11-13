"""确保当前 L2 train/validation view 拥有完整 baseline anchor。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ensure the current L2 event baseline anchor")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path, default=Path("src/config/manifests.json"))
    parser.add_argument(
        "--feature-cache-manifest-path",
        type=Path,
        default=Path("data/cq08/cache/features/feature-cache-manifest.json"),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/cq08/cache/l2-baseline-anchor.pt"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="忽略已有 anchor，按当前 reference 输入重新生成",
    )
    parser.add_argument(
        "--skip-data-validation",
        action="store_true",
        help="跳过数据文件 size/hash 校验（verify_files=False）；manifest 结构与事件路径边界校验仍会执行",
    )
    parser.add_argument("--skip-event-cache-validation", action="store_true")
    args = parser.parse_args(argv)
    try:
        from radar_ln3.training.l2_anchor import ensure_l2_baseline_anchor

        result = ensure_l2_baseline_anchor(
            args.project_root.resolve(),
            manifest_path=args.manifest_path,
            feature_cache_manifest_path=args.feature_cache_manifest_path,
            output_path=args.output_path,
            device=str(args.device),
            batch_size=int(args.batch_size),
            verify_files=not bool(args.skip_data_validation),
            validate_event_cache=not bool(args.skip_event_cache_validation),
            rebuild=bool(args.rebuild),
        )
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"L2 baseline anchor failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
