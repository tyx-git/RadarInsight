from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.core.data.rd_segment_store import build_window_segment_map


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Expert1 window-to-RD-segment map.")
    parser.add_argument("--window-index-name", default="expert1_window_index.csv")
    parser.add_argument("--segment-manifest-name", default="expert1_rd_segment_manifest.csv")
    parser.add_argument("--output-name", default="expert1_window_segment_map.csv")
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument("--allow-overlap", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    summary = build_window_segment_map(
        project_root / "data",
        window_index_name=args.window_index_name,
        segment_manifest_name=args.segment_manifest_name,
        output_name=args.output_name,
        max_windows=args.max_windows,
        strict_contained=not args.allow_overlap,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
