"""Evaluate a set of labeled `.asd` packages with the product cascade."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


SRC_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_ROOT.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate AeroSight .asd packages")
    parser.add_argument("asd_paths", nargs="+", type=Path)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--timestamp-unit", default=None)
    parser.add_argument("--amplitude-mode", default=None, choices=("magnitude_db", "power_db", "db", "raw"))
    parser.add_argument("--pytorch", action="store_true", help="使用 PyTorch pipeline 而不是 full-cascade ONNX")
    parser.add_argument("--directml", action="store_true", help="ONNX 优先使用 DirectML")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        from core.evaluation import evaluate_asd_packages, write_evaluation_report

        report = evaluate_asd_packages(
            args.project_root,
            args.asd_paths,
            timestamp_unit=args.timestamp_unit,
            amplitude_mode=args.amplitude_mode,
            onnx_path=None if args.pytorch else Path("artifacts/radar_ln3/onnx/radarln3.onnx"),
            prefer_directml=bool(args.directml),
        )
        if args.output is not None:
            write_evaluation_report(args.output, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ASD evaluation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
