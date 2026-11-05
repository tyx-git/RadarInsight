"""Run one `.asd` package through the product full cascade."""

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
    parser = argparse.ArgumentParser(description="Infer one AeroSight .asd package")
    parser.add_argument("asd_path", type=Path)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--onnx",
        type=Path,
        default=Path("artifacts/radar_ln3/onnx/radarln3.onnx"),
    )
    parser.add_argument("--timestamp-unit", default=None)
    parser.add_argument("--amplitude-mode", default=None, choices=("magnitude_db", "power_db", "db", "raw"))
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args(argv)
    try:
        from core.inference import ONNXProductInferencePipeline, ProductInferencePipeline

        onnx_path = args.onnx if args.onnx.is_absolute() else args.project_root / args.onnx
        if args.cpu or onnx_path.exists():
            pipeline = ONNXProductInferencePipeline(
                args.project_root,
                onnx_path,
                prefer_directml=not bool(args.cpu),
            )
        else:
            pipeline = ProductInferencePipeline(args.project_root)
        result = pipeline.predict_asd(
            args.asd_path,
            timestamp_unit=args.timestamp_unit,
            amplitude_mode=args.amplitude_mode,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ASD inference failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
