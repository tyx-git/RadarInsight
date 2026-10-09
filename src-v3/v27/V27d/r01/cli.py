"""r01 单一运行入口。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .benchmark import run_benchmark
from .diagnostics import diagnose_excel_from_project, diagnose_x5a, write_json
from .evaluation import evaluate_excel
from .protocol import artifact_path, load_config
from .smoke import run_smoke


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run independent V27d-r01 pointwise denoising experiment")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("smoke")
    x5a = commands.add_parser("diagnose-x5a")
    x5a.add_argument("--limit-files", type=int, default=None)
    x5a.add_argument("--output", type=Path, default=None)
    excel = commands.add_parser("diagnose-excel")
    excel.add_argument("--split", choices=("train", "validation"), default="train")
    excel.add_argument("--output", type=Path, default=None)
    evaluate = commands.add_parser("evaluate-excel")
    evaluate.add_argument("--arm", choices=("zero_control", "radial_decomposed"), default="zero_control")
    evaluate.add_argument("--seed", type=int, default=42)
    evaluate.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    evaluate.add_argument("--output", type=Path, default=None)
    benchmark = commands.add_parser("benchmark")
    benchmark.add_argument("--warmup", type=int, default=2)
    benchmark.add_argument("--repeats", type=int, default=7)
    benchmark.add_argument("--max-records", type=int, default=2000)
    benchmark.add_argument("--excel-points", type=int, default=64)
    benchmark.add_argument("--scaling-steps", type=int, nargs="*", default=None)
    benchmark.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    root = args.project_root.resolve()
    config = load_config(root, args.config)
    if args.command == "smoke":
        value = run_smoke(root)
    elif args.command == "diagnose-x5a":
        value = diagnose_x5a(root, limit_files=args.limit_files)
        if args.output is not None:
            output = args.output if args.output.is_absolute() else artifact_path(root, args.output)
            write_json(output, value)
            value = {**value, "report_path": str(output)}
    elif args.command == "diagnose-excel":
        value = diagnose_excel_from_project(root, split_name=args.split)
        if args.output is not None:
            output = args.output if args.output.is_absolute() else artifact_path(root, args.output)
            write_json(output, value)
            value = {**value, "report_path": str(output)}
    elif args.command == "benchmark":
        value = run_benchmark(
            root,
            warmup=int(args.warmup),
            repeats=int(args.repeats),
            max_records=int(args.max_records),
            excel_points=int(args.excel_points),
            scaling_steps=tuple(args.scaling_steps) if args.scaling_steps else (250, 500, 1000, 2000),
        )
        if args.output is not None:
            output = args.output if args.output.is_absolute() else artifact_path(root, args.output)
            write_json(output, value)
            value = {**value, "report_path": str(output)}
    else:
        value = evaluate_excel(root, arm=args.arm, seed=int(args.seed), device=args.device)
        if args.output is not None:
            output = args.output if args.output.is_absolute() else artifact_path(root, args.output)
            write_json(output, value)
            value = {**value, "report_path": str(output)}
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
