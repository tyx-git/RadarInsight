from __future__ import annotations

import argparse
import json
from pathlib import Path

from radar_ln3.config import load_config, resolve_project_path
from radar_ln3.evaluator import evaluate_checkpoint, strip_detailed_records, write_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a RADAR-LN3 formal checkpoint on val or explicitly enabled test split.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    config_path = args.config if args.config.is_absolute() else project_root / args.config
    config = load_config(config_path.resolve())
    checkpoint = args.checkpoint or resolve_project_path(project_root, config.train.checkpoint_path)
    result = evaluate_checkpoint(project_root, config, checkpoint.resolve(), split_name=args.split, batch_size=args.batch_size)
    report_path = write_report(project_root, config, result, args.split)
    output = strip_detailed_records(result)
    output["report_path"] = str(report_path)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
