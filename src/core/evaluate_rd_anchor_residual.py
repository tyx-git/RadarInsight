from __future__ import annotations

import argparse
import json
from pathlib import Path


RETIRED_STATUS = "retired_incompatible_with_physical_rd_v2"


def evaluate_rd_anchor_residual_checkpoint(
    checkpoint_path: Path,
    output_path: Path,
    *,
    trajectory_report: Path,
    device_name: str,
) -> dict[str, object]:
    raise RuntimeError(
        "anchor-residual evaluation is retired: the legacy baseline logit cache is incompatible with the active Physical RD contract"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation-only evaluator for frozen-anchor RD novel residual checkpoints.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--trajectory-report",
        default="artifacts/20260722-222612/20260722-stage009-v7-trajectory-only-baseline-v1/evaluation_report.json",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    try:
        evaluate_rd_anchor_residual_checkpoint(
            Path(args.checkpoint),
            Path(args.output),
            trajectory_report=Path(args.trajectory_report),
            device_name=args.device,
        )
    except RuntimeError as error:
        print(json.dumps({"status": RETIRED_STATUS, "error": str(error)}, ensure_ascii=False, indent=2))
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
