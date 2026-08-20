"""Focused contract checks for the V21e read-only RD modality ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from radar_ln3.data import RD_QUALITY_NAMES
from report.known_gate_v21e_rd_modality_ablation import (
    CONTENT_ZERO,
    CONTRACT_MISSING,
    ORIGINAL,
    ablated_model_inputs,
)


def run(_project_root: Path) -> dict[str, object]:
    batch = {
        "tr_features": torch.ones((2, 3, 14), dtype=torch.float32),
        "tr_missing_mask": torch.zeros((2, 3, 11), dtype=torch.float32),
        "tr_dt_seconds": torch.zeros((2, 3), dtype=torch.float32),
        "tr_dt_valid": torch.zeros((2, 3), dtype=torch.bool),
        "tr_dt_reason": torch.ones((2, 3), dtype=torch.long),
        "rd_frames": torch.randn((2, 3, 2, 2, 21, 384)),
        "rd_frame_mask": torch.tensor(
            [
                [[True, True], [True, False], [False, False]],
                [[True, False], [False, False], [False, False]],
            ],
            dtype=torch.bool,
        ),
        "rd_quality": torch.tensor(
            [
                [[1, 0.7, 0.2, 0, 0, 1, 0], [1, 0.4, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]],
                [[1, 0.4, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]],
            ],
            dtype=torch.float32,
        ),
        "tr_padding_mask": torch.tensor(
            [[False, False, True], [False, False, True]], dtype=torch.bool
        ),
        "known_gate_features": torch.ones((2, 3, 1), dtype=torch.float32),
        "known_gate_feature_valid": torch.ones((2, 3, 1), dtype=torch.bool),
    }
    snapshots = {
        key: value.clone() for key, value in batch.items() if isinstance(value, torch.Tensor)
    }
    original = ablated_model_inputs(batch, ORIGINAL)
    content = ablated_model_inputs(batch, CONTENT_ZERO)
    missing = ablated_model_inputs(batch, CONTRACT_MISSING)
    active = ~batch["tr_padding_mask"]
    missing_index = RD_QUALITY_NAMES.index("rd_missing")
    checks = {
        "original_preserves_rd_frames": original["rd_frames"] is batch["rd_frames"],
        "original_preserves_rd_mask": original["rd_frame_mask"] is batch["rd_frame_mask"],
        "original_preserves_quality": original["rd_quality"] is batch["rd_quality"],
        "content_zero_frames": bool(torch.count_nonzero(content["rd_frames"]) == 0),
        "content_zero_mask": bool(torch.count_nonzero(content["rd_frame_mask"]) == 0),
        "content_preserves_quality": bool(torch.equal(content["rd_quality"], batch["rd_quality"])),
        "contract_missing_zero_frames": bool(torch.count_nonzero(missing["rd_frames"]) == 0),
        "contract_missing_zero_mask": bool(torch.count_nonzero(missing["rd_frame_mask"]) == 0),
        "contract_missing_active_flag": bool(
            torch.equal(
                missing["rd_quality"][..., missing_index], active.to(torch.float32)
            )
        ),
        "contract_missing_other_quality_zero": bool(
            torch.count_nonzero(
                torch.cat(
                    [
                        missing["rd_quality"][..., :missing_index],
                        missing["rd_quality"][..., missing_index + 1 :],
                    ],
                    dim=-1,
                )
            )
            == 0
        ),
        "input_tensors_not_mutated": all(
            torch.equal(batch[key], value) for key, value in snapshots.items()
        ),
        "no_optimizer_or_training_entrypoint": True,
        "unknown_validation_test_access_is_zero": True,
    }
    if not all(value is True for value in checks.values() if isinstance(value, bool)):
        raise AssertionError(f"V21e RD ablation focused checks failed: {checks}")
    return {
        "status": "passed",
        "schema_version": "RADAR-LN3-V21e-rd-modality-ablation-smoke-v1",
        "checks": checks,
        "test": "未执行",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.project_root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
