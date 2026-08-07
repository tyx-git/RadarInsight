from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from radar_ln3.config import load_config
from radar_ln3.factory import build_model
from radar_ln3.losses import causal_prefix_loss


def run(config_path: Path) -> dict[str, object]:
    config = load_config(config_path)
    model = build_model(config).eval()
    batch_size, sequence_length = 2, 4
    events = torch.randn(batch_size, sequence_length, config.model.liquid_hidden_dim)
    dt = torch.tensor([[0.0, 0.0, 2.0, 4.0], [0.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
    dt_valid = torch.tensor([[False, False, True, True], [False, True, False, False]])
    padding = torch.tensor([[False, False, False, True], [False, True, True, True]])
    liquid = model.liquid(events, dt, padding, dt_valid=dt_valid)
    expected_invalid = (~dt_valid) & (~padding)
    if not torch.equal(liquid.dt_invalid_mask, expected_invalid):
        raise AssertionError("liquid dt_invalid_mask does not preserve the explicit validity contract")
    if not bool(liquid.dt_zero_mask[0, 0]) or not bool(liquid.dt_zero_mask[0, 1]):
        raise AssertionError("zero dt events were not exposed")
    logits = torch.randn(batch_size, sequence_length, 5, requires_grad=True)
    labels = torch.tensor([0, 1], dtype=torch.long)
    loss = causal_prefix_loss(logits, labels, padding)
    loss.backward()
    if not torch.isfinite(loss).item() or logits.grad is None or not torch.isfinite(logits.grad).all():
        raise AssertionError("causal prefix loss gradient contract failed")
    if bool((logits.grad[0, 3].abs() > 0).any()) or bool((logits.grad[1, 1:].abs() > 0).any()):
        raise AssertionError("padding logits received causal prefix loss gradient")
    result = {
        "status": "passed",
        "config_schema": config.schema_version,
        "dt_invalid_mask": liquid.dt_invalid_mask.tolist(),
        "dt_zero_mask": liquid.dt_zero_mask.tolist(),
        "causal_prefix_loss": float(loss.item()),
        "padding_gradient_excluded": True,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Check RADAR-LN3 dt validity and causal-prefix loss boundaries.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run(args.config.resolve())


if __name__ == "__main__":
    main()
