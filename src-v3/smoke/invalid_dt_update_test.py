from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from radar_ln3.config import load_config
from radar_ln3.model import CfCLiquidCausalEncoder


def run(config_path: Path) -> dict[str, object]:
    config = load_config(config_path)
    if config.schema_version != "RADAR-LN3-V4":
        raise AssertionError("invalid dt update smoke requires a RADAR-LN3-V4 config")
    if config.model.liquid_invalid_dt_update_mode != "bounded_observation":
        raise AssertionError("V4 smoke requires bounded_observation mode")
    gain = config.model.liquid_invalid_dt_observation_gain
    encoder = CfCLiquidCausalEncoder(
        input_dim=4,
        hidden_dim=6,
        tau_min_seconds=config.model.liquid_tau_min_seconds,
        tau_max_seconds=config.model.liquid_tau_max_seconds,
        zero_dt_event_floor_seconds=config.model.liquid_zero_dt_event_floor_seconds,
        dt_max_seconds=config.model.liquid_dt_max_seconds,
        invalid_dt_update_mode=config.model.liquid_invalid_dt_update_mode,
        invalid_dt_observation_gain=gain,
    )
    events = torch.tensor(
        [[[0.5, -0.2, 0.1, 0.7], [0.1, 0.4, -0.3, 0.2], [0.2, 0.1, 0.6, -0.4]]],
        dtype=torch.float32,
    )
    dt = torch.tensor([[0.0, 2.0, 0.0]], dtype=torch.float32)
    dt_valid = torch.tensor([[False, True, False]], dtype=torch.bool)
    padding = torch.zeros((1, 3), dtype=torch.bool)
    state = encoder(events, dt, padding, dt_valid=dt_valid)
    if not torch.isfinite(state.sequence).all():
        raise AssertionError("V4 invalid-dt update produced non-finite sequence")
    expected_invalid = ~dt_valid & ~padding
    if not torch.equal(state.dt_invalid_mask, expected_invalid):
        raise AssertionError("V4 invalid-dt mask does not preserve the explicit validity contract")
    first_update = state.sequence[:, 0]
    if torch.allclose(first_update, torch.zeros_like(first_update), atol=1.0e-7, rtol=1.0e-7):
        raise AssertionError("bounded observation invalid-dt event did not update the state")
    if not 0.0 < gain <= 1.0:
        raise AssertionError("invalid-dt observation gain is outside the bounded interval")

    logits = state.sequence.sum(dim=-1, keepdim=True).repeat(1, 1, 5)
    loss = logits.square().mean()
    loss.backward()
    if not torch.isfinite(loss).item():
        raise AssertionError("V4 invalid-dt smoke loss is non-finite")
    gradients = [parameter.grad for parameter in encoder.parameters() if parameter.grad is not None]
    if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
        raise AssertionError("V4 invalid-dt smoke gradients are missing or non-finite")
    result = {
        "status": "passed",
        "config_schema": config.schema_version,
        "mode": config.model.liquid_invalid_dt_update_mode,
        "observation_gain": gain,
        "dt_invalid_mask": state.dt_invalid_mask.tolist(),
        "dt_zero_mask": state.dt_zero_mask.tolist(),
        "state_finite": True,
        "gradient_finite": True,
        "formal_training": "未执行",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the bounded invalid-dt liquid update contract.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run(args.config.resolve())


if __name__ == "__main__":
    main()
