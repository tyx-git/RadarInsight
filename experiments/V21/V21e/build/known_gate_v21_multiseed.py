from __future__ import annotations

from pathlib import Path





def config_relative(seed: int) -> Path:
    return Path(
        "config/RADAR-LN3/"
        f"RADAR-LN3-V21-cq08-liquid-known-gate-baseline-seed{int(seed)}-config.json"
    )


def artifact_relative(seed: int) -> Path:
    return Path(
        "artifacts/"
        f"RADAR-LN3-V21-cq08-liquid-known-gate-baseline-seed{int(seed)}-formal"
    )















