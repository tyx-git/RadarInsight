from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class RunPaths:
    output_dir: Path
    timestamp_label: str
    experiment_name: str


def prepare_run_output_dir(requested_output_dir: str | Path) -> RunPaths:
    requested = Path(requested_output_dir)
    parent = requested.parent
    experiment_name = requested.name
    timestamp_base = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = 0
    while True:
        timestamp_label = timestamp_base if suffix == 0 else f"{timestamp_base}_{suffix:02d}"
        output_dir = parent / timestamp_label / experiment_name
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=False)
            return RunPaths(
                output_dir=output_dir,
                timestamp_label=timestamp_label,
                experiment_name=experiment_name,
            )
        suffix += 1
