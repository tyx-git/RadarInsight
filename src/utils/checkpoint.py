from pathlib import Path
from typing import Any

import torch


def load_checkpoint(checkpoint_path: str | Path, map_location: Any) -> dict[str, Any]:
    path = Path(checkpoint_path)
    try:
        payload = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location=map_location)
    return dict(payload)
