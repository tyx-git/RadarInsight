"""Final V28 frozen research L3 calibration and replay."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


from .replay import _action_metrics












def _probabilities(logits: np.ndarray, temperature: float) -> np.ndarray:
    shifted = logits / float(temperature); shifted -= shifted.max(axis=1, keepdims=True); exp = np.exp(shifted); return exp / exp.sum(axis=1, keepdims=True)






def _candidate(probabilities: np.ndarray, names: tuple[str, ...], floors: Mapping[str, Any]) -> list[str]:
    return [name for index, name in enumerate(names) if float(probabilities[index]) + 1e-12 >= float(floors.get(name, 0.0))]



















