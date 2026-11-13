"""L2 类条件非一致性 scorer。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from radar_ln3.runtime import canonical_sha256


SCORER_METHOD = "class_conditional_probability_nonconformity"
SCORER_RULE = "score <= reference_quantile + tolerance"
SCORER_TOLERANCE = 1.0e-12


def build_scorer_descriptor(
    *,
    class_names: Sequence[str],
    conformal: Mapping[str, Any],
    temperature: Mapping[str, Any],
) -> dict[str, Any]:
    names = tuple(str(value) for value in class_names)
    quantiles = conformal.get("quantile_by_class")
    supports = conformal.get("support_by_class")
    if (
        not names
        or len(names) != len(set(names))
        or not isinstance(quantiles, Mapping)
        or not isinstance(supports, Mapping)
    ):
        raise ValueError("scorer binding 无效")
    if set(quantiles) != set(names) or set(supports) != set(names):
        raise ValueError("scorer 类别集合不一致")
    values = np.asarray([float(quantiles[name]) for name in names], dtype=np.float64)
    support_values = np.asarray([int(supports[name]) for name in names], dtype=np.int64)
    if (
        not np.isfinite(values).all()
        or bool((values < 0.0).any())
        or bool((values > 1.0).any())
        or bool((support_values <= 0).any())
    ):
        raise ValueError("scorer quantile/support 无效")
    if temperature.get("status") != "fitted" or int(temperature.get("support", 0)) <= 0:
        raise ValueError("scorer 需要已拟合 temperature")
    body = {
        "status": "implemented_research_only",
        "research_only": True,
        "method": SCORER_METHOD,
        "score": "1 - calibrated_probability",
        "class_names": list(names),
        "reference_quantile_by_class": {
            name: float(value) for name, value in zip(names, values, strict=True)
        },
        "support_by_class": {
            name: int(value) for name, value in zip(names, support_values, strict=True)
        },
        "reference_support": int(support_values.sum()),
        "candidate_rule": SCORER_RULE,
        "tolerance": SCORER_TOLERANCE,
        "temperature": float(temperature["temperature"]),
        "temperature_support": int(temperature["support"]),
        "source_role": "calibration_known_only",
    }
    return {**body, "scorer_sha256": canonical_sha256(body)}


def validate_scorer_descriptor(scorer: Mapping[str, Any]) -> dict[str, Any]:
    body = {key: value for key, value in scorer.items() if key != "scorer_sha256"}
    if (
        scorer.get("status") != "implemented_research_only"
        or scorer.get("research_only") is not True
        or scorer.get("method") != SCORER_METHOD
        or scorer.get("candidate_rule") != SCORER_RULE
        or scorer.get("scorer_sha256") != canonical_sha256(body)
    ):
        raise ValueError("scorer descriptor hash/contract 无效")
    names = tuple(str(value) for value in scorer.get("class_names", ()))
    quantiles = scorer.get("reference_quantile_by_class")
    supports = scorer.get("support_by_class")
    if (
        not names
        or not isinstance(quantiles, Mapping)
        or not isinstance(supports, Mapping)
        or set(quantiles) != set(names)
        or set(supports) != set(names)
    ):
        raise ValueError("scorer descriptor 类别集合无效")
    return {
        "class_names": list(names),
        "reference_quantile_by_class": {
            name: float(quantiles[name]) for name in names
        },
        "tolerance": float(scorer.get("tolerance", SCORER_TOLERANCE)),
        "scorer_sha256": str(scorer["scorer_sha256"]),
    }


__all__ = [
    "build_scorer_descriptor",
    "validate_scorer_descriptor",
]
