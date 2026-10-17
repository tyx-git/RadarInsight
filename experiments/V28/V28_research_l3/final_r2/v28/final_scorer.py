"""Research-only L2 class-conditional nonconformity scorer.

The scorer is deliberately downstream of the frozen classifier and calibration
parameters.  It does not feed classifier probabilities or class identities into
either knownness head.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from radar_ln3.runtime import canonical_sha256


SCORER_METHOD = "class_conditional_probability_nonconformity"
SCORER_RULE = "score <= reference_quantile + tolerance"
SCORER_TOLERANCE = 1.0e-12






def validate_scorer_descriptor(scorer: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one serialized scorer and return its compact binding summary."""

    if scorer.get("status") != "implemented_research_only" or scorer.get("research_only") is not True:
        raise ValueError("L2 scorer is not a research-only implemented scorer")
    if scorer.get("method") != SCORER_METHOD or scorer.get("candidate_rule") != SCORER_RULE:
        raise ValueError("L2 scorer method or candidate rule differs")
    body = {key: value for key, value in scorer.items() if key != "scorer_sha256"}
    if scorer.get("scorer_sha256") != canonical_sha256(body):
        raise ValueError("L2 scorer descriptor hash differs")
    names = tuple(str(value) for value in scorer.get("class_names", ()))
    if not names or len(set(names)) != len(names):
        raise ValueError("L2 scorer class names must be unique and non-empty")
    references = scorer.get("reference_quantile_by_class")
    supports = scorer.get("support_by_class")
    if not isinstance(references, Mapping) or set(references) != set(names):
        raise ValueError("L2 scorer descriptor quantiles do not match class names")
    if not isinstance(supports, Mapping) or set(supports) != set(names):
        raise ValueError("L2 scorer descriptor support does not match class names")
    quantiles = np.asarray([float(references[name]) for name in names], dtype=np.float64)
    support_values = np.asarray([int(supports[name]) for name in names], dtype=np.int64)
    if not np.isfinite(quantiles).all() or bool((quantiles < 0.0).any()) or bool((quantiles > 1.0).any()) or bool((support_values <= 0).any()):
        raise ValueError("L2 scorer descriptor references are invalid")
    tolerance = float(scorer.get("tolerance", SCORER_TOLERANCE))
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("L2 scorer tolerance is invalid")
    return {"class_names": list(names), "reference_quantile_by_class": {name: float(value) for name, value in zip(names, quantiles, strict=True)}, "tolerance": tolerance, "scorer_sha256": str(scorer["scorer_sha256"])}


def _validate_descriptor(scorer: Mapping[str, Any]) -> tuple[tuple[str, ...], np.ndarray, float]:
    summary = validate_scorer_descriptor(scorer)
    names = tuple(summary["class_names"])
    quantiles = np.asarray([summary["reference_quantile_by_class"][name] for name in names], dtype=np.float64)
    return names, quantiles, float(summary["tolerance"])


def score_class_conditional(probabilities: np.ndarray, scorer: Mapping[str, Any]) -> dict[str, Any]:
    """Score one or more probability rows and return explicit candidate sets."""

    names, quantiles, tolerance = _validate_descriptor(scorer)
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] != len(names) or not np.isfinite(values).all():
        raise ValueError("L2 scorer probabilities have an invalid shape or contain non-finite values")
    if bool((values < -SCORER_TOLERANCE).any()) or bool((values > 1.0 + SCORER_TOLERANCE).any()):
        raise ValueError("L2 scorer probabilities are outside [0, 1]")
    if not np.allclose(values.sum(axis=1), 1.0, atol=1.0e-6, rtol=0.0):
        raise ValueError("L2 scorer probabilities must sum to one")
    values = np.clip(values, 0.0, 1.0)
    scores = 1.0 - values
    passes = scores <= quantiles[None, :] + tolerance
    candidate_sets = [[names[index] for index in np.flatnonzero(row)] for row in passes]
    predictions = [names[int(index)] for index in values.argmax(axis=1)]
    result = {
        "class_names": list(names),
        "probability_by_class": values,
        "score_by_class": scores,
        "reference_quantile_by_class": np.broadcast_to(quantiles, scores.shape).copy(),
        "candidate_pass_by_class": passes,
        "candidate_sets": candidate_sets,
        "predictions": predictions,
    }
    return result




__all__ = ["SCORER_METHOD", "SCORER_RULE", "validate_scorer_descriptor", "score_class_conditional"]