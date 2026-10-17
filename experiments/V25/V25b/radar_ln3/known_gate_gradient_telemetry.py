"""Audit-only gradient telemetry for the V25a KnownGate replication.

The recorder is deliberately external to the model, loss, optimiser and data
pipeline.  It observes gradients after ``loss.backward()`` and before the
existing global-norm clipping call, then writes one immutable JSON payload per
completed epoch.  Therefore it cannot become a model input or alter an update.
"""

from __future__ import annotations


from torch import nn


V25A_GRADIENT_TELEMETRY_SCHEMA = (
    "RADAR-LN3-V25a-known-gate-preclip-gradient-batch-telemetry-v1"
)
V25A_GRADIENT_TELEMETRY_DIRECTORY = "gradient-telemetry"
V25A_GRADIENT_MODULE_NAMES = (
    "tr_encoder",
    "rd_encoder",
    "fusion",
    "liquid",
    "classifier",
)
V25A_BASE_LABELS = (
    "DroneTarget",
    "BirdTarget",
    "BalloonTarget",
    "ClutterTarget",
)
V25A_TASK_LABELS = (
    "KNOWN_TARGET_EVIDENCE",
    "KNOWN_CLUTTER_EVIDENCE",
)
















def partition_gradient_modules(model: nn.Module) -> dict[str, list[tuple[str, nn.Parameter]]]:
    """Partition every trainable parameter into one product-model module.

    The explicit partition is a contract check: a newly introduced trainable
    parameter must be assigned deliberately instead of being silently omitted
    from the diagnostic report.
    """

    groups = {name: [] for name in V25A_GRADIENT_MODULE_NAMES}
    unknown: list[str] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            raise ValueError(f"V25a requires full parameter training, found frozen {name}")
        matched = [
            module
            for module in V25A_GRADIENT_MODULE_NAMES
            if name.startswith(module + ".")
        ]
        if len(matched) != 1:
            unknown.append(name)
            continue
        groups[matched[0]].append((name, parameter))
    if unknown or any(not values for values in groups.values()):
        raise ValueError(
            "V25a gradient module partition is incomplete: "
            f"unknown={unknown}, empty={[name for name, values in groups.items() if not values]}"
        )
    all_names = [name for values in groups.values() for name, _ in values]
    expected = [name for name, _ in model.named_parameters()]
    if sorted(all_names) != sorted(expected) or len(all_names) != len(set(all_names)):
        raise AssertionError("V25a gradient parameter partition does not cover each parameter once")
    return groups








__all__ = ["V25A_BASE_LABELS", "V25A_GRADIENT_MODULE_NAMES", "V25A_GRADIENT_TELEMETRY_DIRECTORY", "V25A_GRADIENT_TELEMETRY_SCHEMA", "V25A_TASK_LABELS", "partition_gradient_modules"]