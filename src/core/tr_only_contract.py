"""Shared contracts for the strict TR-only execution path."""

from __future__ import annotations

from collections.abc import Iterable


TR_ONLY_ALLOWED_SOURCES = frozenset({"cq08_track", "raw_tr_rd"})


def validate_tr_only_sources(
    source_names: Iterable[object],
    *,
    context: str,
    required_source_names: Iterable[object] | None = None,
) -> set[str]:
    """Validate the source vocabulary before any TR-only data is consumed.

    Source names are metadata used to locate the correct track encoding and
    normalization statistics.  In the strict path, an unknown or empty source
    is a contract violation rather than a row that can be silently ignored.
    """

    normalized = {str(value).strip() for value in source_names}
    invalid = sorted(
        value
        for value in normalized
        if not value or value.lower() in {"nan", "none", "null"}
        or value not in TR_ONLY_ALLOWED_SOURCES
    )
    if invalid:
        raise ValueError(
            f"{context} requires strict TR-only sources {sorted(TR_ONLY_ALLOWED_SOURCES)}; "
            f"unsupported source_name values={invalid}"
        )
    if required_source_names is not None:
        required = {str(value).strip() for value in required_source_names}
        invalid_required = sorted(required - set(TR_ONLY_ALLOWED_SOURCES))
        if not required or invalid_required:
            raise ValueError(
                f"{context} has an invalid required TR-only source set: "
                f"{sorted(required)}; allowed={sorted(TR_ONLY_ALLOWED_SOURCES)}"
            )
        if normalized != required:
            raise ValueError(
                f"{context} source set mismatch: observed={sorted(normalized)}, "
                f"required={sorted(required)}"
            )
    return normalized


__all__ = ["TR_ONLY_ALLOWED_SOURCES", "validate_tr_only_sources"]
