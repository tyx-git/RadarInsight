"""Shared contracts for the strict TR-only execution path."""

from __future__ import annotations

from collections.abc import Iterable


TR_ONLY_ALLOWED_SOURCES = frozenset({"cq08_track", "raw_tr_rd"})


def validate_tr_only_sources(
    source_names: Iterable[object],
    *,
    context: str,
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
    return normalized


__all__ = ["TR_ONLY_ALLOWED_SOURCES", "validate_tr_only_sources"]
