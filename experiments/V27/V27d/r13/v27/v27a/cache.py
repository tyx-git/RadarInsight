"""Materialized V27a frozen-logit evidence cache and loader."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping

import numpy as np
import torch






def _digest_cache_chunk(digest: Any, value: bytes | memoryview) -> None:
    """Feed one length-delimited cache-hash component."""

    digest.update(len(value).to_bytes(8, byteorder="big", signed=False))
    digest.update(value)


def cache_payload_sha256(payload: Mapping[str, Any]) -> str:
    """Hash every semantic cache field independently of Torch serialization.

    A Torch archive's byte layout is not stable across equivalent saves.  This
    digest instead preserves mapping keys, scalar types, array dtype/shape and
    exact C-order bytes, so cache loading fails closed on partial corruption or
    a changed payload under the same provenance metadata.
    """

    if not isinstance(payload, Mapping):
        raise TypeError("V27a cache payload must be a mapping to hash")
    body = {key: value for key, value in payload.items() if key != "cache_payload_sha256"}
    digest = hashlib.sha256()

    def visit(value: Any) -> None:
        if isinstance(value, torch.Tensor):
            visit(value.detach().cpu().numpy())
            return
        if isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
            if array.dtype.hasobject:
                raise TypeError("V27a cache hash does not permit object arrays")
            _digest_cache_chunk(digest, b"ndarray")
            _digest_cache_chunk(digest, array.dtype.str.encode("ascii"))
            _digest_cache_chunk(digest, repr(tuple(array.shape)).encode("ascii"))
            _digest_cache_chunk(digest, memoryview(array).cast("B"))
            return
        if isinstance(value, np.generic):
            visit(value.item())
            return
        if value is None:
            _digest_cache_chunk(digest, b"none")
            return
        if isinstance(value, bool):
            _digest_cache_chunk(digest, b"bool")
            _digest_cache_chunk(digest, b"1" if value else b"0")
            return
        if isinstance(value, int):
            _digest_cache_chunk(digest, b"int")
            _digest_cache_chunk(digest, str(value).encode("ascii"))
            return
        if isinstance(value, float):
            if not np.isfinite(value):
                raise ValueError("V27a cache hash does not permit non-finite floats")
            _digest_cache_chunk(digest, b"float64")
            _digest_cache_chunk(digest, np.float64(value).tobytes())
            return
        if isinstance(value, str):
            _digest_cache_chunk(digest, b"str")
            _digest_cache_chunk(digest, value.encode("utf-8"))
            return
        if isinstance(value, (list, tuple)):
            _digest_cache_chunk(digest, b"list" if isinstance(value, list) else b"tuple")
            _digest_cache_chunk(
                digest, len(value).to_bytes(8, byteorder="big", signed=False)
            )
            for item in value:
                visit(item)
            return
        if isinstance(value, Mapping):
            if any(not isinstance(key, str) for key in value):
                raise TypeError("V27a cache hash requires string mapping keys")
            _digest_cache_chunk(digest, b"mapping")
            _digest_cache_chunk(
                digest, len(value).to_bytes(8, byteorder="big", signed=False)
            )
            for key in sorted(value):
                visit(key)
                visit(value[key])
            return
        raise TypeError(
            f"V27a cache hash does not support {type(value).__name__} values"
        )

    visit(body)
    return digest.hexdigest()
































__all__ = ["cache_payload_sha256"]