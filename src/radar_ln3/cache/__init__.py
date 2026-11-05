"""通用数据缓存公共 API。"""

from .contracts import (
    CACHE_FORMAT,
    CACHE_KIND,
    CACHE_SCHEMA_VERSION,
    CacheError,
    CacheFormatError,
    CacheIntegrityError,
    CachePathError,
)
from .registry import (
    DEFAULT_CACHE_RELATIVE_PATH,
    CachedParentSequenceDataset,
    ProgressCallback,
    build_event_cache,
    load_event_cache,
)
from .anchor import (
    ANCHOR_NAME,
    ANCHOR_SCHEMA_VERSION,
    EventLogitAnchor,
    anchor_payload_hash,
    load_event_logit_anchor,
    save_event_logit_anchor,
)

__all__ = [
    "CACHE_FORMAT",
    "CACHE_KIND",
    "CACHE_SCHEMA_VERSION",
    "CacheError",
    "CacheFormatError",
    "CacheIntegrityError",
    "CachePathError",
    "DEFAULT_CACHE_RELATIVE_PATH",
    "CachedParentSequenceDataset",
    "ProgressCallback",
    "build_event_cache",
    "load_event_cache",
    "ANCHOR_NAME",
    "ANCHOR_SCHEMA_VERSION",
    "EventLogitAnchor",
    "anchor_payload_hash",
    "load_event_logit_anchor",
    "save_event_logit_anchor",
]
