"""通用数据缓存的契约和异常。"""

from __future__ import annotations

from radar_ln3.data.contracts import DataError


class CacheError(DataError):
    """缓存处理错误的基类。"""


class CachePathError(CacheError):
    """缓存路径越界或不可用。"""


class CacheFormatError(CacheError):
    """缓存文件结构无效。"""


class CacheIntegrityError(CacheError):
    """缓存 hash、数量或来源绑定不一致。"""


CACHE_SCHEMA_VERSION = "RADAR-LN3-EVENT-SOURCE-CACHE-V3"
CACHE_FORMAT = "jsonl"
CACHE_KIND = "event_source_reference_runtime_lru"


__all__ = [
    "CacheError",
    "CachePathError",
    "CacheFormatError",
    "CacheIntegrityError",
    "CACHE_SCHEMA_VERSION",
    "CACHE_FORMAT",
    "CACHE_KIND",
]
