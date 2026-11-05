"""产品数据层的稳定公共 API。"""

from .adapter import DataSourceAdapter
from .asd import ASDAdapter, load_asd
from .contracts import (
    DatasetView,
    DatasetManifest,
    EventData,
    EventRecord,
    EventSequence,
    ParentRecord,
    RawEchoFrame,
    RawEchoProvider,
    RawTargetRecord,
)
from .cq08 import CQ08Adapter
from .canonical_rd import (
    CANONICAL_RD_EVENT_INDEX_NAME,
    CANONICAL_RD_MANIFEST_NAME,
    CANONICAL_RD_RELATIVE_ROOT,
    CANONICAL_RD_SCHEMA_VERSION,
    CANONICAL_RD_VELOCITY_AXIS_HASH,
    CanonicalRDStore,
    canonical_rd_binding,
    canonical_rd_manifest_path,
    canonical_rd_root,
    load_canonical_rd_store,
    materialize_canonical_rd_store,
)
from .rd_canonical import (
    CANONICALIZATION_VERSION,
    CANONICAL_RANGE_BINS,
    CANONICAL_RANGE_OFFSETS_M,
    CANONICAL_VELOCITY_AXIS_SHA256,
    CANONICAL_VELOCITY_BINS,
    CANONICAL_VELOCITY_MPS,
    CanonicalRDFrame,
    canonical_axis_descriptor,
    canonicalize_rd_frame,
)
from .dataset import ParentSequenceDataset
from .manifest import (
    expected_baseline_training_view,
    load_manifest,
    resolve_manifest_view,
    validate_baseline_training_view,
    validate_manifest,
)
from .source import (
    SOURCE_CONFIG_SCHEMA,
    load_source_config,
    resolve_registered_source_path,
)
from .derived import CSV_PATH, build_event_csv_index, register_product_artifacts
from .temporal import (
    RD_FRAME_TIME_SCHEMA_VERSION,
    TR_DT_REASON_NEGATIVE,
    TR_TIME_SCHEMA_VERSION,
    build_temporal_metadata,
)

__all__ = [
    "DataSourceAdapter",
    "ASDAdapter",
    "load_asd",
    "DatasetManifest",
    "DatasetView",
    "EventData",
    "EventRecord",
    "EventSequence",
    "ParentRecord",
    "CQ08Adapter",
    "CANONICAL_RD_EVENT_INDEX_NAME",
    "CANONICAL_RD_MANIFEST_NAME",
    "CANONICAL_RD_RELATIVE_ROOT",
    "CANONICAL_RD_SCHEMA_VERSION",
    "CANONICAL_RD_VELOCITY_AXIS_HASH",
    "CanonicalRDStore",
    "canonical_rd_binding",
    "canonical_rd_manifest_path",
    "canonical_rd_root",
    "load_canonical_rd_store",
    "materialize_canonical_rd_store",
    "CANONICALIZATION_VERSION",
    "CANONICAL_RANGE_BINS",
    "CANONICAL_RANGE_OFFSETS_M",
    "CANONICAL_VELOCITY_AXIS_SHA256",
    "CANONICAL_VELOCITY_BINS",
    "CANONICAL_VELOCITY_MPS",
    "CanonicalRDFrame",
    "canonical_axis_descriptor",
    "canonicalize_rd_frame",
    "ParentSequenceDataset",
    "load_manifest",
    "resolve_manifest_view",
    "expected_baseline_training_view",
    "validate_baseline_training_view",
    "validate_manifest",
    "build_temporal_metadata",
    "TR_TIME_SCHEMA_VERSION",
    "TR_DT_REASON_NEGATIVE",
    "RD_FRAME_TIME_SCHEMA_VERSION",
    "RawEchoFrame",
    "RawEchoProvider",
    "RawTargetRecord",
    "SOURCE_CONFIG_SCHEMA",
    "load_source_config",
    "resolve_registered_source_path",
    "CSV_PATH",
    "build_event_csv_index",
    "register_product_artifacts",
]
