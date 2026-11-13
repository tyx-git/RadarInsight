"""正式 L1/L2 训练和 L3 校准公共 API。"""

from .l1 import load_l1_base_model, train_l1
from .l2 import load_l2_base_model, train_l2
from .l3 import (
    L3ValidationEvidence,
    collect_l3_audit_evidence,
    collect_l3_validation_evidence,
    finalize_l3_audit,
    fit_l3_validation_calibration,
    save_and_register_l3_calibration,
)
from .l3_heads import (
    FrozenEndpointRepresentation,
    build_head_fit_dataset,
    build_head_validation_dataset,
    endpoint_representation_path,
    load_frozen_endpoint_representation,
    load_l3_knownness_head,
    materialize_frozen_endpoint_representation,
    save_frozen_endpoint_representation,
    score_frozen_endpoint_representation,
    train_l3_knownness_head,
)

__all__ = [
    "load_l1_base_model",
    "load_l2_base_model",
    "L3ValidationEvidence",
    "collect_l3_validation_evidence",
    "collect_l3_audit_evidence",
    "finalize_l3_audit",
    "fit_l3_validation_calibration",
    "save_and_register_l3_calibration",
    "train_l1",
    "train_l2",
    "FrozenEndpointRepresentation",
    "build_head_fit_dataset",
    "build_head_validation_dataset",
    "endpoint_representation_path",
    "load_frozen_endpoint_representation",
    "load_l3_knownness_head",
    "materialize_frozen_endpoint_representation",
    "save_frozen_endpoint_representation",
    "score_frozen_endpoint_representation",
    "train_l3_knownness_head",
]
