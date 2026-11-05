"""产品特征目录和历史证据登记。

这里登记的是已经抽离出的计算契约及其证据边界，不是训练开关，也不会
自动把某个历史实验加入 L1/L2/L3。历史结果只用于说明候选来源和限制。
"""

from __future__ import annotations

from typing import Any


FEATURE_REGISTRY_SCHEMA = "RADAR-LN3-FEATURE-REGISTRY-V1"


def feature_catalog() -> dict[str, Any]:
    """返回可审计的公共特征目录。"""

    return {
        "schema_version": FEATURE_REGISTRY_SCHEMA,
        "features": {
            "tr_observation": {
                "module": "radar_ln3.features.tr",
                "schema": "causal-tr-observation-v1",
                "status": "baseline_input_contract",
                "value_dim": 14,
                "missing_dim": 11,
                "causal_scope": "current TR observation only",
            },
            "tr_motion": {
                "module": "radar_ln3.features.motion",
                "schema": "causal-tr-motion-v1",
                "status": "baseline_optional_dynamic_contract",
                "value_dim": 9,
                "invalid_dt_policy": "zero values and false validity; no fabricated derivative",
            },
            "tr_observation_increment": {
                "module": "radar_ln3.features.motion",
                "schema": "causal-tr-observation-increment-v1",
                "status": "optional_time_independent_variant",
                "value_dim": 9,
                "historical_note": "历史时间扰动实验降低了类别 validity-rate 的不平衡；当前不是默认输入",
            },
            "course_circular_incoherence": {
                "module": "radar_ln3.features.course",
                "schema": "causal-prefix-course-circular-incoherence-v1",
                "feature_name": "causal_prefix_course_circular_incoherence",
                "status": "positive_local_candidate_optional",
                "value_dim": 1,
                "history": {
                    "source": "历史课程圆形离散度实验",
                    "finding": "KnownGate development parent-final result was positive and direct feature-zeroing changed Clutter recall",
                    "limitations": [
                        "development feature-discovery reuse",
                        "small Clutter support",
                        "no final open-set or deployment confirmation",
                    ],
                },
            },
            "rd_quality": {
                "module": "radar_ln3.features.rd",
                "schema": "causal-current-event-v1",
                "status": "baseline_quality_contract",
                "value_dim": 7,
                "causal_scope": "current RD event only",
            },
            "rd_physical4": {
                "module": "radar_ln3.features.physical",
                "schema": "current-event-rd-physical-spectrum-v1",
                "status": "baseline_optional_physical_contract",
                "value_dim": 4,
                "causal_scope": "current RD event only",
                "axis_policy": "velocity axis must be supplied and hashed by the caller",
                "history": {
                    "source": "历史 RD physical 实验",
                    "finding": "current-event RD physical branch had local directional/error evidence but failed the full three-seed adoption gate",
                    "status": "retain_as_optional_contract_pending_new_validation",
                },
            },
            "micro_doppler_stable8": {
                "module": "radar_ln3.features.micro_doppler",
                "schema": "raw-iq-broadband-multiscale-micro-doppler-summary-v1",
                "status": "positive_local_candidate_with_coupling_caveat",
                "value_dim": 8,
                "frame_evidence_dim": 23,
                "history": {
                    "source": "历史 stable8 实验",
                    "finding": "local DBB evidence was positive, but shared Fusion coupling failed the frozen adoption guard",
                    "policy": "retain complete 23-dim frame evidence for audit; expose only selected stable8 summary to a model after independent validation",
                },
            },
        },
        "rejected_or_unconfirmed_historical_candidates": {
            "v21a_interframe_spectral": {
                "source": "历史特征筛选记录",
                "status": "not_positive_signal",
                "reason": "paired report did not pass positive or robust-positive signal gates",
            },
            "v21b_modal_consistency": {
                "source": "历史特征筛选记录",
                "status": "not_positive_signal",
                "reason": "paired report did not pass positive or robust-positive signal gates",
            },
            "v21c_target_row_isolation": {
                "source": "历史特征筛选记录",
                "status": "not_positive_signal",
                "reason": "paired report did not pass positive or robust-positive signal gates",
            },
            "v21d_causal_d_range": {
                "source": "历史特征筛选记录",
                "status": "not_positive_signal",
                "reason": "paired report did not pass positive or robust-positive signal gates",
            },
        },
        "excluded_from_features": {
            "density_weighted": {
                "reason": "Liquid state update mode, not an input feature",
                "history": "历史状态更新实验",
            },
            "model_prediction_and_loss": "owned by model/training layers",
            "l1_l2_l3_semantics": "owned by task and decision layers",
        },
        "cache_policy": {
            "schema": "RADAR-LN3-FEATURE-CACHE-V2",
            "required_view_binding": ["view_name", "view_hash", "data_binding"],
            "rd_processing_contract": "radar_ln3-canonical-rd-v1-prefer-local",
            "training_view_resolution": "caller-owned unified DatasetView; no feature cache default",
            "unknown_training_exposure": "must be decided by the supplied view, never inferred from missing fields",
        },
    }


__all__ = ["FEATURE_REGISTRY_SCHEMA", "feature_catalog"]
