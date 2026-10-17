"""V27d/r17 的冻结迁移评估。

评估分为 L1、L2 和无 L3 的 raw route preview。由于 r17 没有 RD、
micro-Doppler、Unknown 或可用 L3 校准，报告不会把 raw route preview
称为最终产品动作或部署结果。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from v27.v27_baseline.metrics import summarize_records

from .data import R17Dataset, build_r17_loader, r17_l1_inputs, r17_l2_inputs
from .frozen import (
    FrozenR17Models,
    load_degraded_l2_checkpoint,
    load_frozen_models,
)
from .protocol import R17_CLASS_NAMES, R17Config, R17_EXPERIMENT_ID, R17_L1_CLASS_NAMES, R17_L2_CLASS_NAMES, R17_REPORT_SCHEMA, artifact_path, build_manifest, project_path, seal_payload, sha256_file, write_json_if_identical


def _device(config: R17Config, requested: str | None = None) -> torch.device:
    value = str(requested or config.training.get("device", "cpu")).lower()
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("r17 requested CUDA but CUDA is unavailable")
        return torch.device("cuda:0")
    if value == "cpu":
        return torch.device("cpu")
    raise ValueError(f"unsupported r17 device: {value}")


def _trim_rd(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """TR-only 运行只保留一个无效 placeholder frame，避免无效 RD 占用显存。"""

    result = dict(inputs)
    result["rd_frames"] = result["rd_frames"][:, :, :1]
    result["rd_frame_mask"] = result["rd_frame_mask"][:, :, :1]
    return result


def _softmax(value: torch.Tensor) -> torch.Tensor:
    return torch.softmax(value, dim=-1)


def _parent_metrics(records: list[dict[str, Any]], class_names: tuple[str, ...], scope: str) -> dict[str, Any]:
    return summarize_records(records, class_names=class_names, scope=scope)


def _run_l1(
    root: Path,
    config: R17Config,
    frozen: FrozenR17Models,
    *,
    split_name: str,
    device: torch.device,
) -> list[dict[str, Any]]:
    dataset = R17Dataset(root, config, split_name=split_name, task="l1")
    loader = build_r17_loader(
        dataset,
        batch_size=int(config.training["batch_size"]),
        shuffle=False,
    )
    model = frozen.l1_model.to(device).eval()
    records: list[dict[str, Any]] = []
    with torch.inference_mode():
        for batch in loader:
            inputs = _trim_rd(r17_l1_inputs(batch, device))
            output = model(
                base_inputs=inputs,
                observation_increment_values=batch["observation_increment_values"].to(device),
            )
            logits = output["final_logits"].detach().cpu()
            sequence = output["sequence_logits"].detach().cpu()
            lengths = batch["lengths"].tolist()
            for row, parent_id in enumerate(batch["parent_sample_ids"]):
                length = int(lengths[row])
                final = logits[row]
                records.append(
                    {
                        "parent_sample_id": str(parent_id),
                        "split_name": split_name,
                        "label_name": str(batch["label_names"][row]),
                        "true_id": int(batch["l1_labels"][row]),
                        "pred_id": int(final.argmax().item()),
                        "logits": [float(value) for value in final.tolist()],
                        "probabilities": [float(value) for value in _softmax(final).tolist()],
                        "event_count": length,
                        "raw_event_count": int(batch["raw_event_counts"][row]),
                        "truncated": bool(batch["model_truncation_applied"][row]),
                        "final_raw_state_norm": float(torch.linalg.vector_norm(output["liquid_sequence"][row, length - 1]).item()),
                        "rd_valid_event_count": int(inputs["rd_frame_mask"][row, :length].any(dim=-1).sum().item()),
                        "rd_missing_event_count": int(
                            inputs["rd_quality"][row, :length, 3].gt(0.5).sum().item()
                        ),
                        "sequence_final_consistency": bool(torch.equal(sequence[row, length - 1], final)),
                    }
                )
    return records


def _run_l2(
    root: Path,
    config: R17Config,
    frozen: FrozenR17Models,
    *,
    split_name: str,
    device: torch.device,
    model_override: torch.nn.Module | None = None,
) -> list[dict[str, Any]]:
    dataset = R17Dataset(root, config, split_name=split_name, task="l2")
    loader = build_r17_loader(
        dataset,
        batch_size=int(config.training["batch_size"]),
        shuffle=False,
    )
    model = (frozen.l2_model if model_override is None else model_override).to(device).eval()
    records: list[dict[str, Any]] = []
    with torch.inference_mode():
        for batch in loader:
            inputs = _trim_rd(r17_l2_inputs(batch, device))
            output = model(**inputs)
            sequence = output["sequence_logits"]
            logits = output["window_logits"].detach().cpu()
            sequence_cpu = sequence.detach().cpu()
            lengths = batch["lengths"].tolist()
            for row, parent_id in enumerate(batch["parent_sample_ids"]):
                length = int(lengths[row])
                final = logits[row]
                records.append(
                    {
                        "parent_sample_id": str(parent_id),
                        "split_name": split_name,
                        "label_name": str(batch["label_names"][row]),
                        "true_id": int(batch["l2_labels"][row]),
                        "pred_id": int(final.argmax().item()),
                        "logits": [float(value) for value in final.tolist()],
                        "probabilities": [float(value) for value in _softmax(final).tolist()],
                        "event_count": length,
                        "raw_event_count": int(batch["raw_event_counts"][row]),
                        "truncated": bool(batch["model_truncation_applied"][row]),
                        "final_raw_state_norm": float(torch.linalg.vector_norm(output["liquid_sequence"][row, length - 1]).item()),
                        "rd_valid_event_count": int(inputs["rd_frame_mask"][row, :length].any(dim=-1).sum().item()),
                        "rd_missing_event_count": int(
                            inputs["rd_quality"][row, :length, 3].gt(0.5).sum().item()
                        ),
                        "l2_range_clipped_event_count": int(
                            batch["l2_range_clipped_count"][row].item()
                        ),
                        "l2_speed_clipped_event_count": int(
                            batch["l2_speed_clipped_count"][row].item()
                        ),
                        "sequence_final_consistency": bool(torch.equal(sequence_cpu[row, length - 1], final)),
                    }
                )
    return records


def _route_preview(
    l1_records: list[dict[str, Any]],
    l2_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    l2_by_parent = {str(row["parent_sample_id"]): row for row in l2_records}
    output: list[dict[str, Any]] = []
    for l1 in l1_records:
        parent_id = str(l1["parent_sample_id"])
        label = str(l1["label_name"])
        l1_pred = int(l1["pred_id"])
        if l1_pred == 1:
            pred_id = 3
            reason = "L1_CLUTTER_PREVIEW"
        elif parent_id in l2_by_parent:
            pred_id = int(l2_by_parent[parent_id]["pred_id"])
            reason = "L1_TARGET_L2_CLASS_PREVIEW"
        else:
            pred_id = -1
            reason = "L1_TARGET_WITHOUT_L2_PARENT"
        true_id = R17_CLASS_NAMES.index(label)
        output.append(
            {
                "parent_sample_id": parent_id,
                "label_name": label,
                "true_id": true_id,
                "pred_id": pred_id,
                "l1_pred_id": l1_pred,
                "l2_pred_id": l2_by_parent[parent_id]["pred_id"] if parent_id in l2_by_parent else None,
                "reason": reason,
                "l3_action": "PENDING_ABSTAIN",
                "l3_reason": "CALIBRATION_NOT_FITTED_OR_UNAVAILABLE",
            }
        )
    return output


def _safe_metrics(records: list[dict[str, Any]], classes: tuple[str, ...], scope: str) -> dict[str, Any]:
    if not records:
        return {"scope": scope, "support": 0, "status": "no_records"}
    return _parent_metrics(records, classes, scope)


def evaluate_frozen(
    project_root: str | Path,
    config: R17Config,
    *,
    device: str | None = None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    frozen = load_frozen_models(root, config)
    runtime_device = _device(config, device)
    reports: dict[str, Any] = {}
    all_route: list[dict[str, Any]] = []
    telemetry: dict[str, Any] = {}
    for split in ("train", "validation"):
        l1 = _run_l1(root, config, frozen, split_name=split, device=runtime_device)
        l2 = _run_l2(root, config, frozen, split_name=split, device=runtime_device)
        route = _route_preview(l1, l2)
        all_route.extend(route)
        reports[split] = {
            "l1": {"records": l1, "metrics": _safe_metrics(l1, R17_L1_CLASS_NAMES, f"{split}:l1_parent_final")},
            "l2": {"records": l2, "metrics": _safe_metrics(l2, R17_L2_CLASS_NAMES, f"{split}:l2_parent_final")},
            "raw_route_preview": {
                "records": route,
                "metrics": _safe_metrics([row for row in route if int(row["pred_id"]) >= 0], R17_CLASS_NAMES, f"{split}:raw_route_preview_parent_final"),
            },
        }
        telemetry[split] = {
            "l1_parent_count": len(l1),
            "l2_parent_count": len(l2),
            "l1_truncated_parent_count": sum(bool(row["truncated"]) for row in l1),
            "l2_truncated_parent_count": sum(bool(row["truncated"]) for row in l2),
            "l1_final_raw_state_norm": {
                "min": float(min(row["final_raw_state_norm"] for row in l1)),
                "max": float(max(row["final_raw_state_norm"] for row in l1)),
            },
            "l2_final_raw_state_norm": {
                "min": float(min(row["final_raw_state_norm"] for row in l2)),
                "max": float(max(row["final_raw_state_norm"] for row in l2)),
            },
            "rd_valid_event_count": 0,
            "rd_missing_event_count": sum(row["rd_missing_event_count"] for row in l1),
            "l2_range_clipped_event_count": sum(
                row["l2_range_clipped_event_count"] for row in reports[split]["l2"]["records"]
            ),
            "l2_speed_clipped_event_count": sum(
                row["l2_speed_clipped_event_count"] for row in reports[split]["l2"]["records"]
            ),
            "snr_model_input": "missing_zero_with_validity_mask",
        }
    payload = {
        "schema_version": R17_REPORT_SCHEMA,
        "experiment_id": R17_EXPERIMENT_ID,
        "status": "completed_frozen_tr_only_migration_audit",
        "device": str(runtime_device),
        "config_sha256": config.sha256,
        "config_identity_sha256": config.identity_sha256,
        "frozen_layers": {
            "l1": {"experiment_id": "V27d-r15", "arm": "v10_observation_increment", "seed": 42, "epoch": 11, "checkpoint_sha256": config.source["r15_checkpoint_sha256"]},
            "l2": {"experiment_id": "V27c-r23", "arm": "robust_control", "seed": 42, "epoch": 1, "checkpoint_sha256": config.source["r23_checkpoint_sha256"], "stable8_residual_used": False},
        },
        "data_scope": {"source": "OriData/excel", "test": "not_created", "unknown_payload": "not_created", "rd_payload": "not_available", "snr_used": False},
        "l3": {"status": "unavailable", "action_replay": "not_executed", "deployment_eligible": False},
        "splits": reports,
        "telemetry": telemetry,
        "not_proven": ["CQ08 full-modal performance", "strict open-set Unknown rejection", "L3 calibrated actions", "deployment performance"],
    }
    sealed = seal_payload(payload)
    report_path = artifact_path(
        root, config.artifacts["frozen_report_path"], task="l1", seed=42
    )
    write_json_if_identical(report_path, sealed)
    return {"report_path": str(report_path), **payload}


def evaluate_l2_checkpoint(
    project_root: str | Path,
    config: R17Config,
    *,
    checkpoint_path: str | Path,
    device: str | None = None,
) -> dict[str, Any]:
    """Evaluate a trained r17 L2 checkpoint with the frozen r17 L1 route."""

    root = Path(project_root).resolve()
    frozen = load_frozen_models(root, config)
    trained_l2, checkpoint = load_degraded_l2_checkpoint(
        root, config, frozen, checkpoint_path
    )
    runtime_device = _device(config, device)
    reports: dict[str, Any] = {}
    for split in ("train", "validation"):
        l1 = _run_l1(root, config, frozen, split_name=split, device=runtime_device)
        l2 = _run_l2(
            root,
            config,
            frozen,
            split_name=split,
            device=runtime_device,
            model_override=trained_l2,
        )
        route = _route_preview(l1, l2)
        reports[split] = {
            "l1": {
                "records": l1,
                "metrics": _safe_metrics(
                    l1, R17_L1_CLASS_NAMES, f"{split}:l1_parent_final"
                ),
            },
            "l2": {
                "records": l2,
                "metrics": _safe_metrics(
                    l2, R17_L2_CLASS_NAMES, f"{split}:l2_parent_final"
                ),
            },
            "raw_route_preview": {
                "records": route,
                "metrics": _safe_metrics(
                    [row for row in route if int(row["pred_id"]) >= 0],
                    R17_CLASS_NAMES,
                    f"{split}:raw_route_preview_parent_final",
                ),
            },
        }
    payload = {
        "schema_version": R17_REPORT_SCHEMA,
        "experiment_id": R17_EXPERIMENT_ID,
        "status": "completed_l2_tr_only_degraded_checkpoint_evaluation",
        "device": str(runtime_device),
        "config_sha256": config.sha256,
        "config_identity_sha256": config.identity_sha256,
        "checkpoint": {
            "path": str(project_path(root, checkpoint_path)),
            "sha256": sha256_file(project_path(root, checkpoint_path)),
            "seed": int(checkpoint["seed"]),
            "epoch": int(checkpoint["epoch"]),
            "manifest_sha256": checkpoint["manifest_sha256"],
            "model_state_sha256": checkpoint["model_state_sha256"],
        },
        "provenance": {
            "training_manifest_code_hashes_match_current": manifest_code_hashes_match_current(
                root, config
            ),
            "evaluation_code_drift_policy": "only code_hashes may differ; config/data/source/model contract must match",
        },
        "frozen_l1": {
            "experiment_id": "V27d-r15",
            "arm": "v10_observation_increment",
            "seed": 42,
            "epoch": 11,
            "checkpoint_sha256": config.source["r15_checkpoint_sha256"],
        },
        "data_scope": {
            "source": "OriData/excel",
            "test": "not_created",
            "unknown_payload": "not_created",
            "rd_payload": "not_available",
            "snr_used": False,
        },
        "l3": {
            "status": "unavailable",
            "action_replay": "not_executed",
            "deployment_eligible": False,
        },
        "splits": reports,
        "not_proven": [
            "CQ08 full-modal performance",
            "strict open-set Unknown rejection",
            "L3 calibrated actions",
            "deployment performance",
        ],
    }
    sealed = seal_payload(payload)
    report_path = artifact_path(
        root,
        config.artifacts["report_template"],
        task="l2",
        seed=int(checkpoint["seed"]),
    )
    report_path = report_path.with_name(report_path.stem + "-v2" + report_path.suffix)
    write_json_if_identical(report_path, sealed)
    return {"report_path": str(report_path), **payload}


def manifest_code_hashes_match_current(root: Path, config: R17Config) -> bool:
    """Report whether the training manifest's recorded r17 code is current."""

    manifest, _, _ = __import__("v27.V27d.r17.protocol", fromlist=["load_manifest"]).load_manifest(
        root, config, allow_evaluation_code_drift=True
    )
    return manifest.get("code_hashes") == build_manifest(root, config).get("code_hashes")


__all__ = ["evaluate_frozen", "evaluate_l2_checkpoint"]