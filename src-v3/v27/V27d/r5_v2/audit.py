"""Sample-level audit for the frozen V27d-r5 control Bird-route errors."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from v27.V27d.r5.training import _head_inputs, load_checkpoint

from .features import load_population_cache
from .protocol import artifact_path, load_context, read_sealed_json, write_sealed_json


EVENT_FACT_NAMES = (
    "rd_present",
    "rd_frame_count_scaled",
    "rd_local_duration_scaled",
    "rd_missing",
    "rd_local_time_non_monotonic",
    "rd_multiple_frames",
    "rd_frames_truncated",
    "dt_seconds_scaled",
    "dt_valid",
    "dt_first_event",
    "dt_zero_delta",
    "dt_positive_delta",
    "rd_frame_observed",
    "physical_valid",
    "micro_doppler_valid",
    "course_value",
    "course_valid",
    "course_support_scaled",
    "zero_delta_ordinal_scaled",
)


def _slice_norm(trajectory: torch.Tensor, start: int, stop: int) -> torch.Tensor:
    return torch.linalg.vector_norm(trajectory[..., start:stop], dim=-1)


def _finite_float(value: torch.Tensor) -> list[float]:
    return [float(item) for item in value.detach().cpu().tolist()]


def _threshold_from_r5_report(root: Path, config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    report = read_sealed_json(root / config["source"]["r5_report_path"])
    control = report["arms"]["control"]
    score_record = control["calibration_transfer"]["bird_knownness_probability"]
    threshold_source = score_record["threshold_source"]
    if threshold_source.get("higher_is_known") is not True or threshold_source.get("threshold") is None:
        raise ValueError("V27d-r5 control threshold is unavailable")
    return report, threshold_source


def _event_record(
    *,
    event_index: int,
    score: float,
    score_delta: float | None,
    quality: bool,
    task_facts: torch.Tensor,
    event_facts: torch.Tensor,
    endpoint_logits: torch.Tensor,
    raw_trajectory: torch.Tensor,
    stable_summary: torch.Tensor,
    stable_delta: float | None,
    actual_update_fraction: float,
    prefix_bird_fraction: float,
    prefix_class_consistency: float,
) -> dict[str, Any]:
    competitors = torch.cat((endpoint_logits[:1], endpoint_logits[2:]))
    bird_margin = endpoint_logits[1] - competitors.max()
    class_index = int(endpoint_logits.argmax().item())
    reason = "first_event_no_predecessor" if bool(event_facts[9]) else "zero_delta_at_second_precision" if bool(event_facts[10]) else "positive_delta_at_second_precision" if bool(event_facts[11]) else "padding"
    return {
        "event_index": event_index,
        "quality_eligible": bool(quality),
        "dt_seconds_scaled": float(event_facts[7]),
        "dt_valid": bool(event_facts[8]),
        "dt_reason": reason,
        "rd_quality": {name: float(event_facts[index]) for index, name in enumerate(EVENT_FACT_NAMES[:7])},
        "rd_frame_observed": bool(event_facts[12]),
        "physical_valid": bool(event_facts[13]),
        "micro_doppler_valid": bool(event_facts[14]),
        "task_facts": _finite_float(task_facts),
        "knownness_score": score,
        "knownness_score_delta": score_delta,
        "l2_predicted_class_index": class_index,
        "l2_predicted_bird": class_index == 1,
        "l2_bird_probability": float(torch.softmax(endpoint_logits, dim=0)[1]),
        "l2_bird_margin": float(bird_margin),
        "raw_state_norm": float(raw_trajectory[:48].norm()),
        "candidate_innovation_norm": float(raw_trajectory[48:96].norm()),
        "mixed_memory_innovation_norm": float(raw_trajectory[96:144].norm()),
        "actual_update_norm": float(raw_trajectory[144:192].norm()),
        "gate_mean": float(raw_trajectory[192:240].mean()),
        "alpha_mean": float(raw_trajectory[240:288].mean()),
        "tau_mean": float(raw_trajectory[288:336].mean()),
        "actual_update_fraction_of_track": actual_update_fraction,
        "stable8_summary": _finite_float(stable_summary),
        "stable8_event_delta_l2": stable_delta,
        "prefix_bird_fraction": prefix_bird_fraction,
        "prefix_class_consistency": prefix_class_consistency,
    }


def audit_error_samples(project_root: str | Path, config_path: str | Path | None = None) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config, manifest = load_context(root, config_path)
    calibration = load_population_cache(artifact_path(root, config["artifacts"]["cache_template"].format(role="calibration"), must_exist=True))
    r5_report, threshold_source = _threshold_from_r5_report(root, config)
    model, checkpoint, _ = load_checkpoint(root, arm="control", seed=42)
    model.eval()
    tensors = calibration["tensors"]
    indices = torch.arange(len(calibration["parent_ids"]))
    with torch.inference_mode():
        output = model(**_head_inputs(calibration, indices, torch.device("cpu")))
    scores = torch.sigmoid(output["knownness_logit_sequence"])
    final_scores = scores[torch.arange(len(indices)), tensors["lengths"] - 1]
    threshold = float(threshold_source["threshold"])
    negative = tensors["bird_route_negative"].bool()
    negative_rows = torch.where(negative)[0]
    accepted = final_scores.ge(threshold)
    if int(negative_rows.numel()) != 12:
        raise ValueError(f"V27d-r5-v2 expected 12 calibration Bird-route Unknowns, got {negative_rows.numel()}")
    unknown_records: list[dict[str, Any]] = []
    trajectory = tensors["event_trajectory"]
    for row_tensor in negative_rows:
        row = int(row_tensor)
        length = int(tensors["lengths"][row])
        row_scores = scores[row, :length]
        row_endpoint_logits = tensors["endpoint_sequence_logits"][row, :length]
        row_quality = tensors["quality_target_sequence"][row, :length].bool()
        row_event_facts = tensors["event_facts_sequence"][row, :length]
        row_task_facts = tensors["task_facts_sequence"][row, :length]
        row_trajectory = trajectory[row, :length]
        row_stable = tensors["stable8_summary"][row, :length]
        actual_norm = _slice_norm(row_trajectory, 144, 192)
        total_actual = float(actual_norm.sum().clamp_min(1.0e-8))
        actual_fraction = actual_norm / actual_norm.sum().clamp_min(1.0e-8)
        stable_delta = torch.linalg.vector_norm(row_stable[1:] - row_stable[:-1], dim=-1) if length > 1 else torch.zeros(0)
        pred = row_endpoint_logits.argmax(dim=-1)
        records: list[dict[str, Any]] = []
        for event in range(length):
            prefix = pred[: event + 1]
            counts = torch.bincount(prefix, minlength=3)
            consistency = float(counts.max().float() / float(event + 1))
            delta = None if event == 0 else float(row_scores[event] - row_scores[event - 1])
            records.append(_event_record(
                event_index=event + 1,
                score=float(row_scores[event]),
                score_delta=delta,
                quality=bool(row_quality[event]),
                task_facts=row_task_facts[event],
                event_facts=row_event_facts[event],
                endpoint_logits=row_endpoint_logits[event],
                raw_trajectory=row_trajectory[event],
                stable_summary=row_stable[event],
                stable_delta=None if event == 0 else float(stable_delta[event - 1]),
                actual_update_fraction=float(actual_fraction[event]),
                prefix_bird_fraction=float((prefix == 1).float().mean()),
                prefix_class_consistency=consistency,
            ))
        score_deltas = row_scores[1:] - row_scores[:-1] if length > 1 else torch.zeros(0)
        reversal_count = int(((score_deltas[:-1] * score_deltas[1:]) < 0).sum()) if score_deltas.numel() > 1 else 0
        stable_p90 = float(torch.quantile(stable_delta, 0.90)) if stable_delta.numel() else 0.0
        top_event = int(actual_norm.argmax()) + 1
        final_jump = float(row_scores[-1] - row_scores[-2]) if length > 1 else 0.0
        prefix_start = max(0, min(length - 1, 2))
        prefix_values = row_scores[prefix_start:]
        pattern_flags = {
            "low_quality_but_accepted": bool(not tensors["quality_target_final"][row]),
            "single_event_update_dominant": bool(float(actual_fraction.max()) >= float(config["audit"]["single_event_contribution_flag"])),
            "prefix_inconsistent": bool(float(prefix_values.max() - prefix_values.min()) >= float(config["audit"]["prefix_score_range_flag"]) or reversal_count >= int(config["audit"]["prefix_reversal_count_flag"])),
            "rd_micro_doppler_unstable": bool(stable_p90 >= float(config["audit"]["stable8_event_delta_p90_flag"])),
        }
        unknown_records.append({
            "row": row,
            "parent_id": calibration["parent_ids"][row],
            "label": calibration["label_names"][row],
            "r5_control_decision": "BIRD_ACCEPTED" if bool(accepted[row]) else "UNKNOWN_REJECTED",
            "r5_control_final_knownness_score": float(final_scores[row]),
            "r5_control_threshold": threshold,
            "effective_event_count": length,
            "quality_eligible_event_count": int(row_quality.sum()),
            "final_quality_eligible": bool(tensors["quality_target_final"][row]),
            "prefix_score_min_after_maturity": float(prefix_values.min()),
            "prefix_score_max_after_maturity": float(prefix_values.max()),
            "prefix_score_range_after_maturity": float(prefix_values.max() - prefix_values.min()),
            "prefix_score_reversal_count": reversal_count,
            "terminal_knownness_jump": final_jump,
            "top_actual_update_event": top_event,
            "top_actual_update_fraction": float(actual_fraction.max()),
            "stable8_event_delta_p90": stable_p90,
            "pattern_flags": pattern_flags,
            "events": records,
        })
    aggregate = {
        "negative_support": len(unknown_records),
        "accepted_unknown_count": sum(item["r5_control_decision"] == "BIRD_ACCEPTED" for item in unknown_records),
        "rejected_unknown_count": sum(item["r5_control_decision"] == "UNKNOWN_REJECTED" for item in unknown_records),
        "quality_ineligible_final_count": sum(not item["final_quality_eligible"] for item in unknown_records),
        "single_event_update_dominant_count": sum(item["pattern_flags"]["single_event_update_dominant"] for item in unknown_records),
        "prefix_inconsistent_count": sum(item["pattern_flags"]["prefix_inconsistent"] for item in unknown_records),
        "rd_micro_doppler_unstable_count": sum(item["pattern_flags"]["rd_micro_doppler_unstable"] for item in unknown_records),
        "threshold_support_warning": "Only 12 quality-eligible Bird-route Unknowns are available; all counts are diagnostic and formal L3 remains disabled.",
    }
    payload = {
        "schema_version": "RADAR-LN3-V27d-r5-bird-route-v2-error-audit-v1",
        "experiment_id": "V27d-r5-bird-route-v2",
        "audit_type": "frozen_r5_control_bird_route_error_samples",
        "direct_baseline": "V27d-r5-control",
        "seed": 42,
        "v2_manifest_sha256": manifest["manifest_sha256"],
        "r5_report_payload_sha256": r5_report["payload_sha256"],
        "r5_control_checkpoint": checkpoint.get("selected_epoch"),
        "r5_control_threshold_source": threshold_source,
        "formal_l3_enabled": False,
        "formal_unknown_support_required": int(config["calibration"]["formal_unknown_support_required"]),
        "aggregate": aggregate,
        "samples": unknown_records,
    }
    path = artifact_path(root, config["artifacts"]["audit_path"])
    write_sealed_json(path, payload)
    return {"status": "completed", "audit_path": str(path.relative_to(root)).replace("\\", "/"), "aggregate": aggregate}


__all__ = ["audit_error_samples"]
