"""Export per-parent raw endpoint and frozen L3 audit diagnostics."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export per-parent L1/L2 and frozen L3 audit diagnostics"
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument(
        "--calibration-path",
        type=Path,
        default=Path("artifacts/radar_ln3/l3/radar_ln3_calibration.json"),
    )
    parser.add_argument("--l1-checkpoint-path", type=Path, default=None)
    parser.add_argument("--l2-checkpoint-path", type=Path, default=None)
    parser.add_argument("--l1-config-path", type=Path, default=None)
    parser.add_argument("--l2-config-path", type=Path, default=None)
    parser.add_argument("--model-config-path", type=Path, default=Path("src/config/model.json"))
    parser.add_argument("--audit-manifest-path", type=Path, required=True)
    parser.add_argument(
        "--audit-feature-cache-manifest-path", type=Path, required=True
    )
    parser.add_argument(
        "--representation-root",
        type=Path,
        default=Path("artifacts/radar_ln3/l3/representations"),
        help="冻结 endpoint representation 缓存目录",
    )
    parser.add_argument("--audit-split-name", default="test")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--l1-head-path", type=Path, default=None)
    parser.add_argument("--l2-head-path", type=Path, default=None)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("artifacts/radar_ln3/l3/audit-report.json"),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip-data-validation",
        action="store_true",
        help="复用前置 data_index 校验结果，不重复扫描完整 TR/RD 文件",
    )
    return parser


def _softmax_row(logits: Sequence[float], temperature: float) -> list[float]:
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("audit logits 必须是一维有限数组")
    if not np.isfinite(float(temperature)) or float(temperature) <= 0.0:
        raise ValueError("calibration temperature 必须为正数")
    shifted = values / float(temperature)
    shifted -= float(shifted.max())
    probabilities = np.exp(shifted)
    probabilities /= float(probabilities.sum())
    return [float(value) for value in probabilities.tolist()]


def _candidate_set(
    probabilities: Sequence[float],
    class_names: Sequence[str],
    floors: Mapping[str, Any],
) -> list[str]:
    return [
        str(name)
        for index, name in enumerate(class_names)
        if floors.get(str(name)) is not None
        and float(probabilities[index]) + 1.0e-12 >= float(floors[str(name)])
    ]


def _knownness(
    probabilities: Sequence[float],
    class_names: Sequence[str],
    thresholds: Mapping[str, Any],
    *,
    target_override: Mapping[str, Any] | None = None,
) -> tuple[str, float, float | None]:
    predicted = str(class_names[int(np.argmax(np.asarray(probabilities)))])
    route = "target" if predicted == "KNOWN_TARGET_EVIDENCE" else "clutter"
    selected = thresholds.get(route, {}).get("selected")
    threshold = None if not isinstance(selected, Mapping) else float(selected["threshold"])
    if route == "target" and isinstance(target_override, Mapping):
        threshold = float(target_override["threshold"])
    return predicted, float(max(probabilities)), threshold


def _margin_and_entropy(probabilities: Sequence[float]) -> tuple[float, float, float]:
    values = np.sort(np.asarray(probabilities, dtype=np.float64))[::-1]
    margin = float(values[0] - values[1]) if values.size >= 2 else float(values[0])
    entropy = float(-np.sum(np.asarray(probabilities, dtype=np.float64) * np.log(
        np.clip(np.asarray(probabilities, dtype=np.float64), 1.0e-12, 1.0)
    )))
    return float(values[0]), margin, entropy


def _expected_endpoint(label: str) -> tuple[str | None, str | None]:
    if label in {"DroneTarget", "BirdTarget", "BalloonTarget"}:
        return "KNOWN_TARGET_EVIDENCE", label
    if label == "ClutterTarget":
        return "KNOWN_CLUTTER_EVIDENCE", None
    return None, None


def _atomic_json(path: Path, payload: Mapping[str, Any], *, overwrite: bool) -> None:
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode(
        "utf-8"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        if path.read_bytes() == encoded:
            return
        raise FileExistsError(f"拒绝覆盖不同的审计报告：{path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _mean(values: list[float]) -> float | None:
    return None if not values else float(np.mean(np.asarray(values, dtype=np.float64)))


def run(args: argparse.Namespace) -> dict[str, Any]:
    from radar_ln3.decision import load_l3_calibration
    from radar_ln3.decision.l3 import (
        L1_CLASSES,
        L2_CLASSES,
        L3CalibrationPolicy,
        _effective_l1_conformal_floors,
        resolve_l3,
    )
    from radar_ln3.training import collect_l3_audit_evidence
    from radar_ln3.training.l3 import _collect_l3_head_scores

    root = args.project_root.resolve()
    calibration = load_l3_calibration(root, args.calibration_path)
    evidence = collect_l3_audit_evidence(
        root,
        calibration,
        manifest_path=args.audit_manifest_path,
        feature_cache_manifest_path=args.audit_feature_cache_manifest_path,
        thresholds_frozen_before_access=True,
        split_name=str(args.audit_split_name),
        device=str(args.device),
        batch_size=args.batch_size,
        verify_files=not bool(args.skip_data_validation),
        representation_root=args.representation_root,
        l1_checkpoint_path=args.l1_checkpoint_path,
        l2_checkpoint_path=args.l2_checkpoint_path,
        l1_config_path=args.l1_config_path,
        l2_config_path=args.l2_config_path,
        model_config_path=args.model_config_path,
    )
    if dict(evidence.endpoint_sha256) != dict(calibration.get("endpoint_sha256") or {}):
        raise ValueError("L3 audit report endpoint hash 与 calibration 不一致")
    head_scores = None
    if calibration.get("knownness_source") == "independent_l3_head":
        head_ids, l1_head_scores, l2_head_scores, head_hashes, _ = _collect_l3_head_scores(
            root,
            manifest_path=args.audit_manifest_path,
            feature_cache_manifest_path=args.audit_feature_cache_manifest_path,
            split_name=str(args.audit_split_name),
            training_view=False,
            l1_head_path=args.l1_head_path or "artifacts/radar_ln3/l3/heads/l1-knownness.pt",
            l2_head_path=args.l2_head_path or "artifacts/radar_ln3/l3/heads/l2-knownness.pt",
            device=str(args.device),
            batch_size=args.batch_size,
            require_training_manifest_match=False,
            verify_files=not bool(args.skip_data_validation),
            representation_root=args.representation_root,
            l1_checkpoint_path=args.l1_checkpoint_path,
            l2_checkpoint_path=args.l2_checkpoint_path,
            l1_config_path=args.l1_config_path,
            l2_config_path=args.l2_config_path,
            model_config_path=args.model_config_path,
        )
        if head_ids != evidence.parent_ids:
            raise ValueError("L3 audit knownness head parent binding 不一致")
        if dict(calibration.get("head_checkpoint_sha256") or {}) != head_hashes:
            raise ValueError("L3 audit knownness head hash 与 calibration 不一致")
        head_scores = (l1_head_scores, l2_head_scores)
    policy = L3CalibrationPolicy(**dict(calibration["policy"]))
    l1_temperature = float(calibration["temperature"]["l1"]["temperature"])
    l2_temperature = float(calibration["temperature"]["l2"]["temperature"])
    l1_floors = _effective_l1_conformal_floors(calibration)
    l2_floors = calibration["conformal"]["l2"][
        "candidate_probability_floor_by_class"
    ]
    records: list[dict[str, Any]] = []
    label_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    by_label_action: dict[str, Counter[str]] = defaultdict(Counter)
    by_label_reason: dict[str, Counter[str]] = defaultdict(Counter)
    endpoint_errors = Counter()
    bird_knownness: dict[str, list[float]] = {
        "known_bird": [],
        "unknown_predicted_bird": [],
        "known_bird_predicted_other": [],
    }
    bird_margin: dict[str, list[float]] = {
        "known_bird": [],
        "unknown_predicted_bird": [],
        "known_bird_predicted_other": [],
    }

    for index, parent_id in enumerate(evidence.parent_ids):
        label = str(evidence.label_names[index])
        l1_logits = [float(value) for value in evidence.l1_logits[index].tolist()]
        l2_logits = [float(value) for value in evidence.l2_logits[index].tolist()]
        l1_endpoint_probabilities = _softmax_row(l1_logits, l1_temperature)
        if head_scores is not None and calibration.get("l1_router_source") == "independent_l3_head":
            route_values = np.asarray(head_scores[0][index], dtype=np.float64)
            route_total = float(route_values.sum())
            if route_total <= 1.0e-12:
                raise ValueError("L1 route head score 总和必须为正")
            l1_probabilities = [float(value) for value in (route_values / route_total).tolist()]
        else:
            l1_probabilities = l1_endpoint_probabilities
        l2_probabilities = _softmax_row(l2_logits, l2_temperature)
        l1_top1, l1_margin, l1_entropy = _margin_and_entropy(l1_endpoint_probabilities)
        l1_router_top1, l1_router_margin, l1_router_entropy = _margin_and_entropy(l1_probabilities)
        l1_endpoint_prediction = str(L1_CLASSES[int(np.argmax(np.asarray(l1_endpoint_probabilities)))])
        l1_router_prediction = str(L1_CLASSES[int(np.argmax(np.asarray(l1_probabilities)))])
        l2_top1, l2_margin, l2_entropy = _margin_and_entropy(l2_probabilities)
        l1_prediction, l1_knownness_score, l1_knownness_threshold = _knownness(
            l1_probabilities,
            L1_CLASSES,
            calibration["knownness_thresholds"]["l1"],
            target_override=calibration.get("l1_target_knownness_override"),
        )
        l2_prediction = str(L2_CLASSES[int(np.argmax(np.asarray(l2_probabilities)))])
        l2_knownness_score = float(max(l2_probabilities))
        l2_selected = calibration["knownness_thresholds"]["l2"].get(l2_prediction, {}).get(
            "selected"
        )
        l2_knownness_threshold = (
            None if not isinstance(l2_selected, Mapping) else float(l2_selected["threshold"])
        )
        if head_scores is not None:
            l1_route_index = 0 if l1_router_prediction == "KNOWN_TARGET_EVIDENCE" else 1
            l1_knownness_score = float(head_scores[0][index, l1_route_index])
            if calibration.get("l1_router_source") == "independent_l3_head":
                route_name = "target" if l1_route_index == 0 else "clutter"
                selected_route = calibration["knownness_thresholds"]["l1"][route_name].get("selected")
                l1_knownness_threshold = (
                    None
                    if not isinstance(selected_route, Mapping)
                    else float(selected_route["threshold"])
                )
            l2_knownness_score = float(head_scores[1][index, int(np.argmax(np.asarray(l2_probabilities)))])
        l1_candidates = _candidate_set(l1_probabilities, L1_CLASSES, l1_floors)
        l2_candidates = _candidate_set(l2_probabilities, L2_CLASSES, l2_floors)
        decision = resolve_l3(
            calibration,
            policy=policy,
            parent_sample_id=str(parent_id),
            l1_logits=l1_logits,
            l2_logits=l2_logits,
            l1_knownness_scores=None if head_scores is None else head_scores[0][index].tolist(),
            l2_knownness_scores=None if head_scores is None else head_scores[1][index].tolist(),
            streaming=False,
            allow_exploratory=True,
        )
        expected_l1, expected_l2 = _expected_endpoint(label)
        l1_correct = expected_l1 is not None and l1_endpoint_prediction == expected_l1
        l2_required = expected_l2 is not None
        l2_correct = expected_l2 is not None and l2_prediction == expected_l2
        if expected_l1 is not None and not l1_correct:
            endpoint_errors["l1_classification_error"] += 1
        if l2_required and not l2_correct:
            endpoint_errors["l2_classification_error"] += 1
        if label in {"BirdTarget", "UnknownTarget"}:
            if label == "BirdTarget" and l2_prediction == "BirdTarget":
                bird_knownness["known_bird"].append(l2_knownness_score)
                bird_margin["known_bird"].append(l2_margin)
            elif label == "BirdTarget":
                bird_knownness["known_bird_predicted_other"].append(l2_knownness_score)
                bird_margin["known_bird_predicted_other"].append(l2_margin)
            elif l2_prediction == "BirdTarget":
                bird_knownness["unknown_predicted_bird"].append(l2_knownness_score)
                bird_margin["unknown_predicted_bird"].append(l2_margin)
        action = str(decision["action"])
        reason = str(decision["reason_code"])
        label_counts[label] += 1
        action_counts[action] += 1
        reason_counts[reason] += 1
        by_label_action[label][action] += 1
        by_label_reason[label][reason] += 1
        records.append(
            {
                "parent_sample_id": str(parent_id),
                "label": label,
                "l1_logits": l1_logits,
                "l1_probabilities": l1_endpoint_probabilities,
                "l1_prediction": l1_endpoint_prediction,
                "l1_router_source": str(calibration.get("l1_router_source", "classifier_endpoint")),
                "l1_router_probabilities": l1_probabilities,
                "l1_router_prediction": l1_router_prediction,
                "l1_router_top1_probability": l1_router_top1,
                "l1_router_top2_margin": l1_router_margin,
                "l1_router_entropy": l1_router_entropy,
                "l1_top1_probability": l1_top1,
                "l1_top2_margin": l1_margin,
                "l1_entropy": l1_entropy,
                "l1_endpoint_expected": expected_l1,
                "l1_endpoint_correct": bool(l1_correct) if expected_l1 else None,
                "l1_knownness_score": l1_knownness_score,
                "l1_knownness_route_scores": None if head_scores is None else [float(value) for value in head_scores[0][index].tolist()],
                "knownness_source": "independent_l3_head",
                "l1_knownness_threshold": l1_knownness_threshold,
                "l1_candidate_set": l1_candidates,
                "l2_logits": l2_logits,
                "l2_probabilities": l2_probabilities,
                "l2_prediction": l2_prediction,
                "l2_top1_probability": l2_top1,
                "l2_top2_margin": l2_margin,
                "l2_top2_margin_threshold": float(calibration.get("l2_top2_margin_thresholds", {}).get(l2_prediction, 1.0)),
                "l2_entropy": l2_entropy,
                "l2_endpoint_required": bool(l2_required),
                "l2_endpoint_expected": expected_l2,
                "l2_endpoint_correct": bool(l2_correct) if expected_l2 else None,
                "l2_knownness_score": l2_knownness_score,
                "l2_knownness_route_scores": None if head_scores is None else [float(value) for value in head_scores[1][index].tolist()],
                "l2_knownness_threshold": l2_knownness_threshold,
                "l2_candidate_set": l2_candidates,
                "action": action,
                "reason_code": reason,
                "decision_details": decision.get("details", {}),
            }
        )

    def counter_payload(value: Mapping[str, Counter[str]]) -> dict[str, dict[str, int]]:
        return {
            str(label): {str(key): int(count) for key, count in sorted(counts.items())}
            for label, counts in sorted(value.items())
        }

    summary = {
        "record_count": len(records),
        "label_counts": dict(sorted(label_counts.items())),
        "action_counts": dict(sorted(action_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "by_label_action": counter_payload(by_label_action),
        "by_label_reason": counter_payload(by_label_reason),
        "endpoint_errors": dict(sorted(endpoint_errors.items())),
        "bird_unknown_separation": {
            key: {
                "count": len(values),
                "mean_l2_knownness": _mean(values),
                "min_l2_knownness": None if not values else float(min(values)),
                "max_l2_knownness": None if not values else float(max(values)),
            }
            for key, values in bird_knownness.items()
        },
        "bird_margin_separation": {
            key: {
                "count": len(values),
                "mean_l2_top2_margin": _mean(values),
                "min_l2_top2_margin": None if not values else float(min(values)),
                "max_l2_top2_margin": None if not values else float(max(values)),
            }
            for key, values in bird_margin.items()
        },
    }
    payload = {
        "schema_version": "RADAR-LN3-AUDIT-REPORT-V1",
        "record_count": len(records),
        "calibration_path": str(args.calibration_path),
        "calibration_status": calibration["status"],
        "deployment_eligible": bool(calibration["deployment_eligible"]),
        "source_manifest_sha256": evidence.source_manifest_sha256,
        "endpoint_sha256": dict(evidence.endpoint_sha256),
        "knownness_source": "independent_l3_head",
        "head_checkpoint_sha256": calibration.get("head_checkpoint_sha256"),
        "feature_cache_manifest_sha256": evidence.feature_cache_manifest_sha256,
        "role": evidence.role,
        "split_name": str(args.audit_split_name),
        "temperatures": {"l1": l1_temperature, "l2": l2_temperature},
        "conformal_floors": {"l1": l1_floors, "l2": l2_floors},
        "summary": summary,
        "records": records,
    }
    output_path = args.output_path
    if not output_path.is_absolute():
        output_path = root / output_path
    _atomic_json(output_path, payload, overwrite=bool(args.overwrite))
    return {
        "path": str(output_path),
        "record_count": len(records),
        "calibration_status": calibration["status"],
        "deployment_eligible": bool(calibration["deployment_eligible"]),
        "summary": summary,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = run(args)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"L3 audit report failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
