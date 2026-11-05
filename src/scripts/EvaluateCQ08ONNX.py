"""Run the full-cascade ONNX model through the CQ08 data adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


SRC_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_ROOT.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _sequence(adapter: Any, parent: Any) -> Any:
    from radar_ln3.data.contracts import EventSequence

    records = adapter.discover_events(parent)
    events = [adapter.read_event(record) for record in records]
    return EventSequence(
        parent_sample_id=parent.parent_sample_id,
        track_id=parent.track_id,
        label_id=parent.label_id,
        label_name=parent.label_name,
        split_name=parent.split_name,
        events=events,
        source_name="cq08",
        metadata={"adapter": "CQ08Adapter", "source_path": parent.source_path},
    )


def _build_batch(sequences: list[Any], root: Path, caches: dict[str, Any]) -> dict[str, Any]:
    from radar_ln3.features import build_model_batch
    from radar_ln3.model import load_model_config

    config = load_model_config(root)
    common = {
        "max_rd_frames": config.data.max_rd_frames_per_point,
        "max_sequence_length": config.data.max_sequence_length,
        "feature_caches": caches,
    }
    l1 = build_model_batch(
        sequences,
        task="l1",
        pad_to_max_sequence_length=True,
        **common,
    )
    l2 = build_model_batch(
        sequences,
        task="l2",
        stable_median=np.asarray(config.data.l2.stable8_median, dtype=np.float32),
        stable_scale=np.asarray(config.data.l2.stable8_scale, dtype=np.float32),
        stable_indices=config.data.l2.stable8_indices,
        clipping={
            "bounds": {
                name: {"lower": value[0], "upper": value[1]}
                for name, value in config.data.l2.tr_clipping_bounds.items()
            }
        },
        stable_input_mode="normalized",
        pad_to_max_sequence_length=True,
        **common,
    )
    names = (
        "tr_features", "tr_missing_mask", "tr_dt_seconds", "tr_dt_valid",
        "tr_dt_reason", "tr_padding_mask", "rd_frames", "rd_frame_mask",
        "rd_quality", "rd_physical_features", "rd_physical_valid",
        "micro_doppler_features", "micro_doppler_valid", "course_features",
        "course_valid", "course_support", "observation_increment_values",
        "reference_tr_features", "reference_tr_missing_mask",
        "stable_summary_tokens", "stable_summary_valid", "stable_frame_values",
        "stable_frame_mask", "motion_values",
    )
    l1_names = {"course_features", "course_valid", "course_support", "observation_increment_values"}
    return {
        name: (l1[name] if name in l1_names else l2[name])
        for name in names
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate full ONNX through CQ08Adapter")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--parent-limit", type=int, default=None)
    parser.add_argument("--parent-index", type=int, default=None)
    parser.add_argument("--split", choices=("train", "validation", "test", "all"), default="validation")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--model", type=Path, default=Path("artifacts/radar_ln3/onnx/radarln3.onnx"))
    parser.add_argument("--feature-cache-manifest", type=Path, default=Path("data/cq08/cache/features/l2/validation/feature-cache-manifest.json"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        from radar_ln3.data import CQ08Adapter
        from radar_ln3.features import load_feature_cache_bundle
        from radar_ln3.inference.onnx import ONNXCascadeRunner
        from core.evaluation.pipeline import action_matches_label

        root = args.project_root.resolve()
        if int(args.batch_size) <= 0:
            raise ValueError("batch-size 必须为正整数")
        adapter_limit = None if args.split != "all" or args.parent_index is None else int(args.parent_index) + 1
        adapter = CQ08Adapter(
            root,
            parent_limit=adapter_limit,
            parent_split=None if args.split == "all" else args.split,
        )
        parents = adapter.discover_parents()
        if args.split != "all":
            parents = [parent for parent in parents if parent.split_name == args.split]
        if args.parent_limit is not None and args.parent_index is None:
            parents = parents[: int(args.parent_limit)]
        if args.parent_index is not None:
            if args.parent_index < 0 or args.parent_index >= len(parents):
                raise ValueError("parent-index 超出 CQ08 adapter parent 范围")
            parents = [parents[int(args.parent_index)]]
        caches = load_feature_cache_bundle(
            root,
            args.feature_cache_manifest,
            verify_integrity=True,
        )
        cache_parent_ids = set(next(iter(caches.values())).parent_sample_ids)
        original_parent_count = len(parents)
        parents = [parent for parent in parents if parent.parent_sample_id in cache_parent_ids]
        skipped_parent_count = original_parent_count - len(parents)
        model_path = args.model if args.model.is_absolute() else root / args.model
        runner = ONNXCascadeRunner(model_path, prefer_directml=False)
        all_rows: list[dict[str, Any]] = []
        for start in range(0, len(parents), int(args.batch_size)):
            chunk = parents[start : start + int(args.batch_size)]
            sequences = [_sequence(adapter, parent) for parent in chunk]
            batch = _build_batch(sequences, root, caches)
            output = runner.predict(batch)
            actions = output["actions"].tolist()
            labels = [str(sequence.label_name) for sequence in sequences]
            l2_ids = np.asarray(output["l2_probabilities"]).argmax(axis=1)
            l2_names = ("DroneTarget", "BirdTarget", "BalloonTarget")
            l2_predictions = [l2_names[int(value)] for value in l2_ids]
            for index, sequence in enumerate(sequences):
                all_rows.append({
                    "parent_sample_id": sequence.parent_sample_id,
                    "label": labels[index],
                    "l2_prediction": l2_predictions[index],
                    "action": actions[index],
                    "reason": output["reasons"].tolist()[index],
                    "l1_probabilities": np.asarray(output["l1_probabilities"])[index].tolist(),
                    "l2_probabilities": np.asarray(output["l2_probabilities"])[index].tolist(),
                    "cascade_probabilities": np.asarray(output["cascade_probabilities"])[index].tolist(),
                    "stable8_valid_events": int(batch["stable_summary_valid"][index].sum().item()),
                    "rd_frames": int(batch["rd_frame_mask"][index].sum().item()),
                    "event_count": len(sequence.events),
                })
        labels = [row["label"] for row in all_rows]
        l2_predictions = [row["l2_prediction"] for row in all_rows]
        actions = [row["action"] for row in all_rows]
        pending_rows = [row for row in all_rows if row["action"] == "PENDING_ABSTAIN"]
        non_pending_rows = [row for row in all_rows if row not in pending_rows]
        stable_count = sum(row["stable8_valid_events"] for row in all_rows)
        rd_count = sum(row["rd_frames"] for row in all_rows)
        result = {
            "adapter": "CQ08Adapter",
            "split": args.split,
            "parent_count": len(all_rows),
            "source_parent_count": original_parent_count,
            "skipped_without_feature_cache": skipped_parent_count,
            "parents": [row["parent_sample_id"] for row in all_rows],
            "labels": labels,
            "l2_predictions": l2_predictions,
            "actions": actions,
            "reasons": [row["reason"] for row in all_rows],
            "l2_closed_set_accuracy": float(sum(a == b for a, b in zip(labels, l2_predictions, strict=True)) / len(labels)) if labels else None,
            "action_accuracy": float(
                sum(action_matches_label(a, b) for a, b in zip(actions, labels, strict=True))
                / len(labels)
            )
            if labels
            else None,
            "raw_action_string_accuracy": float(
                sum(a == b for a, b in zip(actions, labels, strict=True))
                / len(labels)
            )
            if labels
            else None,
            "pending_count": len(pending_rows),
            "action_coverage": len(non_pending_rows) / len(labels) if labels else None,
            "action_accuracy_excluding_pending": (
                float(
                    sum(
                        action_matches_label(row["action"], row["label"])
                        for row in non_pending_rows
                    )
                    / len(non_pending_rows)
                )
                if non_pending_rows
                else None
            ),
            "model_providers": list(runner.providers),
            "model_metadata": runner.metadata,
            "stable8_valid_event_count": stable_count,
            "rd_frame_count": rd_count,
            "records": all_rows,
        }
        text = json.dumps(result, ensure_ascii=False, indent=2)
        if args.output is not None:
            target = args.output if args.output.is_absolute() else root / args.output
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text + "\n", encoding="utf-8")
        print(text)
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"CQ08 ONNX evaluation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
