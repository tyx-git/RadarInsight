"""Compare ASD sidecar inference with direct CQ08 adapter inference."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any
import zipfile

import numpy as np


SRC_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_ROOT.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from radar_ln3.inference.onnx.eval_support import sequence_from_adapter  # noqa: E402
from radar_ln3.model.io import FULL_CASCADE_INPUTS, L1_SOURCE_ONLY_INPUTS  # noqa: E402


MODEL_INPUT_NAMES = FULL_CASCADE_INPUTS
L1_FEATURE_NAMES = set(L1_SOURCE_ONLY_INPUTS)
L2_NAMES = ("DroneTarget", "BirdTarget", "BalloonTarget")


def _asd_parent_ids(asd_dir: Path) -> list[str]:
    result: list[str] = []
    for path in sorted(asd_dir.glob("*.asd")):
        with zipfile.ZipFile(path) as archive:
            manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
        parent_id = str(manifest.get("parent_sample_id") or "")
        if not parent_id:
            raise ValueError(f"ASD 缺少 parent_sample_id: {path}")
        result.append(parent_id)
    if len(result) != len(set(result)):
        raise ValueError("ASD 样本存在重复 parent_sample_id")
    return result


def _build_batch(
    sequences: list[Any],
    root: Path,
    raw_provider: Any,
) -> dict[str, Any]:
    from radar_ln3.features import build_model_batch
    from radar_ln3.model import load_model_config

    config = load_model_config(root)
    common = {
        "max_rd_frames": config.data.max_rd_frames_per_point,
        "max_sequence_length": config.data.max_sequence_length,
        "pad_to_max_sequence_length": True,
        "allow_unmapped_labels": True,
    }
    l1 = build_model_batch(sequences, task="l1", **common)
    l2 = build_model_batch(
        sequences,
        task="l2",
        raw_echo_provider=raw_provider,
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
        **common,
    )
    return {
        name: (l1[name] if name in L1_FEATURE_NAMES else l2[name])
        for name in MODEL_INPUT_NAMES
    }


def _run_cq08(
    root: Path,
    parent_ids: list[str],
    model_path: Path,
    *,
    batch_size: int,
) -> dict[str, Any]:
    from radar_ln3.data.cq08 import CQ08Adapter, CQ08RawEchoProvider
    from radar_ln3.inference.onnx import ONNXCascadeRunner
    from core.evaluation.pipeline import action_matches_label

    adapter = CQ08Adapter(root, verify_files=False)
    parents = {
        str(parent.parent_sample_id): parent
        for parent in adapter.discover_parents()
    }
    missing = [parent_id for parent_id in parent_ids if parent_id not in parents]
    if missing:
        raise ValueError(f"CQ08 缺少指定 parent: {missing}")
    raw_provider = CQ08RawEchoProvider(
        root,
        source_root=adapter.source_root,
        cache_size=2,
    )
    runner = ONNXCascadeRunner(model_path, prefer_directml=False)
    rows: list[dict[str, Any]] = []
    for start in range(0, len(parent_ids), int(batch_size)):
        chunk_ids = parent_ids[start : start + int(batch_size)]
        sequences = [sequence_from_adapter(adapter, parents[parent_id]) for parent_id in chunk_ids]
        batch = _build_batch(sequences, root, raw_provider)
        output = runner.predict(batch)
        l2_probabilities = np.asarray(output["l2_probabilities"])
        l2_ids = l2_probabilities.argmax(axis=1)
        actions = output["actions"].tolist()
        reasons = output["reasons"].tolist()
        for index, sequence in enumerate(sequences):
            rows.append(
                {
                    "parent_sample_id": sequence.parent_sample_id,
                    "label": str(sequence.label_name),
                    "l2_prediction": L2_NAMES[int(l2_ids[index])],
                    "action": actions[index],
                    "reason": reasons[index],
                    "l1_probabilities": np.asarray(
                        output["l1_probabilities"]
                    )[index].tolist(),
                    "l2_probabilities": l2_probabilities[index].tolist(),
                    "cascade_probabilities": np.asarray(
                        output["cascade_probabilities"]
                    )[index].tolist(),
                    "stable8_valid_events": int(
                        batch["stable_summary_valid"][index].sum().item()
                    ),
                    "micro_doppler_valid_events": int(
                        batch["micro_doppler_valid"][index].sum().item()
                    ),
                    "rd_frames": int(batch["rd_frame_mask"][index].sum().item()),
                    "event_count": len(sequence.events),
                    "sequence_truncated": bool(
                        len(sequence.events) > 29
                    ),
                }
            )
    labels = [row["label"] for row in rows]
    l2_predictions = [row["l2_prediction"] for row in rows]
    actions = [row["action"] for row in rows]
    known_rows = [
        row for row in rows if row["label"] in set(L2_NAMES)
    ]
    pending_rows = [row for row in rows if row["action"] == "PENDING_ABSTAIN"]
    non_pending_rows = [row for row in rows if row not in pending_rows]
    return {
        "adapter": "CQ08Adapter",
        "parent_count": len(rows),
        "parents": parent_ids,
        "labels": labels,
        "l2_predictions": l2_predictions,
        "actions": actions,
        "reasons": [row["reason"] for row in rows],
        "l2_closed_set_accuracy": float(
            sum(row["label"] == row["l2_prediction"] for row in known_rows)
            / len(known_rows)
        )
        if known_rows
        else None,
        "l2_all_parent_label_match": float(
            sum(a == b for a, b in zip(labels, l2_predictions, strict=True))
            / len(labels)
        )
        if labels
        else None,
        "action_accuracy": float(
            sum(action_matches_label(a, b) for a, b in zip(actions, labels, strict=True))
            / len(labels)
        )
        if labels
        else None,
        "raw_action_string_accuracy": float(
            sum(a == b for a, b in zip(actions, labels, strict=True)) / len(labels)
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
        "stable8_valid_event_count": sum(
            row["stable8_valid_events"] for row in rows
        ),
        "micro_doppler_valid_event_count": sum(
            row["micro_doppler_valid_events"] for row in rows
        ),
        "rd_frame_count": sum(row["rd_frames"] for row in rows),
        "records": rows,
    }


def _compare(asd_report: dict[str, Any], cq08_report: dict[str, Any]) -> dict[str, Any]:
    asd_rows = {
        str(row["parent_sample_id"]): row for row in asd_report["records"]
    }
    cq_rows = {
        str(row["parent_sample_id"]): row for row in cq08_report["records"]
    }
    parent_ids = list(cq08_report["parents"])
    if set(parent_ids) != set(asd_rows) or set(parent_ids) != set(cq_rows):
        raise ValueError("ASD/CQ08 parent 集合不一致")
    rows: list[dict[str, Any]] = []
    for parent_id in parent_ids:
        asd = asd_rows[parent_id]
        cq = cq_rows[parent_id]
        asd_l2 = L2_NAMES[
            int(np.argmax(np.asarray(asd["l2_probabilities"], dtype=np.float32)))
        ]
        asd_l2_prob = np.asarray(asd["l2_probabilities"], dtype=np.float32)
        cq_l2_prob = np.asarray(cq["l2_probabilities"], dtype=np.float32)
        rows.append(
            {
                "parent_sample_id": parent_id,
                "label": asd["label_name"],
                "asd_l2_prediction": asd_l2,
                "cq08_l2_prediction": cq["l2_prediction"],
                "asd_action": asd["action"],
                "cq08_action": cq["action"],
                "asd_reason": asd["reason_code"],
                "cq08_reason": cq["reason"],
                "l2_prediction_equal": asd_l2 == cq["l2_prediction"],
                "action_equal": asd["action"] == cq["action"],
                "l2_probability_max_abs_diff": float(
                    np.max(np.abs(asd_l2_prob - cq_l2_prob))
                ),
                "asd_stable8_available": bool(asd["stable8_available"]),
                "cq08_stable8_valid_events": cq["stable8_valid_events"],
                "cq08_micro_doppler_valid_events": cq[
                    "micro_doppler_valid_events"
                ],
            }
        )
    return {
        "parent_count": len(rows),
        "asd": {
            "l2_closed_set_accuracy": asd_report["l2_closed_set_top1_accuracy"],
            "action_accuracy": asd_report["labeled_action_accuracy"],
            "raw_action_string_accuracy": asd_report[
                "raw_action_string_accuracy"
            ],
            "pending_count": asd_report["pending_count"],
            "action_coverage": asd_report["action_coverage"],
            "action_accuracy_excluding_pending": asd_report[
                "labeled_action_accuracy_excluding_pending"
            ],
            "stable8_available_count": asd_report["stable8_available_count"],
            "micro_doppler_available_count": asd_report[
                "micro_doppler_available_count"
            ],
        },
        "cq08": {
            "l2_closed_set_accuracy": cq08_report["l2_closed_set_accuracy"],
            "action_accuracy": cq08_report["action_accuracy"],
            "raw_action_string_accuracy": cq08_report[
                "raw_action_string_accuracy"
            ],
            "pending_count": cq08_report["pending_count"],
            "action_coverage": cq08_report["action_coverage"],
            "action_accuracy_excluding_pending": cq08_report[
                "action_accuracy_excluding_pending"
            ],
            "stable8_valid_event_count": cq08_report[
                "stable8_valid_event_count"
            ],
            "micro_doppler_valid_event_count": cq08_report[
                "micro_doppler_valid_event_count"
            ],
        },
        "delta_cq08_minus_asd": {
            "l2_closed_set_accuracy": cq08_report["l2_closed_set_accuracy"]
            - asd_report["l2_closed_set_top1_accuracy"],
            "action_accuracy": cq08_report["action_accuracy"]
            - asd_report["labeled_action_accuracy"],
            "action_accuracy_excluding_pending": cq08_report[
                "action_accuracy_excluding_pending"
            ]
            - asd_report["labeled_action_accuracy_excluding_pending"],
        },
        "action_equal_count": sum(row["action_equal"] for row in rows),
        "l2_prediction_equal_count": sum(
            row["l2_prediction_equal"] for row in rows
        ),
        "max_l2_probability_abs_diff": max(
            row["l2_probability_max_abs_diff"] for row in rows
        ),
        "mean_l2_probability_abs_diff": float(
            np.mean([row["l2_probability_max_abs_diff"] for row in rows])
        ),
        "records": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare 20 ASD packages with direct CQ08Adapter ONNX inference"
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--asd-dir", type=Path, default=Path("data/asd"))
    parser.add_argument(
        "--asd-report",
        type=Path,
        default=Path("reports/asd_20_onnx_evaluation_with_evidence.json"),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("artifacts/radar_ln3/onnx/radarln3.onnx"),
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/cq08_vs_asd_20_onnx_comparison.json"),
    )
    args = parser.parse_args(argv)
    try:
        root = args.project_root.resolve()
        asd_dir = args.asd_dir if args.asd_dir.is_absolute() else root / args.asd_dir
        asd_report_path = (
            args.asd_report
            if args.asd_report.is_absolute()
            else root / args.asd_report
        )
        model_path = args.model if args.model.is_absolute() else root / args.model
        asd_report = json.loads(asd_report_path.read_text(encoding="utf-8"))
        parent_ids = _asd_parent_ids(asd_dir)
        if len(parent_ids) != 20:
            raise ValueError(f"需要恰好 20 个 ASD 样本，实际为 {len(parent_ids)}")
        cq08_report = _run_cq08(
            root,
            parent_ids,
            model_path,
            batch_size=int(args.batch_size),
        )
        comparison = _compare(asd_report, cq08_report)
        result = {"cq08": cq08_report, "comparison": comparison}
        text = json.dumps(result, ensure_ascii=False, indent=2)
        target = args.output if args.output.is_absolute() else root / args.output
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text + "\n", encoding="utf-8")
        print(text)
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        print(
            f"CQ08/ASD comparison failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
