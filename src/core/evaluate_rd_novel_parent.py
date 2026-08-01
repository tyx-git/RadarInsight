from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from .metrics import compute_classification_metrics
from .rd_novel_parent_config import RDNovelParentConfig
from .rd_novel_parent_data import NovelParentSequenceDataset, collate_novel_parent_batch
from .rd_novel_parent_model import FrozenPhysicalV2NovelParentState
from .rd_novel_parent_trainer import _move_batch
from .system_metrics import compute_per_class_metrics


RD_ONLY_LABELS = ["DroneTarget", "BirdTarget"]


def _causal_payload(predictions: list[dict[str, object]]) -> dict[str, object]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for prediction in predictions:
        grouped[str(prediction["parent_sample_id"])].append(prediction)
    final_logits: list[torch.Tensor] = []
    targets: list[int] = []
    records: list[dict[str, object]] = []
    flips: list[int] = []
    stable_times: list[float] = []
    first_times: list[float] = []
    for parent_sample_id, rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda row: (float(row["window_end_seconds"]), str(row["sample_id"])))
        labels = {str(row["label_name_true"]) for row in ordered}
        if len(labels) != 1:
            raise ValueError(f"parent has inconsistent labels: {parent_sample_id}")
        label_name = next(iter(labels))
        target = RD_ONLY_LABELS.index(label_name)
        logits = torch.tensor([[row["bird_logit_drone"], row["bird_logit_bird"]] for row in ordered], dtype=torch.float32)
        prediction_indices = logits.argmax(dim=-1).tolist()
        correct = [int(index) == target for index in prediction_indices]
        suffix_correct = [False] * len(correct)
        running = True
        for position in range(len(correct) - 1, -1, -1):
            running = running and correct[position]
            suffix_correct[position] = running
        first_correct_index = next((position for position, value in enumerate(correct) if value), None)
        stable_correct_index = next((position for position, value in enumerate(suffix_correct) if value), None)
        flip_count = sum(previous != current for previous, current in zip(prediction_indices, prediction_indices[1:]))
        if first_correct_index is not None:
            first_times.append(float(ordered[first_correct_index]["window_end_seconds"]))
        if stable_correct_index is not None:
            stable_times.append(float(ordered[stable_correct_index]["window_end_seconds"]))
        final_logits.append(logits[-1])
        targets.append(target)
        flips.append(flip_count)
        records.append(
            {
                "parent_sample_id": parent_sample_id,
                "label_name_true": label_name,
                "update_count": len(ordered),
                "prefix_prediction_flip_count": flip_count,
                "prefix_predictions": [RD_ONLY_LABELS[index] for index in prediction_indices],
                "final_parent_pred": RD_ONLY_LABELS[prediction_indices[-1]],
                "final_parent_correct": bool(correct[-1]),
                "first_correct_window_end_seconds": (
                    float(ordered[first_correct_index]["window_end_seconds"])
                    if first_correct_index is not None else None
                ),
                "stable_correct_window_end_seconds": (
                    float(ordered[stable_correct_index]["window_end_seconds"])
                    if stable_correct_index is not None else None
                ),
            }
        )
    logits_tensor = torch.stack(final_logits)
    targets_tensor = torch.tensor(targets, dtype=torch.long)
    metrics = compute_classification_metrics(logits_tensor, targets_tensor, num_classes=2)
    return {
        "metrics": asdict(metrics),
        "per_class": [asdict(item) for item in compute_per_class_metrics(logits_tensor, targets_tensor, RD_ONLY_LABELS)],
        "parent_count": len(records),
        "final_correct_count": int(sum(record["final_parent_correct"] for record in records)),
        "prefix_flip_count_total": int(sum(flips)),
        "prefix_flip_count_mean": float(sum(flips) / len(flips)) if flips else 0.0,
        "first_correct_parent_count": len(first_times),
        "first_correct_window_end_seconds_mean": float(sum(first_times) / len(first_times)) if first_times else None,
        "stable_correct_parent_count": len(stable_times),
        "stable_correct_window_end_seconds_mean": float(sum(stable_times) / len(stable_times)) if stable_times else None,
        "records": records,
    }


def _trajectory_parent_predictions(trajectory_report: Path, parent_ids: set[str]) -> dict[str, dict[str, str]]:
    payload = json.loads(trajectory_report.read_text(encoding="utf-8"))
    rows = sorted(
        (row for row in payload.get("predictions", []) if str(row.get("parent_sample_id")) in parent_ids),
        key=lambda row: str(row["sample_id"]),
    )
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["parent_sample_id"])].append(row)
    predictions: dict[str, dict[str, str]] = {}
    for parent_sample_id, parent_rows in grouped.items():
        truths = {str(row["label_name_true"]) for row in parent_rows}
        if len(truths) != 1:
            raise ValueError(f"trajectory parent label conflict: {parent_sample_id}")
        votes = Counter(str(row["label_name_traj_pred"]) for row in parent_rows)
        predictions[parent_sample_id] = {
            "truth": next(iter(truths)),
            "trajectory_majority_pred": votes.most_common(1)[0][0],
        }
    if set(predictions) != parent_ids:
        raise ValueError("trajectory report does not cover all validation parents")
    return predictions


def _parent_shape_signature(project_root: Path) -> dict[str, str]:
    frame_index = pd.read_csv(project_root / "data" / "processed" / "cq08_rd_physical_canonical_frame_index_v2.csv", low_memory=False)
    return {
        str(parent_sample_id): "+".join(sorted(set(group["rd_shape"].astype(str))))
        for parent_sample_id, group in frame_index.groupby("parent_sample_id", sort=False)
    }


def _rescue_harm_payload(causal: dict[str, object], trajectory_report: Path, project_root: Path) -> dict[str, object]:
    records = list(causal["records"])
    parent_ids = {str(record["parent_sample_id"]) for record in records}
    trajectory = _trajectory_parent_predictions(trajectory_report, parent_ids)
    shapes = _parent_shape_signature(project_root)
    outcomes: Counter[str] = Counter()
    by_truth: Counter[str] = Counter()
    bird_31x360_harm = 0
    comparison_records: list[dict[str, object]] = []
    for record in records:
        parent_sample_id = str(record["parent_sample_id"])
        trajectory_record = trajectory[parent_sample_id]
        truth = str(record["label_name_true"])
        if trajectory_record["truth"] != truth:
            raise ValueError(f"trajectory/parent-state truth mismatch: {parent_sample_id}")
        trajectory_correct = trajectory_record["trajectory_majority_pred"] == truth
        state_correct = bool(record["final_parent_correct"])
        if not trajectory_correct and state_correct:
            outcome = "rescue"
        elif trajectory_correct and not state_correct:
            outcome = "harm"
        elif trajectory_correct:
            outcome = "both_correct"
        else:
            outcome = "both_wrong"
        outcomes[outcome] += 1
        by_truth[f"{truth}|{outcome}"] += 1
        shape_signature = shapes.get(parent_sample_id, "missing")
        bird_31x360_harm += int(outcome == "harm" and truth == "BirdTarget" and shape_signature == "31x360")
        comparison_records.append(
            {
                "parent_sample_id": parent_sample_id,
                "label_name_true": truth,
                "trajectory_majority_pred": trajectory_record["trajectory_majority_pred"],
                "parent_state_pred": record["final_parent_pred"],
                "shape_signature_diagnostic_only": shape_signature,
                "outcome": outcome,
            }
        )
    return {
        "parent_count": len(records),
        "outcome_counts": dict(outcomes),
        "rescue_count": int(outcomes["rescue"]),
        "harm_count": int(outcomes["harm"]),
        "net_gain": int(outcomes["rescue"] - outcomes["harm"]),
        "by_truth_outcome": dict(by_truth),
        "bird_31x360_harm_diagnostic_only": bird_31x360_harm,
        "records": comparison_records,
    }


def evaluate_rd_novel_parent_checkpoint(
    checkpoint_path: Path,
    output_path: Path,
    *,
    trajectory_report: Path,
    device_name: str,
) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config_values = dict(checkpoint["config"])
    config_values["device"] = device_name
    config_values["run_test_evaluation"] = False
    config = RDNovelParentConfig(**config_values)
    if config.run_test_evaluation:
        raise ValueError("test evaluation is forbidden")
    device = torch.device(config.device if config.device == "cpu" or torch.cuda.is_available() else "cpu")
    project_root = Path.cwd()
    dataset = NovelParentSequenceDataset(project_root, index_name=config.novel_update_index_name, split_name="val")
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=0, collate_fn=collate_novel_parent_batch)
    model = FrozenPhysicalV2NovelParentState(
        checkpoint["base_checkpoint"],
        adapter_dim=config.adapter_dim,
        state_dim=config.state_dim,
        dropout=config.dropout,
    ).to(device)
    model.load_trainable_state_dict(checkpoint["trainable_state_dict"])
    model.eval()
    predictions: list[dict[str, object]] = []
    with torch.no_grad():
        for source_batch in loader:
            batch = _move_batch(source_batch, device)
            output = model(batch)
            for batch_index, parent_sample_id in enumerate(batch.parent_sample_ids):
                parent_rows = dataset.rows_by_parent[parent_sample_id]
                final_rows = parent_rows[parent_rows["is_final_chunk"].map(lambda value: str(value).lower() in {"true", "1"})]
                final_rows = final_rows.sort_values(["update_index", "chunk_index"], kind="stable")
                final_positions = torch.nonzero(batch.is_final_chunk[batch_index] & ~batch.chunk_padding_mask[batch_index], as_tuple=False).flatten().tolist()
                if len(final_positions) != len(final_rows):
                    raise ValueError(f"final update alignment mismatch for {parent_sample_id}")
                for chunk_position, (_, row) in zip(final_positions, final_rows.iterrows()):
                    logit = output.logits[batch_index, chunk_position].detach().cpu()
                    prediction_index = int(logit.argmax().item())
                    predictions.append(
                        {
                            "sample_id": str(row["sample_id"]),
                            "parent_sample_id": parent_sample_id,
                            "label_name_true": str(row["label_name"]),
                            "label_name_parent_state_pred": RD_ONLY_LABELS[prediction_index],
                            "bird_logit_drone": float(logit[0]),
                            "bird_logit_bird": float(logit[1]),
                            "is_correct": bool(RD_ONLY_LABELS[prediction_index] == str(row["label_name"])),
                            "window_start_seconds": float(row["window_start_seconds"]),
                            "window_end_seconds": float(row["window_end_seconds"]),
                            "update_index": int(row["update_index"]),
                        }
                    )
    causal = _causal_payload(predictions)
    rescue_harm = _rescue_harm_payload(causal, trajectory_report, project_root)
    payload = {
        "evaluation_scope": "validation_only_rd_physical_v2_novel_evidence_parent_state",
        "checkpoint": checkpoint_path.as_posix(),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "base_checkpoint": checkpoint["base_checkpoint"],
        "base_checkpoint_sha256": checkpoint["base_checkpoint_sha256"],
        "device": str(device),
        "config": asdict(config),
        "causal_parent_state": causal,
        "trajectory_parent_rescue_harm": rescue_harm,
        "prediction_count": len(predictions),
        "test_evaluation_executed": False,
        "fusion_enabled": False,
        "timing_status": "track_point_anchor_diagnostic_only",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation-only evaluator for RD novel-evidence causal parent-state checkpoints.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--trajectory-report",
        default="artifacts/20260722-222612/20260722-stage009-v7-trajectory-only-baseline-v1/evaluation_report.json",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    result = evaluate_rd_novel_parent_checkpoint(
        Path(args.checkpoint),
        Path(args.output),
        trajectory_report=Path(args.trajectory_report),
        device_name=args.device,
    )
    print(
        json.dumps(
            {
                "output": args.output,
                "parent": result["causal_parent_state"]["metrics"],
                "rescue_harm": {key: value for key, value in result["trajectory_parent_rescue_harm"].items() if key != "records"},
                "test_evaluation_executed": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
