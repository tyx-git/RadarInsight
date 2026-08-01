from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch

from .metrics import compute_classification_metrics
from .system_metrics import compute_per_class_metrics


RD_ONLY_LABELS = ["DroneTarget", "BirdTarget"]


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def audit_baseline_anchor(
    project_root: Path,
    *,
    novel_update_index_name: str,
    baseline_report_path: Path,
    output_name: str,
) -> dict[str, object]:
    index = pd.read_csv(project_root / "data" / "processed" / novel_update_index_name, low_memory=False)
    validation_updates = index[
        index["split_name"].astype(str).eq("val")
        & index["is_final_chunk"].map(_as_bool)
    ].copy()
    if validation_updates["sample_id"].astype(str).duplicated().any():
        raise ValueError("novel update index has duplicate final validation sample_id")
    baseline_report = json.loads(baseline_report_path.read_text(encoding="utf-8"))
    baseline_rows = baseline_report.get("predictions", [])
    baseline_by_id = {str(row["sample_id"]): row for row in baseline_rows}
    update_ids = set(validation_updates["sample_id"].astype(str))
    baseline_ids = set(baseline_by_id)
    if update_ids != baseline_ids:
        raise ValueError(
            f"baseline/update sample mismatch: missing_baseline={len(update_ids - baseline_ids)} "
            f"unexpected_baseline={len(baseline_ids - update_ids)}"
        )
    grouped: dict[str, list[tuple[pd.Series, dict[str, object]]]] = defaultdict(list)
    for _, update_row in validation_updates.iterrows():
        sample_id = str(update_row["sample_id"])
        baseline_row = baseline_by_id[sample_id]
        for field in ("parent_sample_id", "label_name_true"):
            update_value = str(update_row["parent_sample_id"] if field == "parent_sample_id" else update_row["label_name"])
            if update_value != str(baseline_row[field]):
                raise ValueError(f"baseline/update {field} mismatch for {sample_id}")
        grouped[str(update_row["parent_sample_id"])].append((update_row, baseline_row))
    final_logits: list[torch.Tensor] = []
    targets: list[int] = []
    parent_records: list[dict[str, object]] = []
    for parent_sample_id, rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda pair: (float(pair[0]["window_end_seconds"]), str(pair[0]["sample_id"])))
        labels = {str(baseline_row["label_name_true"]) for _, baseline_row in ordered}
        if len(labels) != 1:
            raise ValueError(f"parent label conflict: {parent_sample_id}")
        label_name = next(iter(labels))
        window_logits = torch.tensor(
            [[float(row["bird_logit_drone"]), float(row["bird_logit_bird"])] for _, row in ordered],
            dtype=torch.float32,
        )
        anchor_logits = window_logits.cumsum(dim=0) / torch.arange(1, len(ordered) + 1, dtype=torch.float32).unsqueeze(1)
        final_logits.append(anchor_logits[-1])
        targets.append(RD_ONLY_LABELS.index(label_name))
        parent_records.append(
            {
                "parent_sample_id": parent_sample_id,
                "label_name_true": label_name,
                "update_count": len(ordered),
                "final_anchor_pred": RD_ONLY_LABELS[int(anchor_logits[-1].argmax())],
            }
        )
    logits_tensor = torch.stack(final_logits)
    targets_tensor = torch.tensor(targets, dtype=torch.long)
    metrics = compute_classification_metrics(logits_tensor, targets_tensor, num_classes=2)
    payload = {
        "scope": "validation-only frozen Stage 009I cumulative-logit anchor audit; no training, cache build, fusion, threshold tuning, or test evaluation",
        "inputs": {
            "novel_update_index": novel_update_index_name,
            "baseline_report": baseline_report_path.as_posix(),
        },
        "alignment": {
            "validation_final_update_count": int(len(validation_updates)),
            "baseline_prediction_count": int(len(baseline_rows)),
            "parent_count": int(len(parent_records)),
            "sample_ids_exact_match": True,
        },
        "cumulative_anchor": {
            "metrics": asdict(metrics),
            "per_class": [asdict(item) for item in compute_per_class_metrics(logits_tensor, targets_tensor, RD_ONLY_LABELS)],
            "records": parent_records,
        },
        "test_evaluation_executed": False,
        "fusion_enabled": False,
    }
    output_path = project_root / "tmp" / output_name
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Stage 009I frozen logits as a causal parent anchor.")
    parser.add_argument("--novel-update-index-name", default="expert1_cq08_rd_physical_novel_update_index_v2.csv")
    parser.add_argument("--baseline-report", default="tmp/20260727-stage009m-physical-v2-causal-parent-baseline.json")
    parser.add_argument("--output-name", default="20260728-stage009aa-baseline-anchor-audit.json")
    args = parser.parse_args()
    result = audit_baseline_anchor(
        Path.cwd(),
        novel_update_index_name=args.novel_update_index_name,
        baseline_report_path=Path(args.baseline_report),
        output_name=args.output_name,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
