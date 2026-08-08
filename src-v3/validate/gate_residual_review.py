from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from report.gate_residual_review import (
    ALLOWED_REVIEW_DECISIONS,
    REVIEW_COLUMNS,
    SCHEMA_VERSION,
    _sha256,
)


DEFAULT_PACKAGE = Path(
    "src-v3/processed/report/RADAR-LN3-V10-gate-residual-manual-review"
)


def _resolve(root: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def validate(package_dir: Path) -> dict[str, Any]:
    summary_path = package_dir / "summary.json"
    index_path = package_dir / "parent-index.csv"
    events_path = package_dir / "all-events.csv"
    review_path = package_dir / "manual-review.csv"
    html_path = package_dir / "index.html"
    readme_path = package_dir / "README.md"
    required = (summary_path, index_path, events_path, review_path, html_path, readme_path)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"manual-review package lacks required files: {missing}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("schema_version") != SCHEMA_VERSION or summary.get("status") != "passed":
        raise ValueError("manual-review summary schema/status mismatch")
    if summary.get("test") != "未执行" or summary.get("training") != "未执行":
        raise ValueError("manual-review package has an invalid execution scope")
    parents = pd.read_csv(index_path, low_memory=False)
    events = pd.read_csv(events_path, low_memory=False)
    review = pd.read_csv(review_path, keep_default_na=False, low_memory=False)
    if len(parents) != 12 or parents["parent_sample_id"].nunique() != 12:
        raise ValueError("manual-review package does not contain exactly 12 parents")
    if set(parents["current_label"].astype(str).value_counts().to_dict().items()) != {
        ("BirdTarget", 6), ("UnknownTarget", 6)
    }:
        raise ValueError("manual-review package has unexpected label counts")
    if len(events) != 232 or events["parent_sample_id"].nunique() != 12:
        raise ValueError("manual-review event coverage changed")
    if list(review.columns) != list(REVIEW_COLUMNS) or len(review) != 12:
        raise ValueError("manual-review template schema changed")
    decision_values = review["review_decision"].astype(str).str.strip()
    invalid_decisions = sorted(set(decision_values[decision_values.ne("")]) - set(ALLOWED_REVIEW_DECISIONS))
    if invalid_decisions:
        raise ValueError(f"manual-review package contains invalid decisions: {invalid_decisions}")
    decision_mask = decision_values.ne("")
    decision_count = int(decision_mask.sum())
    for column in ("reviewed_semantic_label", "reviewer", "review_date"):
        orphan_mask = review[column].astype(str).str.strip().ne("") & ~decision_mask
        if bool(orphan_mask.any()):
            raise ValueError(f"manual-review package contains {column} without a review decision")
    if decision_count:
        missing_reviewer = review["reviewer"].astype(str).str.strip().eq("") & decision_mask
        missing_date = review["review_date"].astype(str).str.strip().eq("") & decision_mask
        if bool(missing_reviewer.any()) or bool(missing_date.any()):
            raise ValueError("completed review decisions require reviewer and review_date")
        keep_mask = decision_values.eq("KEEP_CURRENT_LABEL")
        label_mismatch = (
            review["reviewed_semantic_label"].astype(str).str.strip()
            != review["current_label"].astype(str).str.strip()
        ) & keep_mask
        if bool(label_mismatch.any()):
            raise ValueError("KEEP_CURRENT_LABEL must preserve the current semantic label")
    observation_mask = (
        review["review_notes"].astype(str).str.strip().ne("")
        | review["additional_evidence_required"].astype(str).str.strip().ne("")
    )
    observation_count = int(observation_mask.sum())
    if (
        int(summary.get("review_decisions_completed", -1)) != decision_count
        or int(summary.get("automatic_label_changes", -1)) != 0
    ):
        raise ValueError("manual-review summary claims an unperformed review/label change")
    if int(summary.get("human_observations_recorded", -1)) != observation_count:
        raise ValueError("manual-review summary observation count does not match review notes")
    if set(summary.get("allowed_review_decisions", [])) != set(ALLOWED_REVIEW_DECISIONS):
        raise ValueError("manual-review decision vocabulary changed")
    all_keep_current = (
        decision_count == len(review)
        and bool(decision_values.eq("KEEP_CURRENT_LABEL").all())
    )
    expected_review_status = (
        "completed_keep_current_labels_under_user_policy"
        if all_keep_current
        else "awaiting_human_review"
    )
    expected_v11_release = (
        "ready_for_design_pending_explicit_authorization"
        if all_keep_current
        else "blocked_pending_human_review"
    )
    if summary.get("review_status") != expected_review_status:
        raise ValueError("manual-review summary review status does not match decisions")
    if summary.get("v11_release") != expected_v11_release:
        raise ValueError("manual-review V11 release state does not match decisions")
    if all_keep_current:
        policy = summary.get("decision_policy", {})
        if (
            not isinstance(policy, dict)
            or policy.get("current_labels_changed") is not False
            or policy.get("reopen_requires_multiple_independent_evidence_sources") is not True
            or policy.get("keep_current_label_does_not_independently_prove_ground_truth") is not True
        ):
            raise ValueError("completed review lacks the user-confirmed label policy metadata")

    generated_files = summary.get("generated_files", [])
    generated_by_path = {
        Path(str(item["path"])).resolve(): item
        for item in generated_files
        if isinstance(item, dict) and "path" in item
    }
    review_metadata = generated_by_path.get(review_path.resolve())
    if review_metadata is None:
        raise ValueError("manual-review summary lacks review-file integrity metadata")
    if (
        str(review_metadata.get("sha256")) != _sha256(review_path)
        or int(review_metadata.get("bytes", -1)) != review_path.stat().st_size
    ):
        raise ValueError("manual-review review-file hash/size mismatch")

    for parent in parents.itertuples(index=False):
        event_path = package_dir / str(parent.event_csv_path)
        plot_path = package_dir / str(parent.plot_path)
        rd_plot_path = package_dir / str(parent.rd_plot_path)
        parent_json_path = package_dir / str(parent.parent_json_path)
        for path in (event_path, plot_path, rd_plot_path, parent_json_path):
            if not path.exists() or path.stat().st_size <= 0:
                raise FileNotFoundError(f"manual-review parent artifact missing/empty: {path}")
        parent_events = events[events["parent_sample_id"].astype(str).eq(str(parent.parent_sample_id))]
        if len(parent_events) != int(parent.event_count):
            raise ValueError(f"manual-review event count mismatch: {parent.parent_sample_id}")
        expected = list(range(int(parent.event_count)))
        actual = pd.to_numeric(parent_events["event_index"], errors="raise").astype(int).tolist()
        if actual != expected:
            raise ValueError(f"manual-review events are not contiguous: {parent.parent_sample_id}")
    probabilities = pd.to_numeric(events["probability_cu"], errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(probabilities).all() or bool(((probabilities < 0.0) | (probabilities > 1.0)).any()):
        raise ValueError("manual-review package has invalid probabilities")
    rd_present = events["rd_present"].astype(str).str.lower().isin({"true", "1", "yes", "y"})
    if int(rd_present.sum()) != 217:
        raise ValueError("manual-review RD-present event count changed")
    if int((~rd_present).sum()) != 15:
        raise ValueError("manual-review RD-missing event count changed")

    result = {
        "status": "passed",
        "schema_version": SCHEMA_VERSION,
        "package_dir": str(package_dir),
        "parent_count": len(parents),
        "event_count": len(events),
        "rd_present_event_count": int(rd_present.sum()),
        "rd_missing_event_count": int((~rd_present).sum()),
        "review_decisions_completed": decision_count,
        "human_observations_recorded": observation_count,
        "automatic_label_changes": 0,
        "review_status": expected_review_status,
        "v11_release": expected_v11_release,
        "test": "未执行",
        "training": "未执行",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the V10 residual manual-review package.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE)
    args = parser.parse_args()
    root = args.project_root.resolve()
    validate(_resolve(root, args.package_dir))


if __name__ == "__main__":
    main()
