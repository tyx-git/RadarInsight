from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import pandas as pd

from report.gate_residual_review import (
    DEFAULT_DETAIL,
    DEFAULT_INDEX,
    DEFAULT_INTEGRITY,
    DEFAULT_MANIFEST,
    REVIEW_COLUMNS,
    _resolve,
    run,
)
from validate.gate_residual_review import DEFAULT_PACKAGE, validate


DEFAULT_OUTPUT = Path("src-v3/smoke/RADAR-LN3-V10-gate-residual-review-contract-result.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Focused contract test for the V10 residual review package.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    root = args.project_root.resolve()
    validation = validate(_resolve(root, DEFAULT_PACKAGE))
    if int(validation.get("human_observations_recorded", -1)) != 12:
        raise AssertionError("expected all 12 non-decisional user observations to be recorded")
    if int(validation.get("review_decisions_completed", -1)) != 12:
        raise AssertionError("expected all 12 user-confirmed review decisions")
    if validation.get("v11_release") != "ready_for_design_pending_explicit_authorization":
        raise AssertionError("completed review must still require explicit V11 authorization")

    smoke_root = root / "src-v3" / "smoke"
    smoke_root.mkdir(parents=True, exist_ok=True)
    guard_passed = False
    with tempfile.TemporaryDirectory(prefix="v10-review-guard-", dir=smoke_root) as temporary:
        output_dir = Path(temporary)
        row = {name: "" for name in REVIEW_COLUMNS}
        row.update({
            "parent_sample_id": "cq08|track|333",
            "track_id": 333,
            "current_label": "BirdTarget",
            "review_decision": "KEEP_CURRENT_LABEL",
            "reviewer": "guard-test",
        })
        pd.DataFrame([row], columns=REVIEW_COLUMNS).to_csv(
            output_dir / "manual-review.csv", index=False, encoding="utf-8-sig"
        )
        try:
            run(
                root,
                _resolve(root, DEFAULT_INDEX),
                _resolve(root, DEFAULT_DETAIL),
                _resolve(root, DEFAULT_INTEGRITY),
                _resolve(root, DEFAULT_MANIFEST),
                output_dir,
            )
        except FileExistsError as error:
            guard_passed = "refusing to overwrite" in str(error)
        if not guard_passed:
            raise AssertionError("manual review overwrite guard did not reject human input")

    invalid_decision_guard = False
    with tempfile.TemporaryDirectory(prefix="v10-review-invalid-", dir=smoke_root) as temporary:
        package_dir = Path(temporary) / "package"
        shutil.copytree(_resolve(root, DEFAULT_PACKAGE), package_dir)
        review_path = package_dir / "manual-review.csv"
        review = pd.read_csv(review_path, keep_default_na=False, low_memory=False)
        review.loc[0, "review_decision"] = "INVALID_DECISION"
        review.to_csv(review_path, index=False, encoding="utf-8-sig")
        try:
            validate(package_dir)
        except ValueError as error:
            invalid_decision_guard = "invalid decisions" in str(error)
        if not invalid_decision_guard:
            raise AssertionError("manual review validator accepted an invalid decision")

    result = {
        "status": "passed",
        "schema_version": "RADAR-LN3-V10-gate-residual-review-contract-test-v1",
        "package_validation": validation,
        "human_input_overwrite_guard": "passed",
        "invalid_decision_guard": "passed",
        "human_observations_recorded": 12,
        "review_decisions_completed": 12,
        "automatic_label_changes": 0,
        "v11_release": "ready_for_design_pending_explicit_authorization",
        "test": "未执行",
        "training": "未执行",
    }
    output_path = _resolve(root, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
