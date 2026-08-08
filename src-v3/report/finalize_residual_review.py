from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_PACKAGE = Path(
    "src-v3/processed/report/RADAR-LN3-V10-gate-residual-manual-review"
)
DECISION = "KEEP_CURRENT_LABEL"
REVIEW_STATUS = "completed_keep_current_labels_under_user_policy"
V11_RELEASE = "ready_for_design_pending_explicit_authorization"

POLICY_NOTE = (
    "正式裁决（2026-08-08）：原始采集记录已丢失，当前按项目规则保留现标签。"
    "UnknownTarget包含风筝、昆虫等异质目标，运动强度不能证明其属于DBB；Bird可能静止或低速，"
    "低峰值速度不能否定Bird。模型路由、置信度和单一TR/RD指标均不是改标证据。"
    "本裁决表示当前证据不足以推翻标签，不等于独立证明标签客观正确。"
)
EVIDENCE_POLICY = (
    "仅当源标签记录、TR/RD配对或物理证据、独立人工背景、独立模型等多方证据共同冲突时，"
    "才向项目用户提交重新裁决或改标申请；不得自动改标。"
)


def _resolve(root: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _append_once(current: str, addition: str) -> str:
    current = str(current).strip()
    if addition in current:
        return current
    return f"{current} {addition}".strip()


def finalize(package_dir: Path, reviewer: str, review_date: str) -> dict[str, Any]:
    review_path = package_dir / "manual-review.csv"
    summary_path = package_dir / "summary.json"
    readme_path = package_dir / "README.md"
    for path in (review_path, summary_path, readme_path):
        if not path.exists():
            raise FileNotFoundError(path)

    review = pd.read_csv(review_path, keep_default_na=False, low_memory=False)
    if len(review) != 12 or review["parent_sample_id"].nunique() != 12:
        raise ValueError("expected exactly 12 distinct residual parents")
    existing = review["review_decision"].astype(str).str.strip()
    if bool((existing.ne("") & existing.ne(DECISION)).any()):
        conflicts = review.loc[
            existing.ne("") & existing.ne(DECISION),
            ["parent_sample_id", "review_decision"],
        ].to_dict(orient="records")
        raise ValueError(f"refusing to overwrite conflicting review decisions: {conflicts}")

    review["review_decision"] = DECISION
    review["reviewed_semantic_label"] = review["current_label"].astype(str)
    review["reviewer"] = reviewer
    review["review_date"] = review_date
    review["review_notes"] = review["review_notes"].map(
        lambda value: _append_once(value, POLICY_NOTE)
    )
    review["additional_evidence_required"] = EVIDENCE_POLICY
    track_333 = review["track_id"].astype(int).eq(333)
    review.loc[track_333, "additional_evidence_required"] = review.loc[
        track_333, "additional_evidence_required"
    ].map(
        lambda value: _append_once(
            value,
            "Track 333的RD覆盖完整性单独审计；RD缺失本身不改变BirdTarget标签。",
        )
    )
    review.to_csv(review_path, index=False, encoding="utf-8-sig")

    readme = readme_path.read_text(encoding="utf-8")
    completion = (
        "\n## 2026-08-08裁决状态\n\n"
        "12条记录均依据项目用户确认的标签政策填写为`KEEP_CURRENT_LABEL`。这里的含义是："
        "当前证据不足以推翻现标签，并非独立证明标签客观正确。Unknown异质性、静止/低速Bird、"
        "模型置信度或单一TR/RD异常均不能单独触发改标。任何改标仍需多方独立证据并另行申请。\n"
    )
    if "## 2026-08-08裁决状态" not in readme:
        readme_path.write_text(readme.rstrip() + "\n" + completion, encoding="utf-8")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["review_status"] = REVIEW_STATUS
    summary["review_decisions_completed"] = 12
    summary["human_observations_recorded"] = 12
    summary["automatic_label_changes"] = 0
    summary["decision_counts"] = {DECISION: 12}
    summary["decision_policy"] = {
        "confirmed_by": reviewer,
        "confirmed_on": review_date,
        "current_labels_changed": False,
        "unknown_is_heterogeneous": True,
        "motion_strength_is_not_dbb_identity_evidence": True,
        "low_speed_does_not_invalidate_bird": True,
        "reopen_requires_multiple_independent_evidence_sources": True,
        "keep_current_label_does_not_independently_prove_ground_truth": True,
    }
    summary["v11_release"] = V11_RELEASE
    summary["interpretation_limits"] = [
        "all 12 review decisions keep the current label under the user-confirmed project policy",
        "KEEP_CURRENT_LABEL means current evidence does not justify relabeling; it is not independent proof of ground truth",
        "model predictions, confidence, motion strength and any single TR/RD metric are not label evidence",
        "UnknownTarget is heterogeneous and remains distinct from a calibrated reject/abstain action",
        "no data label value, data index, config or checkpoint was changed",
        "V11 design or implementation still requires explicit user authorization",
        "test, threshold fitting, training, stable, formal cascade and P95 were not executed",
    ]
    generated = []
    for item in summary.get("generated_files", []):
        path = Path(str(item["path"])).resolve()
        if not path.exists():
            raise FileNotFoundError(f"recorded review artifact is missing: {path}")
        generated.append(
            {"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size}
        )
    summary["generated_files"] = generated
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    result = {
        "status": "passed",
        "package_dir": str(package_dir),
        "review_decisions_completed": 12,
        "decision_counts": {DECISION: 12},
        "current_labels_changed": False,
        "reviewer": reviewer,
        "review_date": review_date,
        "manual_review_sha256": _sha256(review_path),
        "summary_sha256": _sha256(summary_path),
        "v11_release": V11_RELEASE,
        "training": "未执行",
        "test": "未执行",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Finalize all 12 V10 residual reviews under the user-confirmed label policy."
    )
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--reviewer", default="project_owner_user")
    parser.add_argument("--review-date", default="2026-08-08")
    args = parser.parse_args()
    root = args.project_root.resolve()
    finalize(_resolve(root, args.package_dir), str(args.reviewer), str(args.review_date))


if __name__ == "__main__":
    main()
