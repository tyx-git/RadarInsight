"""Fit and seal the exploratory L3 calibration artifact from validation data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


POLICY_PATH = Path("src/config/radar_ln3_policy.json")
DECISION_PROFILE_PATH = Path("src/config/radar_ln3_decision.json")


def _policy(path: Path) -> Any:
    from radar_ln3.decision import L3CalibrationPolicy

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("policy"), dict):
        payload = payload["policy"]
    if not isinstance(payload, dict):
        raise ValueError("policy JSON must be an object")
    return L3CalibrationPolicy(**payload)


def _router_from_head(profile: dict[str, Any]) -> bool:
    value = profile.get("l1_router_from_head", False)
    if not isinstance(value, bool):
        raise ValueError("decision profile l1_router_from_head 必须是布尔值")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit an exploratory L3 calibration")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path, default=Path("src/config/manifests.json"))
    parser.add_argument("--model-config-path", type=Path, default=Path("src/config/model.json"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--root", type=Path, default=Path("artifacts/radar_ln3"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip-data-validation",
        action="store_true",
        help="跳过数据文件 size/hash 校验（verify_files=False）；manifest 结构与事件路径边界校验仍会执行",
    )
    parser.add_argument("--skip-event-cache-validation", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    from radar_ln3.artifacts import is_default_artifact_root, l3_layout
    from radar_ln3.checkpoint import Checkpoint
    from radar_ln3.model import load_model_config
    from radar_ln3.training import fit_l3_validation_calibration

    root = args.project_root.resolve()
    layout = l3_layout(root, args.root)
    config = load_model_config(
        root, args.model_config_path, artifact_root=layout.root
    )
    register = is_default_artifact_root(layout.root)
    if not register:
        print(
            json.dumps(
                {
                    "stage": "register_skipped",
                    "root": layout.root.as_posix(),
                    "reason": "非默认 Root 的 calibration 不登记到正式 product manifest",
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    policy = _policy(root / POLICY_PATH)
    decision_profile = json.loads(
        (root / DECISION_PROFILE_PATH).read_text(encoding="utf-8")
    )
    if not isinstance(decision_profile, dict):
        raise ValueError("decision profile JSON 必须是对象")
    l1_head = Checkpoint.load(root, layout.l1_head)
    l2_head = Checkpoint.load(root, layout.l2_head)
    decision_profile["source_head_model_state_sha256"] = {
        "l1": str(l1_head.payload.get("model_state_sha256", "")),
        "l2": str(l2_head.payload.get("model_state_sha256", "")),
    }
    artifact = fit_l3_validation_calibration(
        root,
        policy=policy,
        manifest_path=args.manifest_path,
        feature_cache_manifest_path=layout.validation_feature_cache,
        device=str(args.device),
        batch_size=args.batch_size,
        output_path=layout.exploratory_calibration,
        overwrite=bool(args.overwrite),
        l1_head_path=layout.l1_head,
        l2_head_path=layout.l2_head,
        l1_head_checkpoint=l1_head,
        l2_head_checkpoint=l2_head,
        verify_files=not bool(args.skip_data_validation),
        validate_event_cache=not bool(args.skip_event_cache_validation),
        representation_root=layout.representations,
        model_config_path=args.model_config_path,
        l1_checkpoint_path=layout.l1_checkpoint,
        l2_checkpoint_path=layout.l2_checkpoint,
        config=config,
        l1_router_from_head=_router_from_head(decision_profile),
        l2_knownness_threshold_mode=str(
            decision_profile.get("l2_knownness_threshold_mode", "shared")
        ),
        decision_profile=decision_profile,
        register=register,
    )
    return {
        "path": layout.exploratory_calibration.as_posix(),
        "status": artifact["status"],
        "deployment_eligible": bool(artifact["deployment_eligible"]),
        "source_role": artifact["source_role"],
        "source_manifest_sha256": artifact["source_manifest_sha256"],
        "label_support": artifact["label_support"],
        "status_reasons": artifact["status_reasons"],
        "artifact_sha256": artifact["artifact_sha256"],
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        print(json.dumps(run(args), ensure_ascii=False, indent=2), flush=True)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"L3 calibration failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
