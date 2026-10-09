"""Explicit V28-v2 command line stages."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from .calibration import finalize_audit, fit_calibration
from .features import prepare_classifier_features, prepare_open_set_features
from .head_training import train_heads
from .protocol import V28_L1_TASK, V28_L2_TASK, load_context, prepare_manifest
from .registry import build_registry, write_registry
from .representations import materialize_representations
from .smoke import run_smoke
from .evaluation import evaluate_checkpoint
from .replay import run_replay
from .training import train_task


def _binding(root: Path, path_text: str, *, task: str, seed: int) -> dict[str, object]:
    path = Path(path_text)
    resolved = path if path.is_absolute() else root / path
    resolved = resolved.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"V28 endpoint path escaped project root: {path_text}") from exc
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    payload = torch.load(resolved, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("task") != task or int(payload.get("seed", -1)) != int(seed):
        raise ValueError(f"V28 endpoint seed differs: {resolved}")
    trained_manifest = payload.get("manifest_sha256") or payload.get("system_manifest_sha256")
    return {"path": resolved.relative_to(root.resolve()).as_posix(), "task": task, "seed": int(seed), "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(), "trained_manifest_sha256": None if trained_manifest is None else str(trained_manifest)}


def _device(context):
    requested = str(context.config.training["device"]).lower()
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("V28 requires CUDA but CUDA is unavailable")
        return torch.device("cuda:0")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError(f"unsupported V28 device: {requested}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RadarInsight V28-v2")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare-manifest")
    sub.add_parser("prepare-classifier-features")
    smoke = sub.add_parser("smoke")
    train = sub.add_parser("train")
    train.add_argument("--task", choices=(V28_L1_TASK, V28_L2_TASK), required=True)
    train.add_argument("--seed", type=int, required=True)
    train.add_argument("--epochs", type=int)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--task", choices=(V28_L1_TASK, V28_L2_TASK), required=True)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--seed", type=int, required=True)
    evaluate.add_argument("--split", choices=("train", "validation"), default="validation")
    open_set = sub.add_parser("prepare-open-set")
    open_set.add_argument("--l1-checkpoint", required=True)
    open_set.add_argument("--l2-checkpoint", required=True)
    open_set.add_argument("--l1-seed", type=int, required=True)
    open_set.add_argument("--l2-seed", type=int, required=True)
    rep = sub.add_parser("materialize-representations")
    rep.add_argument("--l1-checkpoint", required=True)
    rep.add_argument("--l2-checkpoint", required=True)
    rep.add_argument("--l1-seed", type=int, required=True)
    rep.add_argument("--l2-seed", type=int, required=True)
    head = sub.add_parser("train-heads")
    head.add_argument("--task", choices=(V28_L1_TASK, V28_L2_TASK), required=True)
    head.add_argument("--seed", type=int, required=True)
    cal = sub.add_parser("calibrate")
    cal.add_argument("--l1-head", required=True)
    cal.add_argument("--l2-head", required=True)
    cal.add_argument("--l1-checkpoint", required=True)
    cal.add_argument("--l2-checkpoint", required=True)
    cal.add_argument("--l1-seed", type=int, required=True)
    cal.add_argument("--l2-seed", type=int, required=True)
    cal.add_argument("--l1-head-seed", type=int, required=True)
    cal.add_argument("--l2-head-seed", type=int, required=True)
    reg = sub.add_parser("build-registry")
    reg.add_argument("--calibration-json", required=True)
    reg.add_argument("--l1-checkpoint", required=True)
    reg.add_argument("--l2-checkpoint", required=True)
    reg.add_argument("--l1-head", required=True)
    reg.add_argument("--l2-head", required=True)
    reg.add_argument("--l1-seed", type=int, required=True)
    reg.add_argument("--l2-seed", type=int, required=True)
    reg.add_argument("--l1-head-seed", type=int, required=True)
    reg.add_argument("--l2-head-seed", type=int, required=True)
    sub.add_parser("replay")
    sub.add_parser("audit")
    sub.add_parser("finalize-audit")
    args = parser.parse_args(argv)
    root = args.project_root.resolve()
    if args.command == "prepare-manifest":
        result = prepare_manifest(root, args.config)
    elif args.command == "prepare-classifier-features":
        result = prepare_classifier_features(root)
    elif args.command == "smoke":
        result = run_smoke(root, args.config)
    elif args.command == "train":
        result = train_task(load_context(root, args.config), task=args.task, seed=args.seed, epochs=args.epochs)
    elif args.command == "evaluate":
        context = load_context(root, args.config)
        result = evaluate_checkpoint(context, task=args.task, checkpoint=args.checkpoint, seed=args.seed, split=args.split, device=_device(context))
    elif args.command == "prepare-open-set":
        context = load_context(root, args.config)
        binding = {V28_L1_TASK: _binding(root, args.l1_checkpoint, task=V28_L1_TASK, seed=args.l1_seed), V28_L2_TASK: _binding(root, args.l2_checkpoint, task=V28_L2_TASK, seed=args.l2_seed)}
        result = prepare_open_set_features(root, endpoint_binding=binding)
    elif args.command == "materialize-representations":
        context = load_context(root, args.config)
        binding = {V28_L1_TASK: _binding(root, args.l1_checkpoint, task=V28_L1_TASK, seed=args.l1_seed), V28_L2_TASK: _binding(root, args.l2_checkpoint, task=V28_L2_TASK, seed=args.l2_seed)}
        result = materialize_representations(context, endpoint_binding=binding, device=_device(context))
    elif args.command == "train-heads":
        result = train_heads(load_context(root, args.config), task=args.task, seed=args.seed, device=_device(load_context(root, args.config)))
    elif args.command == "calibrate":
        context = load_context(root, args.config)
        result = fit_calibration(context, endpoint_binding={V28_L1_TASK: _binding(root, args.l1_checkpoint, task=V28_L1_TASK, seed=args.l1_seed), V28_L2_TASK: _binding(root, args.l2_checkpoint, task=V28_L2_TASK, seed=args.l2_seed)}, head_bindings={V28_L1_TASK: _binding(root, args.l1_head, task=V28_L1_TASK, seed=args.l1_head_seed), V28_L2_TASK: _binding(root, args.l2_head, task=V28_L2_TASK, seed=args.l2_head_seed)})
        target = root / context.config.artifacts["calibration_report_path"]
        from .protocol import write_sealed_json
        write_sealed_json(target, result)
        result["path"] = str(target.relative_to(root).as_posix())
    elif args.command == "build-registry":
        context = load_context(root, args.config)
        from .protocol import read_sealed_json
        calibration_path = Path(args.calibration_json).resolve()
        try:
            calibration_path.relative_to(root)
        except ValueError as exc:
            raise ValueError("V28 calibration report escaped project root") from exc
        calibration_payload = read_sealed_json(calibration_path)
        result = build_registry(context, endpoint_binding={V28_L1_TASK: _binding(root, args.l1_checkpoint, task=V28_L1_TASK, seed=args.l1_seed), V28_L2_TASK: _binding(root, args.l2_checkpoint, task=V28_L2_TASK, seed=args.l2_seed)}, head_bindings={V28_L1_TASK: _binding(root, args.l1_head, task=V28_L1_TASK, seed=args.l1_head_seed), V28_L2_TASK: _binding(root, args.l2_head, task=V28_L2_TASK, seed=args.l2_head_seed)}, calibration=calibration_payload, pool_registry_path=context.config.artifacts["pool_registry_path"], feature_manifest_path=context.config.artifacts["feature_manifest_path"], representation_manifest_path=context.config.artifacts["representation_manifest_path"], calibration_report_path=calibration_path)
        write_registry(context, result)
    elif args.command == "replay":
        context = load_context(root, args.config)
        result = run_replay(context, device=_device(context))
    elif args.command == "audit":
        context = load_context(root, args.config)
        result = run_replay(context, device=_device(context), audit_mode=True)
    elif args.command == "finalize-audit":
        context = load_context(root, args.config)
        from .registry import load_registry
        result = finalize_audit(context, registry=load_registry(context))
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
