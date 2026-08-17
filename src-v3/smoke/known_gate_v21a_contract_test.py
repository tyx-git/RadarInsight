from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from build.known_gate_v21a import ARTIFACT, BASELINE_CONFIG, CONFIG
from radar_ln3.known_gate import (
    build_known_gate_model_for_config,
    known_gate_config_identity_sha256,
    load_known_gate_config,
)
from radar_ln3.known_gate_features import (
    INTERFRAME_SPECTRAL_FEATURE_NAMES,
    V21A_INTERFRAME_VARIANT,
    extract_interframe_spectral_temporal_consistency,
    load_known_gate_feature_cache,
)
from radar_ln3.losses import parent_final_task_ce_with_prefix_auxiliary
from train.known_gate_v21 import _verify_readiness


def _batch(evidence_dim: int) -> dict[str, torch.Tensor]:
    torch.manual_seed(21)
    batch, length, frames = 2, 4, 3
    return {
        "tr_features": torch.randn(batch, length, 14),
        "tr_missing_mask": torch.zeros(batch, length, 11),
        "tr_dt_seconds": torch.tensor(
            [[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 2.0, 1.0]], dtype=torch.float32
        ),
        "tr_dt_valid": torch.tensor(
            [[False, False, True, True], [False, False, True, True]]
        ),
        "tr_dt_reason": torch.tensor([[1, 2, 3, 3], [1, 2, 3, 3]], dtype=torch.long),
        "rd_frames": torch.randn(batch, length, frames, 2, 21, 384),
        "rd_frame_mask": torch.ones(batch, length, frames, dtype=torch.bool),
        "rd_quality": torch.rand(batch, length, 7),
        "tr_padding_mask": torch.zeros(batch, length, dtype=torch.bool),
        "known_gate_features": torch.rand(batch, length, evidence_dim),
        "known_gate_feature_valid": torch.ones(
            batch, length, evidence_dim, dtype=torch.bool
        ),
    }


def _feature_probe() -> dict[str, Any]:
    canonical = np.load(
        Path(__file__).resolve().parents[2]
        / "data/processed/cq08_rd_physical_canonical_v3_1/cq08__track__1_rd_tensor.npz",
        allow_pickle=False,
    )
    velocity = np.asarray(canonical["velocity_mps"])
    identical = np.full((3, 21, 384), -80.0, dtype=np.float32)
    identical[:, 10, 180:185] = 0.0
    stable, stable_valid = extract_interframe_spectral_temporal_consistency(
        identical, velocity
    )
    changed = identical.copy()
    changed[1:, :, :] = -80.0
    changed[1, 10, 280:285] = 0.0
    changed[2, 10, 80:85] = 0.0
    abrupt, abrupt_valid = extract_interframe_spectral_temporal_consistency(
        changed, velocity
    )
    one, one_valid = extract_interframe_spectral_temporal_consistency(
        identical[:1], velocity
    )
    if (
        not stable_valid.all()
        or not abrupt_valid.all()
        or not np.allclose(stable[:2], np.ones((2,), dtype=np.float32), atol=1.0e-6)
        or not np.allclose(stable[2:5], np.zeros((3,), dtype=np.float32), atol=1.0e-6)
        or not (0.0 < float(stable[5]) <= 1.0)
        or not (float(abrupt[0]) < float(stable[0]))
        or not (float(abrupt[1]) < float(stable[1]))
        or one_valid.any()
        or np.any(one != 0.0)
    ):
        raise AssertionError("V21a inter-frame spectral feature semantics failed")
    return {
        "stable": stable.astype(float).tolist(),
        "abrupt": abrupt.astype(float).tolist(),
        "single_frame_valid": one_valid.astype(bool).tolist(),
    }


def _temporary_treatment_config(root: Path, directory: Path) -> Path:
    baseline_payload = json.loads((root / BASELINE_CONFIG).read_text(encoding="utf-8"))
    payload = json.loads(json.dumps(baseline_payload, ensure_ascii=False))
    payload["config_name"] = "RADAR-LN3-V21a-smoke-v21a_interframe-seed43"
    payload["variant"] = V21A_INTERFRAME_VARIANT
    payload["task"]["feature_names"] = list(INTERFRAME_SPECTRAL_FEATURE_NAMES)
    payload["data"].update(
        {
            "feature_cache_manifest_path": "smoke-manifest.json",
            "expected_feature_cache_manifest_sha256": "1" * 64,
            "train_feature_cache_path": "smoke-train.npz",
            "validation_feature_cache_path": "smoke-val.npz",
            "expected_train_feature_cache_sha256": "2" * 64,
            "expected_validation_feature_cache_sha256": "3" * 64,
            "feature_experiment_contract_path": "smoke-contract.json",
            "expected_feature_experiment_contract_sha256": "4" * 64,
        }
    )
    payload["model"]["evidence_dim"] = len(INTERFRAME_SPECTRAL_FEATURE_NAMES)
    artifact = ARTIFACT.as_posix()
    payload["train"].update(
        {
            "checkpoint_path": f"{artifact}/best_model.pt",
            "last_checkpoint_path": f"{artifact}/last_model.pt",
            "formal_artifact_dir": artifact,
            "epoch_checkpoint_dir": f"{artifact}/epochs",
            "checkpoint_schema": (
                "RADAR-LN3-V21-v21a_interframe-seed43-known-gate-checkpoint-v1"
            ),
            "paired_baseline_config_path": BASELINE_CONFIG.as_posix(),
        }
    )
    path = directory / "v21a-interframe-seed43.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def run(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    feature_probe = _feature_probe()
    baseline_path = root / BASELINE_CONFIG
    if not baseline_path.is_file():
        raise FileNotFoundError("V21 seed43 Baseline config is required for V21a smoke")
    baseline_config = load_known_gate_config(baseline_path)
    with tempfile.TemporaryDirectory(prefix="v21a-config-smoke-") as temporary:
        generated_path = _temporary_treatment_config(root, Path(temporary))
        treatment_config = load_known_gate_config(generated_path)
        if (
            treatment_config.variant != V21A_INTERFRAME_VARIANT
            or treatment_config.train.seed != 43
            or baseline_config.train.seed != 43
            or treatment_config.feature_names != INTERFRAME_SPECTRAL_FEATURE_NAMES
        ):
            raise AssertionError("V21a corrected config is not the selected seed43 pair")
        treatment, baseline, evidence = build_known_gate_model_for_config(
            root, treatment_config
        )
    if (
        baseline is None
        or evidence.get("paired_baseline") is not True
        or evidence.get("new_fusion_columns_exactly_zero") is not True
        or evidence.get("seed") != 43
    ):
        raise AssertionError("V21a selected-seed paired initialization failed")
    batch = _batch(treatment_config.model.evidence_dim)
    treatment.eval()
    baseline.eval()
    with torch.inference_mode():
        treatment_logits = treatment(**batch)["sequence_logits"]
        baseline_batch = dict(batch)
        baseline_batch.pop("known_gate_features")
        baseline_batch.pop("known_gate_feature_valid")
        baseline_logits = baseline(**baseline_batch)["sequence_logits"]
    if not torch.equal(treatment_logits, baseline_logits):
        raise AssertionError("V21a zero-column initial logits differ from seed43 Baseline")

    online_state = None
    online_logits: list[torch.Tensor] = []
    with torch.inference_mode():
        for event_index in range(batch["tr_features"].shape[1]):
            step = treatment.forward_online_point(
                tr_features=batch["tr_features"][:, event_index],
                tr_missing_mask=batch["tr_missing_mask"][:, event_index],
                tr_dt_seconds=batch["tr_dt_seconds"][:, event_index],
                tr_dt_valid=batch["tr_dt_valid"][:, event_index],
                tr_dt_reason=batch["tr_dt_reason"][:, event_index],
                rd_frames=batch["rd_frames"][:, event_index],
                rd_frame_mask=batch["rd_frame_mask"][:, event_index],
                rd_quality=batch["rd_quality"][:, event_index],
                liquid_state=online_state,
                known_gate_features=batch["known_gate_features"][:, event_index],
                known_gate_feature_valid=batch["known_gate_feature_valid"][:, event_index],
            )
            online_state = step["liquid_state"]
            online_logits.append(step["window_logits"])
    if not torch.allclose(
        torch.stack(online_logits, dim=1), treatment_logits, atol=1.0e-6, rtol=1.0e-6
    ):
        raise AssertionError("V21a offline/online logits differ")

    treatment.train()
    objective = parent_final_task_ce_with_prefix_auxiliary(
        treatment(**batch)["sequence_logits"],
        torch.tensor([0, 1], dtype=torch.long),
        batch["tr_padding_mask"],
        class_weights=torch.ones(2),
        prefix_auxiliary_weight=0.1,
        prefix_auxiliary_min_events=3,
    )
    objective["loss"].backward()
    new_gradient = treatment.fusion[0].weight.grad[:, baseline.fusion_input_dim :]
    if (
        new_gradient is None
        or not torch.isfinite(new_gradient).all()
        or float(new_gradient.abs().max()) <= 0.0
    ):
        raise AssertionError("V21a evidence columns did not receive a finite non-zero gradient")
    if any(not parameter.requires_grad for parameter in treatment.parameters()):
        raise AssertionError("V21a does not expose full-parameter training")

    with tempfile.TemporaryDirectory(prefix="v21a-cache-smoke-") as temporary:
        path = Path(temporary) / "cache.npz"
        sample_ids = np.asarray(["a", "b"], dtype=np.str_)
        values = np.asarray(
            [[1.0] * len(INTERFRAME_SPECTRAL_FEATURE_NAMES), [0.5] * len(INTERFRAME_SPECTRAL_FEATURE_NAMES)],
            dtype=np.float32,
        )
        valid = np.ones_like(values, dtype=np.bool_)
        np.savez(path, sample_ids=sample_ids, values=values, valid=valid)
        loaded, loaded_valid, metadata = load_known_gate_feature_cache(
            path,
            expected_sample_ids=["a", "b"],
            variant=V21A_INTERFRAME_VARIANT,
        )
        if (
            not np.array_equal(loaded["b"], values[1])
            or not loaded_valid["a"].all()
            or metadata.get("feature_names") != list(INTERFRAME_SPECTRAL_FEATURE_NAMES)
        ):
            raise AssertionError("V21a dedicated six-column cache loader failed")

    last_checkpoint = (
        root
        / "artifacts/RADAR-LN3-V21-cq08-liquid-known-gate-baseline-seed43-formal/last_model.pt"
    )
    if not last_checkpoint.is_file():
        raise FileNotFoundError("V21 seed43 last checkpoint is unavailable for compatibility smoke")
    payload = torch.load(last_checkpoint, map_location="cpu", weights_only=False)
    if payload.get("config_identity_sha256") != known_gate_config_identity_sha256(
        baseline_config
    ):
        raise AssertionError("V21a config extension broke frozen Baseline checkpoint identity")

    old_path = (
        root
        / "config/RADAR-LN3/"
        "RADAR-LN3-V21a-cq08-liquid-known-gate-spectral-temporal-consistency-seed42-config.json"
    )
    old_config = load_known_gate_config(old_path)
    try:
        _verify_readiness(root, old_path, old_config, {"hashes": {}})
    except PermissionError as exc:
        if "superseded" not in str(exc):
            raise
    else:
        raise AssertionError("old intra-frame V21a candidate was not fail-closed")
    return {
        "status": "passed",
        "schema_version": "RADAR-LN3-V21a-interframe-spectral-contract-smoke-v1",
        "feature_probe": feature_probe,
        "selected_baseline_seed": 43,
        "same_seed_initialization": True,
        "new_fusion_columns_exactly_zero": True,
        "initial_logits_exactly_equal": True,
        "new_fusion_columns_receive_gradient": True,
        "all_model_parameters_trainable": True,
        "offline_online_logits_match": True,
        "dedicated_cache_loader": True,
        "old_candidate_training_blocked": True,
        "frozen_baseline_checkpoint_identity_preserved": True,
        "test": "未执行",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke-test corrected V21a inter-frame spectral consistency."
    )
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    args = parser.parse_args()
    print(json.dumps(run(args.project_root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
