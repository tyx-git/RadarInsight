from __future__ import annotations

import json
from pathlib import Path

import pytest

from config.configmanager import load_expert1_config, load_pretrain_track_config


def _write_config(path: Path, expert1: dict[str, object]) -> None:
    path.write_text(
        json.dumps(
            {
                "project": {
                    "data_root": "shared-data",
                    "device": "cpu",
                    "num_workers": 1,
                    "seed": 7,
                },
                "expert1": expert1,
            }
        ),
        encoding="utf-8",
    )


def test_expert_section_values_override_project_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    _write_config(
        config_path,
        {
            "data_root": "expert1-data",
            "device": "cuda:0",
            "num_workers": 3,
            "seed": 11,
        },
    )

    config = load_expert1_config(config_path)

    assert config.data_root == "expert1-data"
    assert config.device == "cuda:0"
    assert config.num_workers == 3
    assert config.seed == 11


def test_project_values_fill_missing_expert_section_values(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    _write_config(config_path, {})

    config = load_expert1_config(config_path)

    assert config.data_root == "shared-data"
    assert config.device == "cpu"
    assert config.num_workers == 1
    assert config.seed == 7


def test_expert1_unknown_field_fails_fast(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    _write_config(config_path, {"epochs": 18, "typo_parameter_should_fail": 123})

    try:
        load_expert1_config(config_path)
    except ValueError as exc:
        assert "typo_parameter_should_fail" in str(exc)
    else:
        raise AssertionError("unknown Expert1 configuration field must fail fast")


def test_pure_tr_only_config_rejects_stage2_checkpoint_preference(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    _write_config(
        config_path,
        {
            "epochs": 18,
            "stage1_traj_only_epochs": 18,
            "prefer_stage2_checkpoint": True,
        },
    )

    try:
        load_expert1_config(config_path)
    except ValueError as exc:
        assert "prefer_stage2_checkpoint=false" in str(exc)
    else:
        raise AssertionError("pure TR-only configuration must reject Stage-2 preference")


def test_pure_tr_only_config_is_explicit_and_loads_when_stage2_preference_disabled(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    _write_config(
        config_path,
        {
            "epochs": 18,
            "stage1_traj_only_epochs": 18,
            "prefer_stage2_checkpoint": False,
        },
    )

    config = load_expert1_config(config_path)

    assert config.is_pure_tr_only


def test_expert1_rejects_unknown_trajectory_feature_schema(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    _write_config(
        config_path,
        {
            "epochs": 18,
            "stage1_traj_only_epochs": 18,
            "prefer_stage2_checkpoint": False,
            "trajectory_feature_schema": "v999",
        },
    )

    try:
        load_expert1_config(config_path)
    except ValueError as exc:
        assert "unsupported trajectory feature schema" in str(exc)
    else:
        raise AssertionError("unsupported trajectory feature schema must fail fast")


def test_pretrain_unknown_field_fails_fast(tmp_path: Path) -> None:
    config_path = tmp_path / "pretrain.json"
    config_path.write_text(
        json.dumps({"project": {"data_root": "data"}, "pretrain_track": {"typo_pretrain_field": 123}}),
        encoding="utf-8",
    )

    try:
        load_pretrain_track_config(config_path)
    except ValueError as exc:
        assert "typo_pretrain_field" in str(exc)
    else:
        raise AssertionError("unknown pretrain configuration field must fail fast")


def test_pretrain_temporal_contract_rejects_unknown_modes(tmp_path: Path) -> None:
    config_path = tmp_path / "pretrain-temporal.json"
    config_path.write_text(
        json.dumps(
            {
                "project": {"data_root": "data"},
                "pretrain_track": {"temporal_encoding_mode": "future_aware"},
            }
        ),
        encoding="utf-8",
    )

    try:
        load_pretrain_track_config(config_path)
    except ValueError as exc:
        assert "temporal encoding mode" in str(exc)
    else:
        raise AssertionError("unknown pretrain temporal encoding must fail fast")


def test_pretrain_source_statistics_weighting_rejects_unknown_mode(tmp_path: Path) -> None:
    config_path = tmp_path / "pretrain-statistics.json"
    config_path.write_text(
        json.dumps(
            {
                "project": {"data_root": "data"},
                "pretrain_track": {"source_statistics_weighting": "window_magic"},
            }
        ),
        encoding="utf-8",
    )

    try:
        load_pretrain_track_config(config_path)
    except ValueError as exc:
        assert "source_statistics_weighting" in str(exc)
    else:
        raise AssertionError("unknown pretrain source statistics weighting must fail fast")


def test_pretrain_parent_sampling_rejects_unknown_mode(tmp_path: Path) -> None:
    config_path = tmp_path / "pretrain-parent-sampling.json"
    config_path.write_text(
        json.dumps(
            {
                "project": {"data_root": "data"},
                "pretrain_track": {"parent_sampling_mode": "window_magic"},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="parent_sampling_mode"):
        load_pretrain_track_config(config_path)


def test_pretrain_parent_validation_requires_split_manifest_and_causal_mode(tmp_path: Path) -> None:
    missing_path = tmp_path / "pretrain-parent-validation-missing.json"
    missing_path.write_text(
        json.dumps(
            {
                "project": {"data_root": "data"},
                "pretrain_track": {
                    "require_train_parent_manifest": True,
                    "require_parent_validation": True,
                    "sampling_mode": "causal_window",
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="parent_split_manifest_path"):
        load_pretrain_track_config(missing_path)

    full_track_path = tmp_path / "pretrain-parent-validation-full-track.json"
    full_track_path.write_text(
        json.dumps(
            {
                "project": {"data_root": "data"},
                "pretrain_track": {
                    "require_train_parent_manifest": True,
                    "require_parent_validation": True,
                    "parent_split_manifest_path": "split.json",
                    "sampling_mode": "full_track",
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="causal_window"):
        load_pretrain_track_config(full_track_path)


def test_expert1_rejects_pretrain_and_warm_start_together(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    _write_config(
        config_path,
        {
            "epochs": 18,
            "stage1_traj_only_epochs": 18,
            "prefer_stage2_checkpoint": False,
            "pretrained_track_checkpoint": "pretrain.pt",
            "trajectory_warm_start_checkpoint": "warm.pt",
        },
    )

    try:
        load_expert1_config(config_path)
    except ValueError as exc:
        assert "mutually exclusive" in str(exc)
    else:
        raise AssertionError("pretrain and warm-start must be mutually exclusive")


def test_source_robust_v3_requires_source_normalization(tmp_path: Path) -> None:
    template = Path("tmp/20260729-expert1-tr-only-v3-18epoch-seed42-v1.template.json")
    payload = json.loads(template.read_text(encoding="utf-8"))
    payload["expert1"]["source_normalization"] = False
    invalid = tmp_path / "invalid-physical-contract.json"
    invalid.write_text(json.dumps(payload), encoding="utf-8")

    try:
        load_expert1_config(invalid)
    except ValueError as exc:
        assert "requires source_normalization=true" in str(exc)
    else:
        raise AssertionError("source_robust_v3 must reject source_normalization=false")
