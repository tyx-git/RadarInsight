"""验证正式配置经过 CLI 后确实传入对应的 knownness 训练分支。"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import train_l3_heads


@pytest.fixture
def head_runner(tmp_path: Path, monkeypatch):
    from radar_ln3 import checkpoint, data, model
    from radar_ln3.data import manifest
    from radar_ln3.training import l3_heads

    source = Path(__file__).resolve().parents[1] / "config/radar_ln3_l3.json"
    document = json.loads(source.read_text(encoding="utf-8"))
    target = tmp_path / "src/config/radar_ln3_l3.json"
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps(document), encoding="utf-8")
    cached = tmp_path / "representation.pt"
    cached.touch()
    representation = SimpleNamespace(parent_ids=("parent",), source_manifest_sha256="manifest")
    config = SimpleNamespace(
        path=tmp_path / "model.json",
        artifacts=SimpleNamespace(l1_output_path=Path("l1.pt"), l2_output_path=Path("l2.pt")),
    )
    monkeypatch.setattr(model, "load_model_config", lambda *a, **kw: config)
    monkeypatch.setattr(data, "load_manifest", lambda *a, **kw: object())
    monkeypatch.setattr(manifest, "data_manifest_hash", lambda *a: "manifest")
    monkeypatch.setattr(checkpoint, "sha256_file", lambda *a: "fixture")
    monkeypatch.setattr(l3_heads, "endpoint_representation_path", lambda *a, **kw: cached)
    monkeypatch.setattr(l3_heads, "load_frozen_endpoint_representation", lambda *a, **kw: representation)
    calls = []

    def record_training(*a, **kwargs):
        calls.append(kwargs)
        return {"checkpoint": {}}

    monkeypatch.setattr(l3_heads, "train_l3_knownness_head", record_training)

    def run(*arguments):
        args = train_l3_heads._parser().parse_args([
            "--project-root", str(tmp_path), "--skip-register", *arguments,
        ])
        train_l3_heads.run(args)
        return calls

    return run, target, document


@pytest.mark.parametrize("task", ["both", "l1", "l2"])
def test_head_training_reads_formal_config_and_keeps_tasks_separate(head_runner, task):
    run, _, document = head_runner
    assert "unknown_loss_weight" not in document["knownness_training"]["l1"]
    assert "bird_loss_weight" not in document["knownness_training"]["l2"]
    calls = run("--task", task)
    assert [c["task"] for c in calls] == (["l1", "l2"] if task == "both" else [task])
    for call in calls:
        assert call["epochs"] == 30
        assert call["batch_size"] == 32
        assert call["learning_rate"] == 0.001
        assert call["weight_decay"] == 0.0001
        assert call["unknown_far_limit"] == 0.65
        if call["task"] == "l1":
            assert call["unknown_loss_weight"] == 0.15
            assert call["l1_loss_mode"] == "endpoint_tail"
            assert call["l1_bce_unknown_weight"] == 1.25
            assert call["l1_endpoint_weight"] == 1.0
            assert call["l1_tail_weight"] == 0.15
            assert call["l1_tail_events"] == 4
            assert call["l1_target_prefix_weight"] == 0.05
            assert call["l1_clutter_weight"] == 0.5
            assert call["l1_selection_mode"] == "operating_point"
            assert call["l2_unknown_loss_weight"] == 1.0
        else:
            assert call["unknown_loss_weight"] is None
            assert call["l2_unknown_loss_weight"] == 0.88
            assert call["l2_classwise"] is False
            assert call["bird_loss_weight"] == 1.0
            assert call["l1_target_prefix_weight"] == 0.0
            assert call["l1_loss_mode"] == "prefix_endpoint"


def test_config_changes_and_explicit_cli_overrides_reach_trainer(head_runner):
    run, target, document = head_runner
    settings = document["knownness_training"]
    settings["common"]["epochs"] = 7
    settings["l1"]["bce_unknown_weight"] = 1.4
    settings["l2"]["unknown_loss_weight"] = 0.9
    target.write_text(json.dumps(document), encoding="utf-8")
    l1, l2 = run("--epochs", "5", "--l2-unknown-loss-weight", "0.92")
    assert l1["epochs"] == l2["epochs"] == 5
    assert l1["l1_bce_unknown_weight"] == 1.4
    assert l2["l2_unknown_loss_weight"] == 0.92


def test_l2_still_rejects_misrouted_unknown_weight(head_runner):
    run, _, _ = head_runner
    with pytest.raises(ValueError, match="L2 请使用 --l2-unknown-loss-weight"):
        run("--task", "l2", "--unknown-loss-weight", "0.88")


def test_l1_unknown_loss_weight_can_be_explicitly_set_from_cli(head_runner):
    run, _, _ = head_runner
    calls = run("--task", "l1", "--unknown-loss-weight", "0.15")
    assert calls[-1]["unknown_loss_weight"] == 0.15
