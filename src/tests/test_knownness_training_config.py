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
    datasets = []

    class FakeDataset:
        def __init__(self, *args, split_name, validate_event_cache, **kwargs):
            self.split_name = split_name
            self.event_cache_validated = bool(validate_event_cache)
            self.view = SimpleNamespace(view_hash=f"{split_name}-view")
            datasets.append(self)

    monkeypatch.setattr(l3_heads, "CachedParentSequenceDataset", FakeDataset)
    calls = []

    def record_training(*a, **kwargs):
        calls.append(kwargs)
        return {"checkpoint": {}}

    monkeypatch.setattr(l3_heads, "train_l3_knownness_head", record_training)

    def run(*arguments, skip_register: bool = True):
        argv = ["--project-root", str(tmp_path)]
        if skip_register:
            argv.append("--skip-register")
        args = train_l3_heads._parser().parse_args([*argv, *arguments])
        train_l3_heads.run(args)
        return calls

    run.datasets = datasets

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


def test_config_changes_reach_trainer(head_runner):
    run, target, document = head_runner
    settings = document["knownness_training"]
    settings["common"]["epochs"] = 7
    settings["l1"]["bce_unknown_weight"] = 1.4
    settings["l2"]["unknown_loss_weight"] = 0.9
    target.write_text(json.dumps(document), encoding="utf-8")
    l1, l2 = run()
    assert l1["epochs"] == l2["epochs"] == 7
    assert l1["l1_bce_unknown_weight"] == 1.4
    assert l2["l2_unknown_loss_weight"] == 0.9


def test_non_classwise_l2_rejects_bird_loss_weight(head_runner):
    run, target, document = head_runner
    document["knownness_training"]["l2"]["bird_loss_weight"] = 1.2
    target.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="bird_loss_weight"):
        run("--task", "l2")


def test_head_training_only_registers_default_root(head_runner, monkeypatch) -> None:
    from radar_ln3.data import derived

    run, _, _ = head_runner
    registrations = []
    monkeypatch.setattr(
        derived,
        "register_product_artifacts",
        lambda *args, **kwargs: registrations.append(kwargs)
        or Path("manifest.json"),
    )

    run("--root", "tmp-out", skip_register=False)
    assert registrations == []

    run("--root", "artifacts/radar_ln3", skip_register=False)
    assert len(registrations) == 1
    assert registrations[0]["manifest_path"] == Path("src/config/manifests.json")
    assert set(registrations[0]["l3_head_descriptors"]) == {"l1", "l2"}


def test_is_default_artifact_root() -> None:
    from radar_ln3.artifacts import is_default_artifact_root

    assert is_default_artifact_root(Path("artifacts", "radar_ln3")) is True
    assert is_default_artifact_root("artifacts/radar_ln3/") is True
    assert is_default_artifact_root(Path("artifacts/radar_ln3_v1")) is False
    assert is_default_artifact_root(Path("tmp/L3")) is False


def test_l3_layout_derives_outputs_from_root(tmp_path: Path) -> None:
    from radar_ln3.artifacts import l3_layout

    layout = l3_layout(tmp_path)
    assert layout.heads == Path("artifacts/radar_ln3/l3/heads")
    assert layout.l1_head == Path("artifacts/radar_ln3/l3/heads/l1-knownness.pt")
    assert layout.l2_head == Path("artifacts/radar_ln3/l3/heads/l2-knownness.pt")
    assert layout.l1_checkpoint == Path("artifacts/radar_ln3/l1/l1.pt")
    assert layout.l2_checkpoint == Path("artifacts/radar_ln3/l2/l2.pt")
    assert layout.representations == Path("artifacts/radar_ln3/l3/representations")
    assert layout.exploratory_calibration == Path(
        "artifacts/radar_ln3/l3/radar_ln3_calibration.exploratory.json"
    )
    assert layout.calibration == Path("artifacts/radar_ln3/l3/radar_ln3_calibration.json")
    assert layout.audit_manifest == Path(
        "artifacts/radar_ln3/l3/radar_ln3_audit_manifest.json"
    )
    assert layout.audit_report == Path(
        "artifacts/radar_ln3/l3/radar_ln3_audit_report.json"
    )
    assert layout.audit_evidence == Path("artifacts/radar_ln3/l3/audit-evidence.jsonl")
    assert layout.head_fit_feature_cache == Path(
        "data/cq08/cache/features/l3/head_fit/feature-cache-manifest.json"
    )
    assert layout.validation_feature_cache == Path(
        "data/cq08/cache/features/l3/validation/feature-cache-manifest.json"
    )
    assert layout.audit_feature_cache == Path(
        "data/cq08/cache/features/l3/audit/feature-cache-manifest.json"
    )

    custom = l3_layout(tmp_path, "tmp/L3")
    assert custom.root == Path("tmp/L3")
    assert custom.heads == Path("tmp/L3/l3/heads")
    assert custom.l1_checkpoint == Path("tmp/L3/l1/l1.pt")
    assert custom.l2_checkpoint == Path("tmp/L3/l2/l2.pt")
    assert custom.audit_evidence == Path("tmp/L3/l3/audit-evidence.jsonl")
    with pytest.raises(ValueError, match="Root 必须位于项目目录内"):
        l3_layout(tmp_path, tmp_path.parent / "outside-root")


def test_head_training_constructs_one_validated_dataset_per_split(head_runner):
    run, _, _ = head_runner
    run("--task", "both")
    assert [dataset.split_name for dataset in run.datasets] == ["train", "validation"]
    assert all(dataset.event_cache_validated for dataset in run.datasets)


def test_head_training_hashes_each_feature_cache_manifest_once(head_runner, monkeypatch):
    from radar_ln3 import checkpoint

    run, target, _ = head_runner
    root = target.parents[2]
    hash_calls: list[Path] = []
    monkeypatch.setattr(
        checkpoint,
        "sha256_file",
        lambda path: hash_calls.append(Path(path)) or "fixture",
    )

    run("--task", "both")

    assert hash_calls.count(
        root / "data/cq08/cache/features/l3/head_fit/feature-cache-manifest.json"
    ) == 1
    assert hash_calls.count(
        root / "data/cq08/cache/features/l3/validation/feature-cache-manifest.json"
    ) == 1


def test_event_cache_skip_is_independent_from_data_validation() -> None:
    from scripts import build_l3_cache

    args = build_l3_cache._parser().parse_args(
        ["--project-root", ".", "--role", "validation", "--skip-event-cache-validation"]
    )
    assert args.skip_event_cache_validation is True
    assert args.skip_data_validation is False


def test_head_training_reuses_each_split_dataset_for_l1_and_l2(head_runner, monkeypatch):
    from radar_ln3 import checkpoint, features
    from radar_ln3.training import l3, l3_heads

    run, target, document = head_runner
    root = target.parents[2]
    hashed: list[Path] = []
    monkeypatch.setattr(
        checkpoint,
        "sha256_file",
        lambda path: hashed.append(Path(path)) or "fixture",
    )
    monkeypatch.setattr(
        l3_heads,
        "load_frozen_endpoint_representation",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("rebuild")),
    )
    monkeypatch.setattr(
        features,
        "load_feature_cache_bundle",
        lambda *args, **kwargs: {},
    )
    endpoint_models = (object(), object(), SimpleNamespace(file_sha256="a" * 64), SimpleNamespace(file_sha256="b" * 64))
    endpoint_loads = []
    monkeypatch.setattr(
        l3,
        "_load_endpoint_models",
        lambda *args, **kwargs: endpoint_loads.append(kwargs) or endpoint_models,
    )
    materialized = []

    def materialize(*args, **kwargs):
        materialized.append(kwargs)
        return SimpleNamespace(parent_ids=("parent",), source_manifest_sha256="manifest")

    monkeypatch.setattr(l3_heads, "materialize_frozen_endpoint_representation", materialize)
    monkeypatch.setattr(l3_heads, "save_frozen_endpoint_representation", lambda *args, **kwargs: None)
    monkeypatch.setattr(l3_heads, "endpoint_representation_path", lambda root, *, task, split_name, representation_root: Path(f"{task}-{split_name}.pt"))

    run("--task", "both", "--root", "tmp-out")

    assert len(materialized) == 4
    assert materialized[0]["dataset"] is materialized[2]["dataset"]
    assert materialized[1]["dataset"] is materialized[3]["dataset"]
    assert materialized[0]["dataset"] is not materialized[1]["dataset"]
    # endpoint 绑定必须跟随 --root，而不是固定正式目录。
    assert root / "tmp-out/l1/l1.pt" in hashed
    assert root / "tmp-out/l2/l2.pt" in hashed
    assert len(endpoint_loads) == 1
    assert endpoint_loads[0]["l1_checkpoint_path"] == Path("tmp-out/l1/l1.pt")
    assert endpoint_loads[0]["l2_checkpoint_path"] == Path("tmp-out/l2/l2.pt")
    # 未显式传 --device/--batch-size 时回落到 knownness_training.common。
    common = document["knownness_training"]["common"]
    assert {entry["device"] for entry in materialized} == {common["device"]}
    assert {entry["batch_size"] for entry in materialized} == {common["batch_size"]}


@pytest.mark.parametrize(
    ("skip_data", "skip_event", "expected_files", "expected_events"),
    [(True, False, False, True), (False, True, True, False)],
)
def test_l3_cache_validation_switches_are_independent(
    tmp_path: Path,
    monkeypatch,
    skip_data: bool,
    skip_event: bool,
    expected_files: bool,
    expected_events: bool,
):
    from scripts import build_l3_cache
    from radar_ln3.data import cq08
    import radar_ln3.data as data_module
    from radar_ln3.model import ModelConfig
    from core.training import cache as training_cache

    manifest_obj = SimpleNamespace()
    adapters = []

    class Adapter:
        source_root = tmp_path

        def __init__(self, *args, **kwargs):
            self.verify_files = kwargs["verify_files"]
            self.manifest = kwargs.get("manifest")
            adapters.append(self)

    class Provider:
        def __init__(self, *args, **kwargs):
            pass

    captured = {}
    monkeypatch.setattr(cq08, "CQ08Adapter", Adapter)
    monkeypatch.setattr(cq08, "CQ08RawEchoProvider", Provider)
    monkeypatch.setattr(data_module, "load_manifest", lambda *args, **kwargs: manifest_obj)
    import radar_ln3.model as model
    monkeypatch.setattr(
        model,
        "load_model_config",
        lambda *args, **kwargs: SimpleNamespace(
            data=SimpleNamespace(l2=SimpleNamespace(stable8_median=(), stable8_scale=()))
        ),
    )

    def build(*args, **kwargs):
        captured.update(kwargs)
        return {
            "feature_cache_manifest": "cache.json",
            "view_name": "validation",
            "view_hash": "view",
            "record_count": 1,
        }

    monkeypatch.setattr(training_cache, "build_feature_training_cache", build)
    cli = ["--project-root", str(tmp_path), "--role", "validation"]
    if skip_data:
        cli.append("--skip-data-validation")
    if skip_event:
        cli.append("--skip-event-cache-validation")
    build_l3_cache.build(build_l3_cache._parser().parse_args(cli))

    assert captured["verify_files"] is expected_files
    assert captured["validate_event_cache"] is expected_events
    assert adapters[-1].manifest is manifest_obj
    assert captured["manifest"] is manifest_obj
