"""L3 knownness head 标签映射回归测试。"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from radar_ln3.training.l3_heads import _epoch_permutation, _route_targets


def test_profile_epochs_use_distinct_deterministic_shuffle_orders() -> None:
    first = _epoch_permutation(32, seed=42, epoch=1, phase_offset=20)
    second = _epoch_permutation(32, seed=42, epoch=2, phase_offset=20)
    repeat = _epoch_permutation(32, seed=42, epoch=1, phase_offset=20)

    assert not torch.equal(first, second)
    assert torch.equal(first, repeat)


def test_l2_non_classwise_counts_all_known_target_labels() -> None:
    labels = (
        "DroneTarget",
        "BirdTarget",
        "BalloonTarget",
        "UnknownTarget",
        "ClutterTarget",
    )

    targets = _route_targets("l2", labels, 1, torch.device("cpu"), l2_classwise=False)
    assert targets.sum(dim=(0, 1)).tolist() == [3.0]


def test_l2_classwise_keeps_per_class_support() -> None:
    labels = (
        "DroneTarget",
        "BirdTarget",
        "BirdTarget",
        "UnknownTarget",
    )

    targets = _route_targets("l2", labels, 1, torch.device("cpu"), l2_classwise=True)
    assert targets.sum(dim=(0, 1)).tolist() == [1.0, 2.0, 0.0]


def test_l1_route_support_keeps_target_and_clutter_semantics() -> None:
    labels = (
        "DroneTarget",
        "BalloonTarget",
        "ClutterTarget",
        "UnknownTarget",
    )

    targets = _route_targets("l1", labels, 1, torch.device("cpu"))
    assert targets.sum(dim=(0, 1)).tolist() == [2.0, 1.0]


def _materialize_heads(
    monkeypatch,
    *,
    records: list[dict[str, str]],
    view_sample_ids: tuple[str, ...] = ("p:event-0",),
    manifest_sha256: str | None = None,
):
    import radar_ln3.training.l3_heads as heads

    class Dataset:
        def __init__(self) -> None:
            self.event_cache_validated = False
            self.view = SimpleNamespace(
                sample_ids=view_sample_ids,
                view_name="fixture",
                view_hash="view-hash",
                split_name="train",
            )

        def validate_event_cache(self) -> None:
            self.event_cache_validated = True

        def __len__(self) -> int:
            return len(view_sample_ids)

        def __getitem__(self, index: int):
            return {"sample_id": view_sample_ids[index]}

    dataset = Dataset()
    manifest = SimpleNamespace(tr_records=list(records))
    fake_batch = SimpleNamespace(
        representation=torch.zeros((1, 1, 2)),
        profile_representation=None,
        bird_evidence_representation=None,
        padding_mask=torch.zeros((1, 1), dtype=torch.bool),
        endpoint_logits=torch.zeros((1, 2)),
        parent_ids=["p"],
        label_names=["DroneTarget"],
    )
    checkpoint_l1 = SimpleNamespace(file_sha256="a" * 64)
    checkpoint_l2 = SimpleNamespace(file_sha256="b" * 64)
    hash_calls: list[object] = []

    def manifest_hash(value):
        hash_calls.append(value)
        return "m" * 64

    monkeypatch.setattr(heads, "data_manifest_hash", manifest_hash)
    monkeypatch.setattr(
        heads, "_endpoint_representation", lambda *args, **kwargs: fake_batch
    )

    representation = heads.materialize_frozen_endpoint_representation(
        "project",
        task="l1",
        manifest_path="manifest.json",
        feature_cache_manifest_path="cache.json",
        split_name="train",
        training_view=True,
        validate_event_cache=True,
        manifest=manifest,
        config=SimpleNamespace(),
        feature_caches={},
        feature_cache_sha256="c" * 64,
        manifest_sha256=manifest_sha256,
        endpoint_models=(object(), object(), checkpoint_l1, checkpoint_l2),
        dataset=dataset,
    )
    return dataset, representation, hash_calls


def test_materialize_uses_loaded_checkpoint_hashes_when_endpoint_hashes_omitted(monkeypatch) -> None:
    dataset, representation, hash_calls = _materialize_heads(
        monkeypatch,
        records=[{"sample_id": "p:event-0", "label_name": "DroneTarget"}],
    )

    assert dataset.event_cache_validated
    assert dict(representation.endpoint_sha256) == {"l1": "a" * 64, "l2": "b" * 64}
    assert len(hash_calls) == 1


def test_materialize_accepts_caller_manifest_hash(monkeypatch) -> None:
    _, representation, hash_calls = _materialize_heads(
        monkeypatch,
        records=[{"sample_id": "p:event-0", "label_name": "DroneTarget"}],
        manifest_sha256="d" * 64,
    )

    assert hash_calls == []
    assert representation.source_manifest_sha256 == "d" * 64


def test_materialize_filters_labels_by_view(monkeypatch) -> None:
    import pytest

    _materialize_heads(
        monkeypatch,
        records=[
            {"sample_id": "p:event-0", "label_name": "DroneTarget"},
            {"sample_id": "p:event-1", "label_name": "BogusLabel"},
        ],
    )

    with pytest.raises(ValueError, match="knownness view 含未登记标签"):
        _materialize_heads(
            monkeypatch,
            records=[{"sample_id": "p:event-0", "label_name": "BogusLabel"}],
        )
