from __future__ import annotations

import json
import unittest
from pathlib import Path

import pandas as pd
import torch

from src.core.data.collate import collate_track_batch
from src.core.data.rd_segment_store import WindowAssembler, build_window_segment_map, resolve_processed_path
from src.core.data.torch_dataset import TorchTrackDataset
from src.core.expert1_data import build_expert1_dataloaders
from src.core.expert1_data import RDShardLocalBatchSampler
from src.core.experts import LowAltitudeSuspectExpert
from src.core.expert1_trainer import (
    _build_class_weights,
    _build_class_weights_from_index,
    _build_rd_class_weights,
    _build_rd_class_weights_from_index,
    _rd_coverage,
    _rd_coverage_from_index,
)
from src.process.build_expert1_rd_segment_store import build_expert1_rd_segment_store
from src.process.build_expert1_rd_segment_store import _load_parent_rd
from src.process.build_expert1_rd_segment_store import _slice_parent_rd_segment


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"


def _truthy(value: object) -> bool:
    return str(value if value is not None else "").strip().lower() in {"1", "true", "yes", "y"}


class Expert1RDSegmentStoreSmokeTests(unittest.TestCase):
    store_name = "tmp_unittest_expert1_rd_segment_store"
    manifest_name = f"{store_name}_manifest.csv"
    map_name = "tmp_unittest_expert1_window_segment_map.csv"

    @classmethod
    def setUpClass(cls) -> None:
        stale_path = DATA_ROOT / "processed" / cls.store_name / "stale_segments.pt"
        stale_path.parent.mkdir(parents=True, exist_ok=True)
        stale_path.write_bytes(b"stale")
        cls.store_summary = build_expert1_rd_segment_store(
            ROOT,
            index_name="expert1_window_index.csv",
            output_name=cls.store_name,
            max_parents=1,
            source_names=["raw_tr_rd"],
            deep_patch_mode="patch",
            overwrite=True,
        )
        cls.map_summary = build_window_segment_map(
            DATA_ROOT,
            segment_manifest_name=cls.manifest_name,
            output_name=cls.map_name,
        )
        cls.stale_path = stale_path

    def test_build_store_overwrites_stale_shards_and_writes_manifest(self) -> None:
        manifest_path = resolve_processed_path(DATA_ROOT, self.manifest_name)
        summary_path = manifest_path.with_name(f"{self.store_name}_summary.json")
        self.assertFalse(self.stale_path.exists())
        self.assertTrue(manifest_path.exists())
        self.assertTrue(summary_path.exists())
        self.assertEqual(self.store_summary["parent_count"], 1)
        self.assertGreater(self.store_summary["segment_count"], 0)
        self.assertIn("exact_time_sync", self.store_summary["rd_time_sync_status_counts"])

        manifest = pd.read_csv(manifest_path)
        self.assertGreater(len(manifest), 0)
        self.assertTrue((ROOT / str(manifest.iloc[0]["shard_key"])).exists())
        self.assertTrue(
            set(
                [
                    "segment_id",
                    "parent_sample_id",
                    "shard_key",
                    "segment_index",
                    "rd_frame_start_index",
                    "rd_frame_end_index",
                    "rd_time_coverage_ratio",
                ]
            ).issubset(manifest.columns)
        )
        self.assertEqual(str(manifest.iloc[0]["rd_time_sync_status"]), "exact_time_sync")
        self.assertGreaterEqual(int(manifest.iloc[0]["rd_frame_start_index"]), 0)
        self.assertGreaterEqual(float(manifest.iloc[0]["rd_time_coverage_ratio"]), 0.5)

    def test_window_map_contains_only_rd_windows_and_preserves_coverage(self) -> None:
        map_path = resolve_processed_path(DATA_ROOT, self.map_name)
        self.assertTrue(map_path.exists())
        self.assertGreater(self.map_summary["window_count"], 0)
        self.assertGreater(self.map_summary["mapped_window_count"], 0)
        self.assertGreaterEqual(self.map_summary["missing_window_count"], 0)
        self.assertGreater(self.map_summary["rd_coverage_ratio_mean"], 0.0)

        mapped = pd.read_csv(map_path)
        self.assertGreater(len(mapped), 0)
        first = mapped.iloc[0]
        self.assertGreaterEqual(int(first["segment_count"]), 1)
        self.assertGreater(float(first["rd_coverage_ratio"]), 0.0)
        self.assertLessEqual(float(first["rd_coverage_ratio"]), 1.0)

    def test_dataset_reads_segment_store_and_collates_rd_batch(self) -> None:
        map_path = resolve_processed_path(DATA_ROOT, self.map_name)
        mapped = pd.read_csv(map_path)
        row = mapped.iloc[0]
        sample_id = str(row["sample_id"])

        dataset = TorchTrackDataset(
            DATA_ROOT,
            sample_ids=[sample_id],
            trainable_only=False,
            index_name="expert1_window_index.csv",
            use_rd_segment_store=True,
            rd_segment_manifest_name=self.manifest_name,
            window_segment_map_name=self.map_name,
            rd_segment_store_cache_size=2,
        )
        self.assertEqual(len(dataset), 1)
        item = dataset[0]
        self.assertEqual(item.sample_id, sample_id)
        self.assertGreater(item.rd_valid_length, 1)
        self.assertEqual(tuple(item.rd_features.shape[1:]), (512, 4))
        self.assertEqual(tuple(item.rd_patch_features.shape[1:]), (64, 64, 3))
        self.assertEqual(tuple(item.rd_deep_patch_features.shape[1:]), (64, 64, 3))

        segment_ids = set(json.loads(str(row["segment_ids_json"])))
        manifest = pd.read_csv(resolve_processed_path(DATA_ROOT, self.manifest_name))
        selected = manifest[manifest["segment_id"].astype(str).isin(segment_ids)]
        expected_multimodal = any(_truthy(value) for value in selected["rd_is_multimodal"].tolist())
        self.assertEqual(item.rd_is_multimodal, expected_multimodal)
        self.assertEqual(item.rd_time_sync_status, "exact_time_sync")

        batch = collate_track_batch([item])
        self.assertEqual(batch.sample_ids, [sample_id])
        self.assertEqual(batch.rd_time_sync_status, ["exact_time_sync"])
        self.assertEqual(tuple(batch.rd_features.shape), (1, item.rd_valid_length, 512, 4))
        self.assertEqual(int(batch.rd_lengths[0].item()), item.rd_valid_length)
        self.assertFalse(bool(batch.rd_padding_mask[0, 0].item()))
        self.assertEqual(bool(batch.rd_is_multimodal[0].item()), expected_multimodal)

    def test_index_level_statistics_match_dataloader_scan_without_rd_reload(self) -> None:
        sample_ids = ["cq08|track|1010|win0000", "excel|track|1|win0000"]
        dataset = TorchTrackDataset(
            DATA_ROOT,
            sample_ids=sample_ids,
            trainable_only=False,
            index_name="expert1_window_index.csv",
            cache_size=4,
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_track_batch)
        self.assertTrue(torch.allclose(_build_class_weights(loader), _build_class_weights_from_index(dataset.index_df)))
        loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_track_batch)
        self.assertTrue(torch.allclose(_build_rd_class_weights(loader), _build_rd_class_weights_from_index(dataset.index_df)))
        loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_track_batch)
        self.assertEqual(_rd_coverage(loader), _rd_coverage_from_index(dataset.index_df))

    def test_no_rd_batch_uses_trajectory_only_fast_path(self) -> None:
        torch.manual_seed(7)
        model = LowAltitudeSuspectExpert(
            input_dim=4,
            rd_context_dim=0,
            use_rd_patch_cnn=True,
            traj_sequence_dim=16,
            hidden_dim=16,
            dropout=0.0,
        )
        model.eval()
        batch_size = 3
        with torch.no_grad():
            output = model(
                torch.randn(batch_size, 4),
                rd_features=torch.zeros(batch_size, 1, 512, 4),
                traj_sequence_features=torch.randn(batch_size, 5, 16),
                rd_patch_features=torch.zeros(batch_size, 1, 64, 64, 3),
                rd_deep_patch_features=torch.zeros(batch_size, 1, 64, 64, 3),
                rd_is_multimodal=torch.zeros(batch_size, dtype=torch.bool),
                traj_padding_mask=torch.zeros(batch_size, 5, dtype=torch.bool),
                rd_padding_mask=torch.zeros(batch_size, 1, dtype=torch.bool),
                rd_patch_padding_mask=torch.zeros(batch_size, 1, dtype=torch.bool),
                rd_deep_patch_padding_mask=torch.zeros(batch_size, 1, dtype=torch.bool),
                enable_reject=False,
            )
        self.assertFalse(bool(output.tr_rd_active.any().item()))
        self.assertTrue(torch.allclose(output.logits, output.traj_logits, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(output.joint_logits, output.traj_logits, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.equal(output.rd_logits, torch.zeros_like(output.rd_logits)))
        self.assertTrue(torch.equal(output.joint_rd_weight, torch.zeros_like(output.joint_rd_weight)))

    def test_csv_string_false_does_not_activate_multimodal_rd(self) -> None:
        store_dir = DATA_ROOT / "processed" / "tmp_unittest_bool_rd_segment_store"
        store_dir.mkdir(parents=True, exist_ok=True)
        shard_path = store_dir / "synthetic_segments.pt"
        relative_shard = shard_path.relative_to(ROOT).as_posix()
        patch = torch.zeros((2, 64, 64, 3), dtype=torch.float32)
        torch.save(
            {
                "schema_version": 1,
                "rd_is_multimodal": False,
                "segments": [
                    {
                        "segment_id": "synthetic-parent|seg0000",
                        "segment_index": 0,
                        "segment_start_seconds": 0.0,
                        "segment_end_seconds": 2.0,
                        "rd_features": torch.ones((2, 512, 4), dtype=torch.float32),
                        "rd_valid_length": 2,
                        "rd_patch_features": patch,
                        "rd_patch_valid_length": 2,
                        "rd_deep_patch_features": patch,
                        "rd_deep_patch_valid_length": 2,
                    }
                ],
            },
            shard_path,
        )
        manifest_name = "tmp_unittest_bool_rd_segment_manifest.csv"
        manifest_path = DATA_ROOT / "processed" / manifest_name
        manifest_path.write_text(
            "\n".join(
                [
                    "segment_id,parent_sample_id,segment_start_seconds,segment_end_seconds,shard_key,segment_index,rd_time_sync_status,rd_quality_flags,rd_is_multimodal",
                    f"synthetic-parent|seg0000,synthetic-parent,0.0,2.0,{relative_shard},0,exact,ok,False",
                    "unused-parent|seg0001,unused-parent,2.0,4.0,,1,exact,ok,",
                ]
            ),
            encoding="utf-8",
        )
        assembler = WindowAssembler(DATA_ROOT, segment_manifest_name=manifest_name, cache_size=0)
        encoded, rd_is_multimodal, assembly = assembler.assemble(
            {
                "sample_id": "synthetic-parent|win0000",
                "parent_sample_id": "synthetic-parent",
                "window_start_seconds": 0.0,
                "window_end_seconds": 2.0,
            }
        )
        self.assertIsNotNone(encoded)
        self.assertIsNotNone(assembly)
        self.assertFalse(rd_is_multimodal)
        self.assertEqual(encoded.valid_length, 2)

    def test_cq08_proportional_fallback_crops_raw_frames_before_encoding(self) -> None:
        tensor_dir = DATA_ROOT / "processed" / "tmp_unittest_cq08_prop_rd"
        tensor_dir.mkdir(parents=True, exist_ok=True)
        tensor_path = tensor_dir / "parent_rd_tensor.npz"
        amplitude = torch.arange(10 * 4 * 4, dtype=torch.float32).reshape(10, 4, 4).numpy()
        phase = torch.zeros((10, 4, 4), dtype=torch.float32).numpy()
        time_raw = torch.arange(10, dtype=torch.float32).numpy()
        import numpy as np

        np.savez_compressed(
            tensor_path,
            amplitude=amplitude,
            phase=phase,
            time_raw=time_raw,
        )
        parent_rd = _load_parent_rd(
            DATA_ROOT,
            pd.Series(
                {
                    "source_name": "cq08_track",
                    "rd_tensor_path": tensor_path.relative_to(ROOT).as_posix(),
                }
            ),
        )

        segment, status, frame_start, frame_end, time_start, time_end, coverage = _slice_parent_rd_segment(
            parent_rd,
            parent_start=0.0,
            parent_end=100.0,
            segment_start=20.0,
            segment_end=40.0,
            min_frames=1,
            deep_patch_mode="patch",
        )

        self.assertEqual(status, "proportional_fallback")
        self.assertEqual(frame_start, 2)
        self.assertEqual(frame_end, 3)
        self.assertIsNone(time_start)
        self.assertIsNone(time_end)
        self.assertEqual(coverage, 0.0)
        self.assertGreaterEqual(segment.valid_length, 1)

    def test_train_loader_can_use_workers_while_eval_stays_single_process(self) -> None:
        loaders = build_expert1_dataloaders(
            DATA_ROOT,
            index_name="expert1_window_index.csv",
            batch_size=4,
            seed=42,
            num_workers=6,
            eval_num_workers=0,
            dataset_cache_size=0,
            rd_encoded_cache_size=16,
            stage2_rd_shard_local_sampling=True,
            use_rd_segment_store=True,
            rd_segment_manifest_name=self.manifest_name,
            window_segment_map_name=self.map_name,
            rd_segment_store_cache_size=1,
            allow_rd_segment_store_fallback=True,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=1,
        )

        self.assertEqual(loaders.train_loader.num_workers, 6)
        self.assertEqual(loaders.stage2_train_loader.num_workers, 6)
        self.assertEqual(loaders.val_loader.num_workers, 0)
        self.assertEqual(loaders.test_loader.num_workers, 0)
        self.assertFalse(loaders.train_loader.dataset.load_rd)
        self.assertTrue(loaders.stage2_train_loader.dataset.load_rd)
        self.assertIsNotNone(loaders.stage2_train_loader.dataset._window_assembler)
        self.assertEqual(loaders.train_loader.dataset.cache_size, 0)
        self.assertEqual(loaders.val_loader.dataset.cache_size, 0)
        self.assertEqual(loaders.stage2_train_loader.dataset.rd_cache_size, 16)
        self.assertEqual(loaders.stage2_train_loader.dataset._window_assembler.cache_size, 1)
        self.assertEqual(loaders.stage2_train_loader.batch_sampler.__class__.__name__, "RDShardLocalBatchSampler")
        self.assertTrue(loaders.train_loader.pin_memory)
        self.assertTrue(loaders.train_loader.persistent_workers)
        self.assertEqual(loaders.train_loader.prefetch_factor, 1)

    def test_stage2_loader_keeps_rd_when_rd_aware_sampler_is_disabled(self) -> None:
        loaders = build_expert1_dataloaders(
            DATA_ROOT,
            index_name="expert1_window_index.csv",
            batch_size=4,
            seed=42,
            num_workers=0,
            eval_num_workers=0,
            stage2_rd_aware_sampling=False,
            use_rd_segment_store=True,
            rd_segment_manifest_name=self.manifest_name,
            window_segment_map_name=self.map_name,
            rd_segment_store_cache_size=1,
            allow_rd_segment_store_fallback=True,
            pin_memory=False,
            persistent_workers=False,
            prefetch_factor=1,
        )

        self.assertFalse(loaders.train_loader.dataset.load_rd)
        self.assertTrue(loaders.stage2_train_loader.dataset.load_rd)
        self.assertIsNot(loaders.stage2_train_loader, loaders.train_loader)

    def test_rd_shard_local_sampler_prefers_single_shard_batches(self) -> None:
        sampler = RDShardLocalBatchSampler(
            rd_indices=[0, 1, 2, 3, 4, 5],
            no_rd_indices=[6, 7, 8, 9],
            sample_weights=[1.0] * 10,
            batch_size=4,
            rd_fraction=0.5,
            seed=7,
            shard_keys_by_index={
                0: "a",
                1: "a",
                2: "a",
                3: "b",
                4: "b",
                5: "c",
            },
        )
        rd_shards = {
            0: "a",
            1: "a",
            2: "a",
            3: "b",
            4: "b",
            5: "c",
        }
        batches = list(sampler)
        self.assertGreater(len(batches), 0)
        local_batch_count = 0
        for batch in batches[:8]:
            shards = {rd_shards[index] for index in batch if index in rd_shards}
            if len(shards) <= 1:
                local_batch_count += 1
        self.assertGreaterEqual(local_batch_count, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
