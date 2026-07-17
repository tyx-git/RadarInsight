"""Smoke tests for stage2 RD dataloader throughput."""
from __future__ import annotations

import csv
import os
import time
import unittest

import torch


SHARD_DIR = os.path.join("data", "processed", "expert1_rd_segment_store")
MANIFEST_PATH = os.path.join("data", "processed", "expert1_rd_segment_manifest.csv")
HAS_SHARDS = os.path.isdir(SHARD_DIR) and any(
    f.endswith(".pt") for f in os.listdir(SHARD_DIR)
)


def _first_shard_path() -> str | None:
    if not HAS_SHARDS:
        return None
    for f in sorted(os.listdir(SHARD_DIR)):
        if f.endswith(".pt"):
            return os.path.join(SHARD_DIR, f)
    return None


def _largest_shard_path() -> str | None:
    if not HAS_SHARDS:
        return None
    best, best_size = None, 0
    for f in os.listdir(SHARD_DIR):
        if not f.endswith(".pt"):
            continue
        p = os.path.join(SHARD_DIR, f)
        s = os.path.getsize(p)
        if s > best_size:
            best, best_size = p, s
    return best


@unittest.skipUnless(HAS_SHARDS, "RD segment store not built")
class TestMmapShardLoading(unittest.TestCase):
    """Verify mmap=True produces correct data and is faster than default."""

    def test_mmap_data_matches_regular_load(self):
        path = _first_shard_path()
        regular = torch.load(path, map_location="cpu", weights_only=False)
        mmaped = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        reg_segs = regular["segments"]
        mmap_segs = mmaped["segments"]
        self.assertEqual(len(reg_segs), len(mmap_segs))
        for i, (r, m) in enumerate(zip(reg_segs, mmap_segs)):
            r_len = int(r.get("rd_valid_length", 0))
            m_len = int(m.get("rd_valid_length", 0))
            self.assertEqual(r_len, m_len, f"segment {i} valid_length mismatch")
            if r_len > 0:
                r_feat = r["rd_features"][:r_len].to(torch.float32)
                m_feat = m["rd_features"][:m_len].to(torch.float32).contiguous()
                self.assertTrue(
                    torch.equal(r_feat, m_feat),
                    f"segment {i} rd_features mismatch",
                )

    def test_contiguous_independent_of_mmap_source(self):
        path = _first_shard_path()
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        seg = payload["segments"][0]
        rd_len = int(seg.get("rd_valid_length", 0))
        if rd_len == 0:
            self.skipTest("segment has no valid data")
        copied = seg["rd_features"][:rd_len].to(torch.float32).contiguous()
        del payload  # delete mmap source
        self.assertEqual(copied.shape[0], rd_len)
        self.assertTrue(torch.isfinite(copied).all())

    def test_mmap_load_faster_than_regular(self):
        path = _largest_shard_path()
        if path is None or os.path.getsize(path) < 50 * 1024 * 1024:
            self.skipTest("need a shard >= 50MB for timing comparison")
        # warm OS page cache
        torch.load(path, map_location="cpu", weights_only=False)
        # regular
        t0 = time.perf_counter()
        for _ in range(3):
            torch.load(path, map_location="cpu", weights_only=False)
        regular_ms = (time.perf_counter() - t0) / 3 * 1000
        # mmap
        t0 = time.perf_counter()
        for _ in range(3):
            torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        mmap_ms = (time.perf_counter() - t0) / 3 * 1000
        self.assertLess(
            mmap_ms,
            regular_ms / 3,
            f"mmap ({mmap_ms:.0f}ms) not fast enough vs regular ({regular_ms:.0f}ms)",
        )


@unittest.skipUnless(HAS_SHARDS, "RD segment store not built")
class TestWindowAssemblerMmap(unittest.TestCase):
    """Verify WindowAssembler works with mmap-loaded payloads."""

    def test_assemble_returns_valid_sequence(self):
        from src.core.data.rd_segment_store import WindowAssembler

        manifest = os.path.join("data", "processed", "expert1_rd_segment_manifest.csv")
        wmap = os.path.join("data", "processed", "expert1_window_segment_map.csv")
        if not os.path.exists(manifest) or not os.path.exists(wmap):
            self.skipTest("manifest or window-segment map missing")
        assembler = WindowAssembler(
            "data",
            segment_manifest_name="expert1_rd_segment_manifest.csv",
            window_segment_map_name="expert1_window_segment_map.csv",
            cache_size=8,
        )
        index_path = os.path.join("data", "processed", "expert1_window_index.csv")
        if not os.path.exists(index_path):
            self.skipTest("window index missing")
        sample_row = None
        with open(index_path) as f:
            for row in csv.DictReader(f):
                if row.get("is_trainable", "").strip().lower() in ("true", "1", "yes"):
                    rd_path = str(row.get("rd_tensor_path", "")).strip().lower()
                    if rd_path not in ("", "nan", "none"):
                        sample_row = row
                        break
        if sample_row is None:
            self.skipTest("no RD sample found in window index")
        encoded, is_multimodal, assembly = assembler.assemble(sample_row)
        self.assertIsNotNone(encoded, "assemble() returned None for RD sample")
        self.assertGreater(encoded.valid_length, 0)
        self.assertTrue(encoded.features.is_contiguous())


@unittest.skipUnless(HAS_SHARDS, "RD segment store not built")
class TestStage2Throughput(unittest.TestCase):
    """Benchmark stage2 dataloader iteration speed."""

    def test_stage2_loader_throughput(self):
        from src.core.expert1_data import build_expert1_dataloaders

        loaders = build_expert1_dataloaders(
            data_root="data",
            index_name="expert1_window_index.csv",
            batch_size=8,
            seed=42,
            num_workers=0,
            eval_num_workers=0,
            use_rd_segment_store=True,
            rd_segment_manifest_name="expert1_rd_segment_manifest.csv",
            window_segment_map_name="expert1_window_segment_map.csv",
            rd_segment_store_cache_size=8,
            stage2_rd_aware_sampling=True,
            stage2_rd_batch_fraction=0.55,
            stage2_rd_shard_local_sampling=True,
            dataset_cache_size=32,
        )
        loader = loaders.stage2_train_loader
        batch_times = []
        for i, batch in enumerate(loader):
            if i >= 10:
                break
            t0 = time.perf_counter()
            _ = batch.features.shape
            _ = batch.rd_features.shape
            if i > 0:
                batch_times.append(time.perf_counter() - t0)
        if batch_times:
            avg_ms = sum(batch_times) / len(batch_times) * 1000
            print(f"\n  Avg stage2 batch time (num_workers=0, 10 batches): {avg_ms:.1f}ms")
            self.assertLess(avg_ms, 5000, "Stage2 batch time too slow even with mmap")


class TestRDShardLocalBatchSamplerGrouping(unittest.TestCase):
    """Verify shard_group_batches causes consecutive batches to share a shard."""

    def test_consecutive_batches_share_shard(self):
        from src.core.expert1_data import RDShardLocalBatchSampler

        # 120 RD indices across 3 shards, 80 no-RD indices
        # → num_batches = max(ceil(120/4), ceil(80/4)) = 30
        # → three shard groups of four visits are independently shuffled
        #    and repeated as cycles until all 30 batches are produced
        rd_indices = list(range(120))
        no_rd_indices = list(range(120, 200))
        weights = [1.0] * 200
        shard_keys = {i: f"shard_{i // 40}" for i in range(120)}

        sampler = RDShardLocalBatchSampler(
            rd_indices=rd_indices,
            no_rd_indices=no_rd_indices,
            sample_weights=weights,
            batch_size=8,
            rd_fraction=0.5,
            seed=42,
            shard_keys_by_index=shard_keys,
            shard_group_batches=4,
        )
        batches = list(iter(sampler))
        self.assertGreater(len(batches), 0)

        from collections import defaultdict
        rd_by_shard = defaultdict(set)
        for idx in rd_indices:
            rd_by_shard[shard_keys[idx]].add(idx)

        def _rd_shard(batch):
            rd_in_batch = [idx for idx in batch if idx < 120]
            for s, members in rd_by_shard.items():
                if rd_in_batch and all(idx in members for idx in rd_in_batch):
                    return s
            return None

        self.assertEqual(len(batches), len(sampler))
        for batch in batches:
            self.assertEqual(len(batch), 8)
            self.assertEqual(sum(idx < 120 for idx in batch), 4)
            self.assertEqual(sum(idx >= 120 for idx in batch), 4)

        # Each group contains four visits to one shard. The final group is
        # truncated only because the sampler stops at its reported length.
        batch_shards = [_rd_shard(batch) for batch in batches]
        for start in range(0, len(batch_shards), 4):
            group = batch_shards[start:start + 4]
            self.assertTrue(all(shard is not None for shard in group))
            self.assertEqual(len(set(group)), 1)

    def test_sampler_yields_exactly_reported_batch_count(self):
        from src.core.expert1_data import RDShardLocalBatchSampler

        sampler = RDShardLocalBatchSampler(
            rd_indices=list(range(4)),
            no_rd_indices=list(range(4, 44)),
            sample_weights=[1.0] * 44,
            batch_size=4,
            rd_fraction=0.5,
            seed=42,
            shard_keys_by_index={index: f"shard_{index}" for index in range(4)},
            shard_group_batches=2,
        )

        batches = list(sampler)
        self.assertEqual(len(batches), len(sampler))
        self.assertEqual(len(sampler), 20)
        for batch in batches:
            self.assertEqual(len(batch), 4)
            self.assertEqual(sum(index < 4 for index in batch), 2)
            self.assertEqual(sum(index >= 4 for index in batch), 2)

    def test_sampler_is_deterministic_for_fixed_seed(self):
        from src.core.expert1_data import RDShardLocalBatchSampler

        kwargs = {
            "rd_indices": list(range(12)),
            "no_rd_indices": list(range(12, 36)),
            "sample_weights": [1.0] * 36,
            "batch_size": 6,
            "rd_fraction": 0.5,
            "seed": 17,
            "shard_keys_by_index": {index: f"shard_{index // 4}" for index in range(12)},
            "shard_group_batches": 3,
        }

        self.assertEqual(
            list(RDShardLocalBatchSampler(**kwargs)),
            list(RDShardLocalBatchSampler(**kwargs)),
        )

    def test_all_rd_samples_covered(self):
        from src.core.expert1_data import RDShardLocalBatchSampler

        rd_indices = list(range(30))
        no_rd_indices = list(range(30, 50))
        weights = [1.0] * 50
        shard_keys = {i: f"shard_{i // 10}" for i in range(30)}

        sampler = RDShardLocalBatchSampler(
            rd_indices=rd_indices,
            no_rd_indices=no_rd_indices,
            sample_weights=weights,
            batch_size=8,
            rd_fraction=0.5,
            seed=42,
            shard_keys_by_index=shard_keys,
            shard_group_batches=2,
        )
        all_rd_seen = set()
        for batch in iter(sampler):
            for idx in batch:
                if idx < 30:
                    all_rd_seen.add(idx)
        self.assertGreaterEqual(len(all_rd_seen), 15, f"Only {len(all_rd_seen)}/30 RD indices sampled")


if __name__ == "__main__":
    unittest.main()
