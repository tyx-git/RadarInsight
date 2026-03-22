import json
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


class MockDataGenerator:
    """生成用于联调和训练冒烟测试的多类模拟雷达数据。"""

    def __init__(self, session_id: str = "mock_session_20260105"):
        self.session_id = session_id
        self.session_dir = Path(f"./data/sessions/{session_id}")
        self.radar_dir = self.session_dir / "radar"
        self.track_dir = self.radar_dir / "track"
        self.rd_dir = self.radar_dir / "rd"
        self.truth_dir = self.session_dir / "truth"
        self.gnss_dir = self.truth_dir / "gnss"

        for directory in [self.track_dir, self.rd_dir, self.gnss_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _build_targets() -> list[dict[str, float | int | str]]:
        return [
            {
                "id": 1,
                "label": "human",
                "base_r": 110.0,
                "base_v": 1.4,
                "base_az": 24.0,
                "range_amp": 1.5,
                "az_amp": 1.0,
                "speed_amp": 0.4,
                "snr_base": 18.0,
                "snr_amp": 2.5,
                "micro_type": "swing",
            },
            {
                "id": 2,
                "label": "vehicle",
                "base_r": 520.0,
                "base_v": 15.0,
                "base_az": 41.0,
                "range_amp": 2.0,
                "az_amp": 0.5,
                "speed_amp": 0.8,
                "snr_base": 28.0,
                "snr_amp": 1.5,
                "micro_type": "engine",
            },
            {
                "id": 3,
                "label": "drone",
                "base_r": 320.0,
                "base_v": 5.2,
                "base_az": 57.0,
                "range_amp": 3.5,
                "az_amp": 2.2,
                "speed_amp": 1.2,
                "snr_base": 22.0,
                "snr_amp": 3.0,
                "micro_type": "rotor",
            },
            {
                "id": 4,
                "label": "bird",
                "base_r": 220.0,
                "base_v": 8.0,
                "base_az": 68.0,
                "range_amp": 6.0,
                "az_amp": 3.5,
                "speed_amp": 1.8,
                "snr_base": 16.5,
                "snr_amp": 3.5,
                "micro_type": "flap",
            },
            {
                "id": 5,
                "label": "bicycle",
                "base_r": 170.0,
                "base_v": 6.5,
                "base_az": 33.0,
                "range_amp": 2.8,
                "az_amp": 1.4,
                "speed_amp": 0.9,
                "snr_base": 21.0,
                "snr_amp": 2.0,
                "micro_type": "pedal",
            },
        ]

    @staticmethod
    def _kinematics(target: dict[str, float | int | str], time_s: float) -> tuple[float, float, float, float]:
        target_id = int(target["id"])
        phase = target_id * 0.7
        range_m = (
            float(target["base_r"])
            + float(target["base_v"]) * time_s
            + float(target["range_amp"]) * np.sin(0.18 * time_s + phase)
            + np.random.normal(0, 0.35)
        )
        azimuth_deg = (
            float(target["base_az"])
            + float(target["az_amp"]) * np.sin(0.11 * time_s + phase * 0.5)
            + np.random.normal(0, 0.2)
        )
        radial_speed_mps = (
            float(target["base_v"])
            + float(target["speed_amp"]) * np.cos(0.23 * time_s + phase)
            + np.random.normal(0, 0.08)
        )
        snr_db = (
            float(target["snr_base"])
            + float(target["snr_amp"]) * np.cos(0.15 * time_s + phase)
            + np.random.normal(0, 0.6)
        )
        return range_m, azimuth_deg, radial_speed_mps, snr_db

    @staticmethod
    def _add_energy(frame: np.ndarray, r_idx: int, v_idx: int, amplitude: float) -> None:
        if not (0 <= r_idx < frame.shape[0] and 0 <= v_idx < frame.shape[1]):
            return
        frame[r_idx, v_idx] += amplitude
        for dr in (-1, 0, 1):
            for dv in (-1, 0, 1):
                rr = r_idx + dr
                vv = v_idx + dv
                if 0 <= rr < frame.shape[0] and 0 <= vv < frame.shape[1]:
                    frame[rr, vv] += amplitude * 0.18

    def _add_rd_signature(
        self,
        frame: np.ndarray,
        target: dict[str, float | int | str],
        time_s: float,
        range_res: float,
        doppler_res: float,
        zero_doppler_idx: int,
    ) -> None:
        range_m, _, radial_speed_mps, _ = self._kinematics(target, time_s)
        r_idx = int(range_m / range_res)
        v_idx = int(zero_doppler_idx + radial_speed_mps / doppler_res)
        micro_type = str(target["micro_type"])

        if micro_type == "swing":
            offsets = (-1, 0, 1)
            amplitudes = (8.0, 20.0, 8.0)
        elif micro_type == "engine":
            offsets = (-2, -1, 0, 1, 2)
            amplitudes = (6.0, 12.0, 28.0, 12.0, 6.0)
        elif micro_type == "rotor":
            offsets = (-8, -4, 0, 4, 8)
            amplitudes = (10.0, 16.0, 24.0, 16.0, 10.0)
        elif micro_type == "flap":
            dynamic_offset = int(round(3 * np.sin(0.6 * time_s)))
            offsets = (-5 + dynamic_offset, -2, 0, 2, 5 + dynamic_offset)
            amplitudes = (7.0, 12.0, 18.0, 12.0, 7.0)
        else:
            offsets = (-3, 0, 3)
            amplitudes = (8.0, 18.0, 8.0)

        for offset, amplitude in zip(offsets, amplitudes):
            self._add_energy(frame, r_idx, v_idx + offset, amplitude)

        for range_offset in (-2, -1, 1, 2):
            self._add_energy(frame, r_idx + range_offset, v_idx, 4.0)

    def generate_track_and_truth_data(self, num_frames: int = 100):
        """生成 5 类目标的轨迹数据与 GNSS 真值。"""
        track_file = self.track_dir / "track_data.csv"
        targets = self._build_targets()
        track_data: list[dict[str, float | int]] = []
        gnss_by_label: dict[str, list[dict[str, float | str]]] = {str(target["label"]): [] for target in targets}
        start_time = time.time()

        for frame_id in range(num_frames):
            timestamp = start_time + frame_id * 0.5
            time_s = frame_id * 0.5
            for target in targets:
                range_m, azimuth_deg, radial_speed_mps, snr_db = self._kinematics(target, time_s)
                track_data.append(
                    {
                        "t": timestamp,
                        "frame_id": frame_id,
                        "track_id": int(target["id"]),
                        "range_m": round(range_m, 2),
                        "azimuth_deg": round(azimuth_deg, 2),
                        "radial_speed_mps": round(radial_speed_mps, 2),
                        "snr_db": round(snr_db, 1),
                    }
                )

                gnss_by_label[str(target["label"])].append(
                    {
                        "t": timestamp + 0.08,
                        "range_m": round(range_m + np.random.normal(1.2, 0.25), 2),
                        "azimuth_deg": round(azimuth_deg + np.random.normal(0.4, 0.08), 2),
                        "radial_speed_mps": round(radial_speed_mps + np.random.normal(0.15, 0.05), 2),
                        "label": str(target["label"]),
                    }
                )

        pd.DataFrame(track_data).to_csv(track_file, index=False)
        for target in targets:
            label = str(target["label"])
            gnss_file = self.gnss_dir / f"{label}_{int(target['id'])}.csv"
            pd.DataFrame(gnss_by_label[label]).to_csv(gnss_file, index=False)

        print(f"[*] 生成多类 Track 数据: {track_file}")
        print(f"[*] 生成多类 GNSS 真值文件: {self.gnss_dir}")

    def generate_rd_data(self, num_frames: int = 100):
        """生成 5 类目标的多类距离-多普勒数据。"""
        rd_file = self.rd_dir / "rd_data.h5"
        num_range = 256
        num_doppler = 128
        targets = self._build_targets()

        range_res = 8.0
        doppler_res = 0.5
        zero_doppler_idx = 64

        with h5py.File(rd_file, "w") as handle:
            dataset = handle.create_dataset("rd_frames", (num_frames, num_range, num_doppler), dtype="f4")

            for frame_id in range(num_frames):
                time_s = frame_id * 0.5
                rd_frame = np.random.normal(0, 0.9, (num_range, num_doppler)).astype(np.float32)
                for target in targets:
                    self._add_rd_signature(
                        frame=rd_frame,
                        target=target,
                        time_s=time_s,
                        range_res=range_res,
                        doppler_res=doppler_res,
                        zero_doppler_idx=zero_doppler_idx,
                    )
                dataset[frame_id] = rd_frame

            handle.attrs["range_res_m"] = range_res
            handle.attrs["doppler_res_mps"] = doppler_res
            handle.attrs["zero_doppler_idx"] = zero_doppler_idx

        print(f"[*] 生成多类 RD 数据: {rd_file}")

    def generate_meta_session(self):
        """生成会话元数据。"""
        targets = [str(target["label"]) for target in self._build_targets()]
        meta = {
            "session_id": self.session_id,
            "site": {"name": "Lab_Simulation", "radar_install_height_m": 5.0},
            "time": {"start_local": time.strftime("%Y-%m-%d %H:%M:%S")},
            "radar_config": {"track_rate_hz": 2, "range_max_m": 6000},
            "mock_profile": {"labels": targets, "num_classes": len(targets)},
        }
        meta_file = self.session_dir / "meta_session.json"
        with open(meta_file, "w", encoding="utf-8") as file:
            json.dump(meta, file, indent=4, ensure_ascii=False)
        print(f"[*] 生成模拟会话元数据: {meta_file}")


if __name__ == "__main__":
    generator = MockDataGenerator()
    generator.generate_meta_session()
    generator.generate_track_and_truth_data()
    generator.generate_rd_data()
    print("[*] 模拟环境数据生成完毕。")
