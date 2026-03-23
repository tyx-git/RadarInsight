import pandas as pd
import numpy as np
import h5py
import json
from pathlib import Path
from typing import List, Dict, Any

class DataSlicer:
    """
    雷达数据切片类，将连续的轨迹和 RD 矩阵切分为 8s 左右的片段 (Track/RD Clip)。
    符合 toRead.md 4.3 节定义。
    """
    def __init__(self, session_dir: str):
        self.session_dir = Path(session_dir)
        self.track_csv = self.session_dir / "radar/track/track_data.csv"
        self.rd_h5 = self.session_dir / "radar/rd/rd_data.h5"
        self.output_dir = self.session_dir / "labels"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _normalize_track_id(track_id: Any) -> int:
        if hasattr(track_id, "item"):
            track_id = track_id.item()
        return int(str(track_id))

    @staticmethod
    def _nearest_index(axis_values: np.ndarray, target_value: float) -> int:
        return int(np.argmin(np.abs(axis_values - float(target_value))))

    @staticmethod
    def _smooth_indices(indices: list[int], window_size: int = 5) -> list[int]:
        if not indices:
            return []
        if window_size <= 1:
            return list(indices)
        radius = window_size // 2
        smoothed: list[int] = []
        for index in range(len(indices)):
            left = max(0, index - radius)
            right = min(len(indices), index + radius + 1)
            smoothed.append(int(np.median(indices[left:right])))
        return smoothed

    @staticmethod
    def _weighted_peak_index(local_amplitude: np.ndarray, peak_guard_ratio: float) -> tuple[int, int] | None:
        if local_amplitude.size == 0:
            return None
        local_peak_flat_index = int(local_amplitude.argmax())
        local_peak_index = np.unravel_index(local_peak_flat_index, local_amplitude.shape)
        local_peak_value = float(local_amplitude[local_peak_index])
        local_background = float(np.median(local_amplitude))
        if local_peak_value <= 0:
            return None
        if local_peak_value < max(local_background * peak_guard_ratio, 1e-6):
            return None

        support_threshold = local_peak_value * 0.7
        support = np.where(local_amplitude >= support_threshold, local_amplitude, 0.0)
        support_sum = float(support.sum())
        if support_sum <= 0:
            return int(local_peak_index[0]), int(local_peak_index[1])

        row_index_grid, col_index_grid = np.indices(local_amplitude.shape)
        weighted_row = int(round(float((support * row_index_grid).sum() / support_sum)))
        weighted_col = int(round(float((support * col_index_grid).sum() / support_sum)))
        return weighted_row, weighted_col

    def slice_tracks(self, window_size_s: float = 8.0, overlap: float = 0.5) -> List[Dict[str, Any]]:
        """
        按轨迹 ID (track_id) 将连续轨迹切分为片段。
        
        :param window_size_s: 窗口时长（秒）
        :param overlap: 窗口重叠比例
        :return: 轨迹片段元数据列表
        """
        if not self.track_csv.exists():
            print(f"[!] {self.track_csv} 不存在。")
            return []

        df = pd.read_csv(self.track_csv)
        clips = []
        
        # 步长计算
        step_s = window_size_s * (1 - overlap)
        
        # 按 track_id 分组
        for track_id, group in df.groupby('track_id'):
            track_id_int = self._normalize_track_id(track_id)
            group = group.sort_values('t')
            t_start = group['t'].min()
            t_end = group['t'].max()
            
            # 窗口滑动
            current_t0 = t_start
            while current_t0 + window_size_s <= t_end:
                current_t1 = current_t0 + window_size_s
                
                # 获取时间窗内的样本
                window_df = group[(group['t'] >= current_t0) & (group['t'] < current_t1)]
                
                if not window_df.empty:
                    # 样本 ID 唯一化: {session}_{track_id}_{t0}
                    sample_id = f"{self.session_dir.name}_{track_id_int}_{int(current_t0)}"
                    
                    clip_meta = {
                        "sample_id": sample_id,
                        "type": "trackclip",
                        "t0": current_t0,
                        "t1": current_t1,
                        "track_id": track_id_int,
                        "num_frames": len(window_df),
                        "mean_range": round(float(window_df['range_m'].mean()), 2),
                        "mean_speed": round(float(window_df['radial_speed_mps'].mean()), 2)
                    }
                    clips.append(clip_meta)
                
                current_t0 += step_s
                
        # 保存轨迹片段元数据到 JSONL
        labels_file = self.output_dir / "labels_trackclip.jsonl"
        with open(labels_file, 'w', encoding='utf-8') as f:
            for clip in clips:
                f.write(json.dumps(clip) + "\n")
                
        print(f"[*] 生成了 {len(clips)} 个轨迹片段。保存至: {labels_file}")
        return clips

    def slice_rd_clips(
        self,
        track_clips: List[Dict[str, Any]],
        range_win: int = 16,
        doppler_win: int = 16,
        smoothing_window: int = 5,
        refine_range_radius: int = 6,
        refine_doppler_radius: int = 6,
        clip_range_margin: int = 10,
        clip_doppler_margin: int = 6,
        peak_guard_ratio: float = 1.2,
        max_range_shift: int = 4,
        max_doppler_shift: int = 3,
    ):
        """
        根据轨迹片段的时间与位置，从 RD 矩阵中裁剪目标局部的子图。
        
        :param track_clips: 轨迹片段列表
        :param range_win: 距离门裁剪半径 (±dR)
        :param doppler_win: 速度门裁剪半径 (±dV)
        """
        if not self.rd_h5.exists():
            print(f"[!] {self.rd_h5} 不存在。")
            return

        with h5py.File(self.rd_h5, 'r') as f_rd:
            rd_frames = f_rd.get('rd_frames')
            if not isinstance(rd_frames, h5py.Dataset):
                raise TypeError("Expected 'rd_frames' to be an HDF5 dataset.")

            if len(rd_frames.shape) != 3:
                raise ValueError("Expected 'rd_frames' to have shape (frames, range, doppler).")

            rd_frame_count = int(rd_frames.shape[0])
            rd_range_bins = int(rd_frames.shape[1])
            rd_doppler_bins = int(rd_frames.shape[2])
            if 'range_axis_m' in f_rd:
                range_axis_m = np.asarray(f_rd['range_axis_m'][:], dtype=np.float32)
            else:
                range_res = float(f_rd.attrs['range_res_m'])
                range_axis_m = np.arange(rd_range_bins, dtype=np.float32) * range_res
            if 'doppler_axis_mps' in f_rd:
                doppler_axis_mps = np.asarray(f_rd['doppler_axis_mps'][:], dtype=np.float32)
            else:
                doppler_res = float(f_rd.attrs['doppler_res_mps'])
                zero_doppler_idx = int(f_rd.attrs['zero_doppler_idx'])
                doppler_axis_mps = (np.arange(rd_doppler_bins, dtype=np.float32) - zero_doppler_idx) * doppler_res
            
            df_track = pd.read_csv(self.track_csv)
            
            # 创建 RD Clip 保存文件
            rd_clip_file = self.session_dir / "radar/rd/rd_clips.h5"
            alignment_file = self.session_dir / "radar/rd/rd_alignment_diagnostics.jsonl"
            with h5py.File(rd_clip_file, 'w') as f_out:
                with open(alignment_file, 'w', encoding='utf-8') as alignment_handle:
                    for clip in track_clips:
                        window_df = df_track[(df_track['track_id'] == clip['track_id']) & 
                                             (df_track['t'] >= clip['t0']) & 
                                             (df_track['t'] < clip['t1'])]
                        sample_rd_seq: list[np.ndarray] = []
                        predicted_rows: list[pd.Series] = []
                        predicted_r_indices: list[int] = []
                        predicted_v_indices: list[int] = []
                        for _, row in window_df.iterrows():
                            frame_idx = int(row['frame_id'])
                            if frame_idx >= rd_frame_count:
                                continue

                            r_idx = self._nearest_index(range_axis_m, float(row['range_m']))
                            v_idx = self._nearest_index(doppler_axis_mps, float(row['radial_speed_mps']))
                            predicted_rows.append(row)
                            predicted_r_indices.append(r_idx)
                            predicted_v_indices.append(v_idx)

                        if not predicted_rows:
                            continue

                        smoothed_r = self._smooth_indices(predicted_r_indices, window_size=smoothing_window)
                        smoothed_v = self._smooth_indices(predicted_v_indices, window_size=smoothing_window)
                        clip_center_r = int(np.median(smoothed_r))
                        clip_center_v = int(np.median(smoothed_v))
                        clip_alignment_records: list[dict[str, Any]] = []

                        for frame_offset, (row, r_idx, v_idx) in enumerate(zip(predicted_rows, smoothed_r, smoothed_v)):
                            frame_idx = int(row['frame_id'])

                            r_idx = int(np.clip(r_idx, clip_center_r - clip_range_margin, clip_center_r + clip_range_margin))
                            v_idx = int(np.clip(v_idx, clip_center_v - clip_doppler_margin, clip_center_v + clip_doppler_margin))
                            base_r_idx = r_idx
                            base_v_idx = v_idx

                            refine_r_start = max(0, r_idx - refine_range_radius)
                            refine_r_end = min(rd_range_bins, r_idx + refine_range_radius + 1)
                            refine_v_start = max(0, v_idx - refine_doppler_radius)
                            refine_v_end = min(rd_doppler_bins, v_idx + refine_doppler_radius + 1)

                            local_amplitude = np.abs(
                                np.asarray(rd_frames[frame_idx, refine_r_start:refine_r_end, refine_v_start:refine_v_end])
                            )
                            weighted_peak_index = self._weighted_peak_index(local_amplitude, peak_guard_ratio=peak_guard_ratio)
                            if weighted_peak_index is not None:
                                candidate_r_idx = refine_r_start + int(weighted_peak_index[0])
                                candidate_v_idx = refine_v_start + int(weighted_peak_index[1])
                                delta_r = int(np.clip(candidate_r_idx - base_r_idx, -max_range_shift, max_range_shift))
                                delta_v = int(np.clip(candidate_v_idx - base_v_idx, -max_doppler_shift, max_doppler_shift))
                                r_idx = base_r_idx + delta_r
                                v_idx = base_v_idx + delta_v

                            r_start = r_idx - range_win
                            r_end = r_idx + range_win
                            v_start = v_idx - doppler_win
                            v_end = v_idx + doppler_win
                            r_start_valid = max(0, r_start)
                            r_end_valid = min(rd_range_bins, r_end)
                            v_start_valid = max(0, v_start)
                            v_end_valid = min(rd_doppler_bins, v_end)
                            crop_valid = np.asarray(
                                rd_frames[frame_idx, r_start_valid:r_end_valid, v_start_valid:v_end_valid],
                                dtype=rd_frames.dtype,
                            )
                            target_h, target_w = range_win * 2, doppler_win * 2
                            pad_crop = np.zeros((target_h, target_w), dtype=rd_frames.dtype)
                            r_offset = r_start_valid - r_start
                            v_offset = v_start_valid - v_start
                            h, w = crop_valid.shape
                            pad_crop[r_offset:r_offset+h, v_offset:v_offset+w] = crop_valid

                            sample_rd_seq.append(pad_crop)
                            clip_alignment_records.append(
                                {
                                    "frame_id": frame_idx,
                                    "predicted_range_idx": int(predicted_r_indices[frame_offset]),
                                    "predicted_doppler_idx": int(predicted_v_indices[frame_offset]),
                                    "smoothed_range_idx": int(base_r_idx),
                                    "smoothed_doppler_idx": int(base_v_idx),
                                    "refined_range_idx": int(r_idx),
                                    "refined_doppler_idx": int(v_idx),
                                    "range_offset_bins": int(r_idx - base_r_idx),
                                    "doppler_offset_bins": int(v_idx - base_v_idx),
                                }
                            )

                        if sample_rd_seq:
                            rd_array = np.stack(sample_rd_seq, axis=0)
                            f_out.create_dataset(f"rd_clip_{clip['sample_id']}", data=rd_array, compression="gzip")
                            alignment_payload = {
                                "sample_id": clip["sample_id"],
                                "track_id": int(clip["track_id"]),
                                "clip_center_range_idx": clip_center_r,
                                "clip_center_doppler_idx": clip_center_v,
                                "range_window": int(range_win),
                                "doppler_window": int(doppler_win),
                                "frames": clip_alignment_records,
                            }
                            alignment_handle.write(json.dumps(alignment_payload, ensure_ascii=False) + "\n")

        print(f"[*] RD 片段提取完毕。")

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.abspath("."))
    # 示例用法
    slicer = DataSlicer("./data/sessions/test_session")
    # slicer.slice_tracks()
    # slicer.slice_rd_clips([])
    print("[*] Data slicer initialized.")
