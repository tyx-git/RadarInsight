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

    def slice_rd_clips(self, track_clips: List[Dict[str, Any]], range_win: int = 16, doppler_win: int = 16):
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
            range_res = float(f_rd.attrs['range_res_m'])
            doppler_res = float(f_rd.attrs['doppler_res_mps'])
            zero_doppler_idx = int(f_rd.attrs['zero_doppler_idx'])
            
            df_track = pd.read_csv(self.track_csv)
            
            # 创建 RD Clip 保存文件
            rd_clip_file = self.session_dir / "radar/rd/rd_clips.h5"
            with h5py.File(rd_clip_file, 'w') as f_out:
                for clip in track_clips:
                    # 获取该时间段内的轨迹点
                    window_df = df_track[(df_track['track_id'] == clip['track_id']) & 
                                         (df_track['t'] >= clip['t0']) & 
                                         (df_track['t'] < clip['t1'])]
                    
                    # 为每个样本存储一组 RD 子图序列
                    sample_rd_seq: list[np.ndarray] = []
                    for _, row in window_df.iterrows():
                        frame_idx = int(row['frame_id'])
                        if frame_idx >= rd_frame_count:
                            continue
                        
                        # 计算目标在 RD 上的坐标索引
                        r_idx = int(row['range_m'] / range_res)
                        v_idx = int(zero_doppler_idx + row['radial_speed_mps'] / doppler_res)
                        
                        # 确保 r_idx and v_idx within matrix bounds
                        if r_idx < 0 or r_idx >= rd_range_bins or v_idx < 0 or v_idx >= rd_doppler_bins:
                            continue

                        # 边界裁剪 (计算在原图中的实际范围)
                        r_start = r_idx - range_win
                        r_end = r_idx + range_win
                        v_start = v_idx - doppler_win
                        v_end = v_idx + doppler_win
                        
                        # 在原图中的有效部分
                        r_start_valid = max(0, r_start)
                        r_end_valid = min(rd_range_bins, r_end)
                        v_start_valid = max(0, v_start)
                        v_end_valid = min(rd_doppler_bins, v_end)
                        
                        # 提取子矩阵
                        crop_valid = np.asarray(
                            rd_frames[frame_idx, r_start_valid:r_end_valid, v_start_valid:v_end_valid],
                            dtype=np.float32,
                        )
                        
                        # 确保裁剪尺寸一致 (处理边缘情况，使目标保持在中心)
                        target_h, target_w = range_win * 2, doppler_win * 2
                        pad_crop = np.zeros((target_h, target_w), dtype=crop_valid.dtype)
                        
                        # 计算在 pad_crop 中的起始位置
                        r_offset = r_start_valid - r_start
                        v_offset = v_start_valid - v_start
                        
                        # 填充
                        h, w = crop_valid.shape
                        pad_crop[r_offset:r_offset+h, v_offset:v_offset+w] = crop_valid
                        
                        sample_rd_seq.append(pad_crop)
                    
                    # 保存 RD Clip 序列
                    if sample_rd_seq:
                        # 将列表转换为 numpy 数组: (num_frames_in_clip, H, W)
                        rd_array = np.stack(sample_rd_seq, axis=0)
                        f_out.create_dataset(f"rd_clip_{clip['sample_id']}", data=rd_array, compression="gzip")

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
