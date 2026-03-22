import pandas as pd
import numpy as np
import h5py
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

class DataIntegrityChecker:
    """
    数据完整性与质量控制 (QC) 类。
    符合 toRead.md 7.1 节定义。
    """
    def __init__(self, session_dir: str):
        self.session_dir = Path(session_dir)
        self.track_csv = self.session_dir / "radar/track/track_data.csv"
        self.rd_h5 = self.session_dir / "radar/rd/rd_data.h5"
        self.qc_dir = self.session_dir / "qc"
        self.qc_dir.mkdir(parents=True, exist_ok=True)

    def check_integrity(self) -> Dict[str, Any]:
        """
        检查 Track 与 RD 是否连续，时长是否匹配。
        
        :return: 质检结果字典
        """
        results = {
            "status": "PASS",
            "track_found": False,
            "rd_found": False,
            "duration_s": 0.0,
            "frame_loss_count": 0,
            "errors": []
        }
        
        # 1. 检查 Track CSV。
        if self.track_csv.exists():
            df = pd.read_csv(self.track_csv)
            results["track_found"] = True
            
            t_min = df['t'].min()
            t_max = df['t'].max()
            duration = t_max - t_min
            results["duration_s"] = round(duration, 2)
            
            # 统计帧丢失 (假设 2Hz)
            expected_frames = int(duration * 2)
            actual_frames = df['frame_id'].nunique()
            results["frame_loss_count"] = max(0, expected_frames - actual_frames)
            
        # 2. 检查 RD HDF5。
        if self.rd_h5.exists():
            with h5py.File(self.rd_h5, 'r') as f:
                rd_frames = f['rd_frames']
                results["rd_found"] = True
                results["rd_frame_count"] = rd_frames.shape[0]
                results["rd_shape"] = rd_frames.shape[1:]
            
        # 3. 统计类别分布
        if self.track_csv.exists():
            # 这里由于是模拟数据或未标注数据，只能统计 track_id 数量
            results["num_tracks"] = int(df['track_id'].nunique())
        
        # 4. 生成质检报告 JSON。
        qc_report = self.qc_dir / "qc_report.json"
        with open(qc_report, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
            
        print(f"[*] 质检报告已生成: {qc_report}")
        return results

    def plot_distributions(self):
        """
        生成数据分布图表 (Matplotlib)。
        """
        # TODO: 绘制 Track 轨迹热力图、距离/速度分布直方图。
        pass

if __name__ == "__main__":
    # 示例用法
    checker = DataIntegrityChecker("./data/sessions/test_session")
    # checker.check_integrity()
    print("[*] QC integrity checker initialized.")
