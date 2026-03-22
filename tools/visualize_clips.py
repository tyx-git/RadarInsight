import h5py
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse

def visualize_rd_clips(session_dir: str, num_samples: int = 5):
    """
    可视化 HDF5 文件中的 RD 片段。
    """
    session_path = Path(session_dir)
    rd_clip_file = session_path / "radar/rd/rd_clips.h5"
    
    if not rd_clip_file.exists():
        print(f"[!] 未找到 RD 片段文件: {rd_clip_file}")
        return

    with h5py.File(rd_clip_file, 'r') as f:
        clip_keys = list(f.keys())
        if not clip_keys:
            print("[!] HDF5 文件中没有数据。")
            return
            
        print(f"[*] 发现 {len(clip_keys)} 个 RD 片段。")
        
        # 限制显示数量
        display_keys = clip_keys[:num_samples]
        
        for key in display_keys:
            data = f[key][:] # (num_frames, H, W)
            num_frames = data.shape[0]
            
            # 绘制中间一帧
            mid_idx = num_frames // 2
            frame = data[mid_idx]
            
            plt.figure(figsize=(6, 5))
            plt.imshow(10 * np.log10(np.abs(frame) + 1e-6), cmap='jet', aspect='auto')
            plt.colorbar(label='Amplitude (dB)')
            plt.title(f"RD Clip: {key} (Frame {mid_idx})")
            plt.xlabel("Doppler Index")
            plt.ylabel("Range Index")
            
            # 保存预览图
            output_img = session_path / f"qc/preview_{key}.png"
            plt.savefig(output_img)
            plt.close()
            print(f"[+] 预览图已保存: {output_img}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RD 片段可视化工具")
    parser.add_argument("session_dir", help="会话目录路径")
    parser.add_argument("--num", type=int, default=3, help="显示样本数量")
    args = parser.parse_args()
    
    visualize_rd_clips(args.session_dir, args.num)
