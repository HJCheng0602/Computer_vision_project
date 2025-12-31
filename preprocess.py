import os
import glob
import numpy as np
import argparse
from pyvox.parser import VoxParser
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import shutil

def parse_and_process(file_info):
    """
    单个文件的处理逻辑：读取 vox -> 转换 -> 下采样 -> 保存 npy
    """
    src_path, dst_path = file_info

    try:
        # 1. 解析 .vox
        parser = VoxParser(src_path)
        model = parser.parse()
        vox = model.to_dense() # 注意：如果原始模型极其巨大，这里仍可能报错，需要捕获
        
        d, h, w = vox.shape
        
        # 2. 执行你的 Padding/Cropping 逻辑 (对齐到 64x64x64)
        canvas_64 = np.zeros((64, 64, 64), dtype=np.uint8)
        d_end, h_end, w_end = min(d, 64), min(h, 64), min(w, 64)
        
        offset_d = (64 - d_end) // 2
        offset_h = (64 - h_end) // 2
        offset_w = (64 - w_end) // 2
        
        canvas_64[offset_d:offset_d + d_end, 
                  offset_h:offset_h + h_end, 
                  offset_w:offset_w + w_end] = vox[:d_end, :h_end, :w_end]
        
        # 3. 执行你的下采样逻辑 (64 -> 32)
        # 这里的逻辑是将 2x2x2 的块合并为一个，取最大值
        #vox_32 = canvas_64.reshape(32, 2, 32, 2, 32, 2).max(axis=(1, 3, 5))
        
        # preserve the 64 resolution as requested
        vox_32 = canvas_64
        
        # 4. 转为布尔型或极小的整数以节省空间 (0 或 1)
        # vox_32 = (vox_32 > 0).astype(np.uint8)

        # 5. 确保目标文件夹存在
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        
        # 6. 保存为 .npy
        np.save(dst_path, vox_32)
        return True, None

    except Exception as e:
        return False, f"Error processing {src_path}: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description='Preprocess .vox files to .npy')
    parser.add_argument('--src_dir', type=str, default='data/data', help='原始数据目录 (包含 train/test)')
    parser.add_argument('--dst_dir', type=str, default='data/processed_data64', help='输出数据目录')
    parser.add_argument('--workers', type=int, default=8, help='多进程数量 (根据你的CPU核数设置)')
    args = parser.parse_args()

    # 1. 收集所有 .vox 文件路径
    # 假设结构是 src_dir/train/class/*.vox
    search_pattern = os.path.join(args.src_dir, "**", "*.vox")
    files = glob.glob(search_pattern, recursive=True)
    
    print(f"Found {len(files)} .vox files in {args.src_dir}")
    
    tasks = []
    for src_path in files:
        # 构建输出路径：保持相同的目录结构，但后缀改为 .npy
        # 例如: data/data/train/jar/1.vox -> data/processed_data/train/jar/1.npy
        rel_path = os.path.relpath(src_path, args.src_dir)
        dst_path = os.path.join(args.dst_dir, rel_path)
        dst_path = os.path.splitext(dst_path)[0] + '.npy'
        
        tasks.append((src_path, dst_path))

    # 2. 使用多进程并行处理
    # 如果你的某个文件会导致卡死，多进程可能会超时，但通常能处理完其他文件
    print(f"Starting processing with {args.workers} workers...")
    
    error_logs = []
    
    # 使用 tqdm 显示进度条
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = list(tqdm(executor.map(parse_and_process, tasks), total=len(tasks)))

    # 3. 统计结果
    success_count = 0
    for success, msg in results:
        if success:
            success_count += 1
        else:
            error_logs.append(msg)

    print("-" * 30)
    print(f"Processing complete.")
    print(f"Success: {success_count}")
    print(f"Failed: {len(error_logs)}")
    
    if error_logs:
        print("\nErrors occurred in the following files (Check if they are corrupt):")
        for log in error_logs:
            print(log)

if __name__ == "__main__":
    main()