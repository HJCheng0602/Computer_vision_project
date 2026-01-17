import os
import glob
import numpy as np
import argparse
from pyvox.parser import VoxParser
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import shutil

def parse_and_process(file_info):
    src_path, dst_path = file_info

    try:
        parser = VoxParser(src_path)
        model = parser.parse()
        vox = model.to_dense() 
        
        d, h, w = vox.shape
        
        canvas_64 = np.zeros((64, 64, 64), dtype=np.uint8)
        d_end, h_end, w_end = min(d, 64), min(h, 64), min(w, 64)
        
        offset_d = (64 - d_end) // 2
        offset_h = (64 - h_end) // 2
        offset_w = (64 - w_end) // 2
        
        canvas_64[offset_d:offset_d + d_end, 
                  offset_h:offset_h + h_end, 
                  offset_w:offset_w + w_end] = vox[:d_end, :h_end, :w_end]
        
        vox_32 = canvas_64
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        
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

    search_pattern = os.path.join(args.src_dir, "**", "*.vox")
    files = glob.glob(search_pattern, recursive=True)
    
    print(f"Found {len(files)} .vox files in {args.src_dir}")
    
    tasks = []
    for src_path in files:
        rel_path = os.path.relpath(src_path, args.src_dir)
        dst_path = os.path.join(args.dst_dir, rel_path)
        dst_path = os.path.splitext(dst_path)[0] + '.npy'
        
        tasks.append((src_path, dst_path))
    print(f"Starting processing with {args.workers} workers...")
    
    error_logs = []
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = list(tqdm(executor.map(parse_and_process, tasks), total=len(tasks)))

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