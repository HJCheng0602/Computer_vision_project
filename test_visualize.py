import torch
import os
import numpy as np
# 假设你已经把刚才那几个 matplotlib 的 plot 函数放在了同一个文件或者 utils 里
# from utils.visualization import plot, plot_join 

def test_simple(generator, test_dataloader, save_dir, device):
    """
    极简测试函数：只取一个样本，生成并保存图片。
    用于快速查看模型到底学会了没有。
    """
    print("--- 正在生成测试样本 (Snapshot) ---")
    generator.eval()
    
    data_iter = iter(test_dataloader)
    fragments, real_voxels = next(data_iter)
    
    frag_input = fragments[0].unsqueeze(0).unsqueeze(0).float().to(device)
    
    with torch.no_grad():
        output = generator(frag_input)
        fake_voxel = (torch.sigmoid(output) > 0.5).float().cpu().numpy()[0, 0]
    
    real_voxel_np = real_voxels[0].numpy()
    
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"保存生成的陶艺: {save_dir}/snapshot_generated.png")
    plot(fake_voxel, save_dir, filename="snapshot_generated.png")
    
    print(f"保存对比图: {save_dir}/snapshot_comparison.png")
    
    plot_join(real_voxel_np * 1, fake_voxel * 2, save_dir, filename="snapshot_comparison.png")
    
    generator.train() 
    print("--- 测试样本生成完毕 ---")