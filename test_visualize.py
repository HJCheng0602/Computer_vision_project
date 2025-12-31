import torch
from torch.utils.data import DataLoader
import numpy as np

# 如果你的代码在一个单独的文件中（例如 dataset.py），请取消注释并导入
from utils.FragmentDataset import FragmentDataset 

def simple_transform(numpy_array):
    """
    转换函数：
    1. 将 numpy 转为 Tensor
    2. 增加 Channel 维度: (32, 32, 32) -> (1, 32, 32, 32)
    3. 转为 float32 (神经网络标准类型)
    """
    tensor = torch.from_numpy(numpy_array).float()
    return tensor.unsqueeze(0) # Add channel dim

def main():
    print("--- 开始 FragmentDataset 测试 ---")
    
    # 1. 配置路径 (请修改为你真实的 vox 文件夹路径)
    # 假设你的目录结构是: /data/shapenet/train/02843684/1a2b3c.../model.vox
    vox_path = 'data/processed_data/' # 示例路径，请替换
    vox_type = 'train' # 测试 'train' 模式
    
    # 2. 实例化 Dataset
    # 注意：这里传入 simple_transform 以适配 PyTorch 格式
    try:
        dataset = FragmentDataset(
            vox_path=vox_path, 
            vox_type=vox_type, 
            dim_size=64, 
            transform=simple_transform
        )
        print(f"成功初始化 Dataset. 找到样本数: {len(dataset)}")
    except Exception as e:
        print(f"初始化 Dataset 失败，请检查路径。错误信息: {e}")
        return

    if len(dataset) == 0:
        print("警告: 未找到任何 .vox 文件，请检查 vox_path 和 vox_type 设置。")
        return

    # 3. 单样本测试 (__getitem__)
    print("\n[1] 单样本测试:")
    try:
        frag, full_vox = dataset[0] # 获取第一个样本
        
        print(f"  Fragment Shape: {frag.shape} (预期: [1, 32, 32, 32])")
        print(f"  Full Vox Shape: {full_vox.shape} (预期: [1, 32, 32, 32])")
        print(f"  Data Type: {frag.dtype}")
        
        # 理论检查: 碎片应该是整体的一个子集
        # 注意：因为经过了 transform 变成了 float tensor，我们比较数值
        overlap = (frag * full_vox).sum()
        frag_sum = frag.sum()
        
        # 简单验证逻辑：碎片的体素应该都在完整物体内 (overlap 应该等于 frag_sum)
        # 容差设为 0.1 避免浮点误差
        if abs(overlap - frag_sum) < 0.1:
            print("  逻辑验证通过: Fragment 是 Full Voxel 的子集。")
        else:
            print(f"  逻辑警告: Fragment 超出了 Full Voxel 范围! (Overlap: {overlap}, Frag: {frag_sum})")
            
    except Exception as e:
        print(f"单样本加载失败: {e}")
        import traceback
        traceback.print_exc()

    # 4. DataLoader 批次测试
    print("\n[2] DataLoader 批次测试:")
    test_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    
    try:
        # 获取一个 Batch
        batch_frag, batch_vox = next(iter(test_loader))
        
        print(f"  Batch Fragment Shape: {batch_frag.shape} (预期: [4, 1, 32, 32, 32])")
        print(f"  Batch Full Vox Shape: {batch_vox.shape} (预期: [4, 1, 32, 32, 32])")
        
        if batch_frag.shape[0] == 4:
            print("  DataLoader 测试通过！数据维度符合 3D CNN 输入要求。")
            
    except Exception as e:
        print(f"DataLoader 测试失败: {e}")
        print("提示: 检查是否安装了 pyvox，或者 .vox 文件是否损坏。")

if __name__ == "__main__":
    main()