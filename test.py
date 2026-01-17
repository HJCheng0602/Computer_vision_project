## Here you may implement the evaluation method and may call some necessary modules from utils.model_utils.py
## Derive the test function by yourself and implement proper metric such as Dice similarity coeffcient (DSC)[4];
# Jaccard distance[5] and Mean squared error (MSE), etc. following the handout in model_utilss.py

import numpy as np
import torch
from torch import optim
from torch.utils import data
from torch import nn
from utils.FragmentDataset import FragmentDataset
from utils.model import Generator_Init, Generator_res, Generator_wider
import os

def calculate_dice_score(pred, target, smooth=1e-6):
    pred = (pred > 0.5).float()
    
    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)
    
    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1)
    
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.mean().item()

def calculate_iou(pred, target, threshold=0.5, smooth=1e-6):
    pred_bin = (pred > threshold).float()
    target_bin = (target > threshold).float()
    pred_flat = pred_bin.view(pred_bin.size(0), -1)
    target_flat = target_bin.view(target_bin.size(0), -1)
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean().item()

def test(generator, dataloader, save_dir, device, epoch, writer=None, logdir_name="experiment"):
    generator.eval()  
    save_dir = os.path.join(save_dir, logdir_name)
    os.makedirs(save_dir, exist_ok=True)
    epoch_dir = os.path.join(save_dir, f"epoch_{epoch}")
    os.makedirs(epoch_dir, exist_ok=True)
    
    total_dice = 0.0
    total_mse = 0.0
    total_iou = 0.0
    criterion_mse = nn.MSELoss()
    
    print(f"\n[Test] Starting evaluation for Epoch {epoch}...")
    
    with torch.no_grad(): 
        for i, (fragments, real_voxels) in enumerate(dataloader):
            fragments = fragments.unsqueeze(1).float().to(device)
            real_voxels = real_voxels.float().to(device) 
            if len(real_voxels.shape) == 3: # (D, H, W) -> (1, D, H, W)
                 real_voxels = real_voxels.unsqueeze(1)
            elif len(real_voxels.shape) == 4: # (B, D, H, W) -> (B, 1, D, H, W)
                 real_voxels = real_voxels.unsqueeze(1)

            fake_voxels = generator(fragments)
            
            # 计算 Dice Score
            dice = calculate_dice_score(fake_voxels, real_voxels)
            total_dice += dice
            
            # 计算 MSE
            mse = criterion_mse(fake_voxels, real_voxels).item()
            total_mse += mse
            
            # compute IoU
            iou = calculate_iou(fake_voxels, real_voxels)
            total_iou += iou

            if i < 5: 
                sample_np = (fake_voxels[0, 0].cpu().numpy() > 0.5).astype(np.uint8)
                input_np = fragments[0, 0].cpu().numpy().astype(np.uint8)
                gt_np = real_voxels[0, 0].cpu().numpy().astype(np.uint8)
                
                np.save(os.path.join(epoch_dir, f"batch_{i}_generated.npy"), sample_np)
                np.save(os.path.join(epoch_dir, f"batch_{i}_input.npy"), input_np)
                # np.save(os.path.join(epoch_dir, f"batch_{i}_gt.npy"), gt_np)

    avg_dice = total_dice / len(dataloader)
    avg_mse = total_mse / len(dataloader)
    avg_iou = total_iou / len(dataloader)
    
    print(f"[Test] Epoch {epoch} | Avg Dice Score: {avg_dice:.4f} | Avg MSE: {avg_mse:.4f} | Avg IoU: {avg_iou:.4f}")
    
    if writer:
        writer.add_scalar('Test/Dice_Score', avg_dice, epoch)
        writer.add_scalar('Test/MSE', avg_mse, epoch)
        writer.add_scalar('Test/IoU', avg_iou, epoch)
        
    generator.train() 
    return avg_dice