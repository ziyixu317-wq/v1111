import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from pipeline import FlowVortexFusionPipeline
from vortex import calculate_ivd
from data_loader import read_single_vti, read_vti_with_vector

from data_loader import VTIFlowDataset
from vortex import vortex_mae_paper_loss, calculate_iou

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Fused VortexMAE on IVD Masks.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--pretrained_ckpt", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--pos_weight", type=float, default=2.0, help="Positive class weight for paper loss")
    parser.add_argument("--save_dir", type=str, default="./checkpoints_fused_finetune")
    
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    # 自动识别 CUDA 优先
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Dataset
    # Split logic is handled internally in VTIFlowDataset
    train_dataset = VTIFlowDataset(args.data_dir, split="finetune_train", crop_size=128)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    in_chans = train_dataset[0].shape[0]
    
    # 2. Model
    # Patch size aligned with paper (4, 4, 4)
    pipeline = FlowVortexFusionPipeline(mode='segmentation', in_chans=in_chans, patch_size=(4, 4, 4))
    pipeline.to(device)
    
    print(f"Loading pretrained weights from {args.pretrained_ckpt}...")
    ckpt = torch.load(args.pretrained_ckpt, map_location=device)
    pipeline.load_state_dict(ckpt['model_state_dict'], strict=False)
    
    # Pre-train statistics (Min-Max)
    has_norm = 'min' in ckpt
    p_min = ckpt['min'] if has_norm else None
    p_max = ckpt['max'] if has_norm else None
    
    optimizer = AdamW(pipeline.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 3. Fine-tuning Loop
    for epoch in range(1, args.epochs + 1):
        pipeline.train()
        epoch_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # GT IVD (ensure physics consistent mask)
            with torch.no_grad():
                # Note: 'batch' here is normalized, calculate_ivd expects physical u
                if has_norm:
                    u_phys = batch * (p_max.to(device) - p_min.to(device) + 1e-8) + p_min.to(device)
                else:
                    u_phys = batch # Assume fallback
                
                ivd = calculate_ivd(u_phys)
                gt_mask = (ivd > 0).float().unsqueeze(1)
            
            pred_logits = pipeline(batch)
            
            # Weighted Paper Loss (BCE + MSE)
            loss = vortex_mae_paper_loss(pred_logits, gt_mask, pos_weight=args.pos_weight)
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        scheduler.step()
        iou = calculate_iou(torch.sigmoid(pred_logits), gt_mask)
        print(f"Epoch {epoch} | Loss: {epoch_loss/len(train_loader):.6f} | IoU: {iou.item():.4f}")
        
    # Save with normalization stats
    torch.save({
        'model_state_dict': pipeline.state_dict(),
        'min': p_min if has_norm else torch.zeros(1),
        'max': p_max if has_norm else torch.ones(1),
    }, os.path.join(args.save_dir, "fused_best.pth"))
    print(f"Saved fine-tuned model to {args.save_dir}/fused_best.pth")

if __name__ == "__main__":
    main()
