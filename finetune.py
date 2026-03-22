import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    parser.add_argument("--lr", type=float, default=4e-4, help="Learning rate (Paper consistent)")
    parser.add_argument("--pos_weight", type=float, default=5.0, help="Positive class weight for paper loss")
    parser.add_argument("--rec_weight", type=float, default=5.0, help="Weight for reconstruction MSE loss")
    parser.add_argument("--save_dir", type=str, default="./checkpoints_finetune")
    parser.add_argument("--max_files", type=int, default=None, help="Limit number of files for fine-tuning")
    
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    # 自动识别 CUDA 优先
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 2. Model & Checkpoint
    print(f"Loading pretrained weights from {args.pretrained_ckpt}...")
    ckpt = torch.load(args.pretrained_ckpt, map_location=device)
    
    has_norm = 'min' in ckpt
    norm_stats = (ckpt['min'].cpu().numpy(), ckpt['max'].cpu().numpy()) if has_norm else None
    p_min = ckpt['min'].to(device) if has_norm else None
    p_max = ckpt['max'].to(device) if has_norm else None
    
    # 1. Dataset
    train_dataset = VTIFlowDataset(args.data_dir, split="finetune_train", crop_size=128, max_files=args.max_files, norm_stats=norm_stats)
    val_dataset = VTIFlowDataset(args.data_dir, split="finetune_val", crop_size=128, max_files=args.max_files, norm_stats=norm_stats)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    in_chans = train_dataset[0].shape[0]
    
    # 2. Model
    pipeline = FlowVortexFusionPipeline(mode='segmentation', in_chans=in_chans, patch_size=(4, 4, 4))
    pipeline.to(device)
    pipeline.load_state_dict(ckpt['model_state_dict'], strict=False)
    
    from torch.optim.lr_scheduler import StepLR
    optimizer = AdamW(pipeline.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.8)
    
    # PSNR Helper (Inputs are [0, 1] normalized)
    def get_psnr(img1, img2):
        mse = torch.mean((img1 - img2) ** 2)
        if mse == 0: return 100
        return -10 * torch.log10(mse)

    # 3. Fine-tuning Loop
    best_val_iou = -1.0
    
    for epoch in range(1, args.epochs + 1):
        # --- Training Phase ---
        pipeline.train()
        train_loss, train_iou, train_psnr = 0.0, 0.0, 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
            batch = batch.to(device).float()
            optimizer.zero_grad()
            
            # GT Generation
            with torch.no_grad():
                if has_norm:
                    u_phys = batch * (p_max - p_min + 1e-8) + p_min
                else:
                    u_phys = batch
                ivd = calculate_ivd(u_phys)
                gt_mask = (ivd > 0).float().unsqueeze(1)
                
                # Patch-level GT (Binary Selection) via Max Pool
                # We use the same patch_size = (4, 4, 4)
                gt_binary = F.max_pool3d(gt_mask, kernel_size=4, stride=4)
            
            pred_logits, pred_rec, pred_binary = pipeline(batch)
            
            # 1. Voxel-level loss (IVD Segmentation)
            loss_seg = vortex_mae_paper_loss(pred_logits, gt_mask, pos_weight=args.pos_weight)
            
            # 2. Patch-level loss (Binary Selection)
            # Use fixed pos_weight=2.0 for patch-level as it's less sparse than voxel-level
            loss_binary = F.binary_cross_entropy_with_logits(pred_binary, gt_binary, 
                                                            pos_weight=torch.tensor([2.0], device=device))
            
            # 3. Reconstruction loss
            loss_rec = F.mse_loss(pred_rec, batch)
            
            total_loss = loss_seg + 0.5 * loss_binary + args.rec_weight * loss_rec
            
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item()
            train_iou += calculate_iou(torch.sigmoid(pred_logits), gt_mask).item()
            train_psnr += get_psnr(pred_rec, batch).item()
            
        scheduler.step()
        
        # --- Validation Phase ---
        pipeline.eval()
        val_iou, val_psnr = 0.0, 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device).float()
                if has_norm:
                    u_phys = batch * (p_max - p_min + 1e-8) + p_min
                else:
                    u_phys = batch
                ivd = calculate_ivd(u_phys)
                gt_mask = (ivd > 0).float().unsqueeze(1)
                
                pred_logits, pred_rec, pred_binary = pipeline(batch)
                val_iou += calculate_iou(torch.sigmoid(pred_logits), gt_mask).item()
                val_psnr += get_psnr(pred_rec, batch).item()
        
        avg_train_iou = train_iou / len(train_loader)
        avg_val_iou = val_iou / len(val_loader) if len(val_loader) > 0 else 0
        avg_train_psnr = train_psnr / len(train_loader)
        avg_val_psnr = val_psnr / len(val_loader) if len(val_loader) > 0 else 0
        
        print(f"Epoch {epoch} | Loss: {train_loss/len(train_loader):.4f} | "
              f"Train IoU: {avg_train_iou:.4f} | Val IoU: {avg_val_iou:.4f} | "
              f"Train PSNR: {avg_train_psnr:.2f}dB | Val PSNR: {avg_val_psnr:.2f}dB")
        
        # Save Best
        if avg_val_iou > best_val_iou:
            best_val_iou = avg_val_iou
            save_path = os.path.join(args.save_dir, "fused_best.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': pipeline.state_dict(),
                'min': p_min.cpu() if has_norm else torch.zeros(1),
                'max': p_max.cpu() if has_norm else torch.ones(1),
                'val_iou': best_val_iou
            }, save_path)
            print(f" -> Saved NEW BEST checkpoint (Val IoU: {best_val_iou:.4f})")

if __name__ == "__main__":
    main()
