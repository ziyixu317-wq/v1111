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
    # Split: 8 for train, 3 for val (assuming 11 total files)
    train_dataset = VTIFlowDataset(args.data_dir, split="finetune_train", crop_size=128)
    val_dataset = VTIFlowDataset(args.data_dir, split="finetune_val", crop_size=128)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    in_chans = train_dataset[0].shape[0]
    
    # 2. Model
    pipeline = FlowVortexFusionPipeline(mode='segmentation', in_chans=in_chans, patch_size=(4, 4, 4))
    pipeline.to(device)
    
    print(f"Loading pretrained weights from {args.pretrained_ckpt}...")
    ckpt = torch.load(args.pretrained_ckpt, map_location=device)
    pipeline.load_state_dict(ckpt['model_state_dict'], strict=False)
    
    has_norm = 'min' in ckpt
    p_min = ckpt['min'].to(device) if has_norm else None
    p_max = ckpt['max'].to(device) if has_norm else None
    
    optimizer = AdamW(pipeline.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
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
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # GT IVD
            with torch.no_grad():
                if has_norm:
                    u_phys = batch * (p_max - p_min + 1e-8) + p_min
                else:
                    u_phys = batch
                ivd = calculate_ivd(u_phys)
                gt_mask = (ivd > 0).float().unsqueeze(1)
            
            pred_logits, pred_rec = pipeline(batch)
            loss = vortex_mae_paper_loss(pred_logits, gt_mask, pos_weight=args.pos_weight)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_iou += calculate_iou(torch.sigmoid(pred_logits), gt_mask).item()
            train_psnr += get_psnr(pred_rec, batch).item()
            
        scheduler.step()
        
        # --- Validation Phase ---
        pipeline.eval()
        val_iou, val_psnr = 0.0, 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                if has_norm:
                    u_phys = batch * (p_max - p_min + 1e-8) + p_min
                else:
                    u_phys = batch
                ivd = calculate_ivd(u_phys)
                gt_mask = (ivd > 0).float().unsqueeze(1)
                
                pred_logits, pred_rec = pipeline(batch)
                val_iou += calculate_iou(torch.sigmoid(pred_logits), gt_mask).item()
                val_psnr += get_psnr(pred_rec, batch).item()
        
        avg_train_iou = train_iou / len(train_loader)
        avg_val_iou = val_iou / len(val_loader) if len(val_loader) > 0 else 0
        avg_train_psnr = train_psnr / len(train_loader)
        avg_val_psnr = val_psnr / len(val_loader) if len(val_loader) > 0 else 0
        
        print(f"Epoch {epoch} | Loss: {train_loss/len(train_loader):.4f} | "
              f"Train IoU: {avg_train_iou:.4f} | Val IoU: {avg_val_iou:.4f} | "
              f"Rec PSNR: {avg_train_psnr:.2f}dB")
        
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
