"""
train_mae.py - 3D MAE Pre-training Script
=========================================
Pre-trains the 3D Swin-ViT Masked Autoencoder on a sequence of .vti files.
Includes physics-informed loss (MSE + divergence penalty).
"""
import os
import argparse
import glob
import numpy as np
import pyvista as pv
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from pipeline import FlowVortexFusionPipeline
from mae3d import pi_mae_loss
from data_loader import read_single_vti, read_vti_with_vector


def main():
    parser = argparse.ArgumentParser(description="Pre-train 3D MAE on a VTI directory sequence.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the directory containing .vti files")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--mask_ratio", type=float, default=0.75, help="Masking ratio (0.0 to 1.0)")
    parser.add_argument("--lambda_div", type=float, default=0.1, help="Weight for Divergence penalty loss")
    parser.add_argument("--time_window", type=int, default=1, help="Time window size (set to 1 for 3D spatial MAE, >1 for Spatio-temporal)")
    parser.add_argument("--vector_name", type=str, default=None, help="Name of vector array in VTI")
    parser.add_argument("--save_dir", type=str, default="./checkpoints_fusion", help="Directory to save checkpoints")
    parser.add_argument("--max_files", type=int, default=None, help="Limit number of files for pre-training")
    
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    # 自动识别 CUDA 优先
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"==========================================")
    print(f"Using device: {device}")
    
    # 1. Dataset Splitting
    all_vti_files = sorted(glob.glob(os.path.join(args.data_dir, "*.vti")))
    if len(all_vti_files) == 0:
        raise ValueError(f"No .vti files found in {args.data_dir}")
        
    if args.max_files is not None:
        all_vti_files = all_vti_files[:args.max_files]
        
    num_total = len(all_vti_files)
    num_train = int(num_total * 0.8)
    num_eval = num_total - num_train
    
    print(f"Dataset partitioning (80/20): Train: {num_train} | Eval: {num_eval} (Total: {num_total})")
    
    from data_loader import VTIFlowDataset
    
    train_dataset = VTIFlowDataset(
        args.data_dir, split="pretrain_train", time_window=args.time_window, 
        vector_name=args.vector_name, normalize=True, crop_size=128, max_files=args.max_files
    )
    test_dataset = VTIFlowDataset(
        args.data_dir, split="pretrain_val", time_window=args.time_window, 
        vector_name=args.vector_name, normalize=True, crop_size=128, max_files=args.max_files
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    sample_batch = next(iter(train_loader))
    B, C, D, H, W = sample_batch.shape
    print(f"Data shape loaded: {C}x{D}x{H}x{W} (C x D x H x W)")

    # 2. Model Initialization
    print(f"Initializing Fused 3D MAE with mask ratio {args.mask_ratio}...")
    from pipeline import FlowVortexFusionPipeline
    pipeline = FlowVortexFusionPipeline(
        mode='pretrain',
        patch_size=(4, 4, 4), # Synchronized with paper
        in_chans=C,
        embed_dim=48, 
        depths=[2, 2, 6, 2], 
        num_heads=[3, 6, 12, 24], 
        window_size=(4, 4, 4),
        mask_ratio=args.mask_ratio
    )
    
    pipeline = pipeline.to(device)
    optimizer = AdamW(pipeline.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    def get_psnr(img1, img2):
        mse = torch.mean((img1 - img2) ** 2)
        if mse == 0: return 100.0
        return -10.0 * torch.log10(mse)

    # 3. Training Loop
    best_test_loss = float('inf')
    print("Starting Pre-training...")
    
    for epoch in range(1, args.epochs + 1):
        pipeline.train()
        total_train_loss, total_mse, total_div, total_train_psnr = 0.0, 0.0, 0.0, 0.0
        
        for batch_idx, volume in enumerate(train_loader):
            volume = volume.to(device).float()
            optimizer.zero_grad()
            
            # Forward pass
            x_rec, mask, ivd_pred = pipeline(volume)
            
            # Calculate loss (pi_mae_loss internal handles targets)
            loss, mse_loss, div_loss = pi_mae_loss(x_rec, volume, mask, lambda_div=args.lambda_div)
            
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            total_mse += mse_loss.item()
            total_div += div_loss.item()
            total_train_psnr += get_psnr(x_rec, volume).item()
            
        scheduler.step()
        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_psnr = total_train_psnr / len(train_loader)
        
        # Validation
        pipeline.eval()
        total_test_loss, total_test_psnr = 0.0, 0.0
        with torch.no_grad():
            for volume in test_loader:
                volume = volume.to(device).float()
                x_rec, mask, _ = pipeline(volume)
                loss, _, _ = pi_mae_loss(x_rec, volume, mask, lambda_div=args.lambda_div)
                total_test_loss += loss.item()
                total_test_psnr += get_psnr(x_rec, volume).item()
                
        avg_test_loss = total_test_loss / len(test_loader)
        avg_test_psnr = total_test_psnr / len(test_loader)
        
        print(f"Epoch [{epoch}/{args.epochs}] | "
              f"LR: {scheduler.get_last_lr()[0]:.2e} | "
              f"Loss: {avg_test_loss:.4f} | PSNR: {avg_test_psnr:.2f}dB | "
              f"Train MSE: {total_mse/len(train_loader):.4f}")
              
        # Save Checkpoint
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            ckpt_path = os.path.join(args.save_dir, "mae_best_checkpoint.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': pipeline.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_test_loss,
                'min': torch.from_numpy(train_dataset._min),
                'max': torch.from_numpy(train_dataset._max),
            }, ckpt_path)
            print(f"  -> Saved new best checkpoint to {ckpt_path}")
            
        # Optional: Save visual reconstruction
        if epoch % 10 == 0 or epoch == args.epochs:
            sample_rec = x_rec[0].cpu().numpy()
            sample_true = volume[0].cpu().numpy()
            
            # Reverse Min-Max
            v_min = train_dataset._min[0]
            v_max = train_dataset._max[0]
            sample_rec = sample_rec * (v_max - v_min) + v_min
            sample_true = sample_true * (v_max - v_min) + v_min
            
            vis_mesh = pv.ImageData()
            vis_mesh.dimensions = (sample_rec.shape[-1], sample_rec.shape[-2], sample_rec.shape[-3])
            vis_mesh.spacing, vis_mesh.origin = (1.0, 1.0, 1.0), (0.0, 0.0, 0.0)
            
            vis_mesh.point_data["Reconstructed_Velocity"] = np.stack([sample_rec[0].flatten(order='C'), sample_rec[1].flatten(order='C'), sample_rec[2].flatten(order='C')], axis=1)
            vis_mesh.point_data["GroundTruth_Velocity"] = np.stack([sample_true[0].flatten(order='C'), sample_true[1].flatten(order='C'), sample_true[2].flatten(order='C')], axis=1)
            
            out_vti = os.path.join(args.save_dir, f"epoch_{epoch}_test_reconstruction.vti")
            vis_mesh.save(out_vti)
            print(f"  -> Saved reconstructed VTI to {out_vti}")

    print("\nPre-training complete!")

if __name__ == "__main__":
    main()
