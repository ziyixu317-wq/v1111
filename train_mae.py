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
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    try:
        import torch_xla.core.xla_model as xm
        import torch_xla
        device = torch_xla.device() # Correct way for modern torch_xla
        is_tpu = True
        print(f"==========================================")
        print(f"Using TPU device: {device}")
    except (ImportError, AttributeError):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        is_tpu = False
        print(f"==========================================")
        print(f"Using device: {device}")
    
    # 1. Dataset Splitting
    all_vti_files = sorted(glob.glob(os.path.join(args.data_dir, "*.vti")))
    if len(all_vti_files) == 0:
        raise ValueError(f"No .vti files found in {args.data_dir}")
        
    print(f"Found {len(all_vti_files)} total .vti files in directory.")
    
    # 按照 30% 训练集, 5% 验证集 划分 (总数据的前 35%)
    num_total = len(all_vti_files)
    num_train = int(num_total * 0.3)
    num_eval = int(num_total * 0.05)
    
    print(f"Splitting {num_total} timesteps: Train: {num_train} (30%) | Eval: {num_eval} (5%)")
    
    train_files = all_vti_files[:num_train]
    test_files = all_vti_files[num_train : num_train + num_eval]
    
    # 之前在 data_loader.py 里定义的 MultiStepVTIDataset 恰好就是处理一连串文件路径的
    from data_loader import VTIFlowDataset
    
    # VTIFlowDataset 会把传进来的文件夹里所有 VTI 按顺序组合成时间序列
    # 为了精细控制切分，我们可以稍微复写或直接借用一个可以传 file_list 的类
    # 既然 VTIFlowDataset 要求传 data_dir，我们这里实现一个能直接吃 file_list 的封装：
    class FileListVTIDataset(Dataset):
        def __init__(self, file_paths, time_window=1, velocity_names=("u", "v", "w"), vector_name=None):
            self.file_paths = file_paths
            self.time_window = time_window
            self.vector_name = vector_name
            self.velocity_names = velocity_names
            
            print(f"Loading {len(file_paths)} VTI files into memory...")
            self.cache = []
            for f in file_paths:
                if self.vector_name:
                    from data_loader import read_vti_with_vector
                    vel = read_vti_with_vector(f, self.vector_name)
                else:
                    from data_loader import read_single_vti
                    vel = read_single_vti(f, self.velocity_names)
                self.cache.append(vel)
                
            self.cache = np.stack(self.cache, axis=0)
            self.mean = self.cache.mean(axis=(0,2,3,4), keepdims=True)
            self.std = self.cache.std(axis=(0,2,3,4), keepdims=True) + 1e-8
            self.cache = (self.cache - self.mean) / self.std
            
        def __len__(self):
            return max(1, len(self.cache) - self.time_window + 1)
            
        def __getitem__(self, idx):
            # 获取 time_window 个连续时间步作为输入张量
            seq = self.cache[idx : idx + self.time_window]
            return torch.from_numpy(seq)

    train_dataset = FileListVTIDataset(train_files, time_window=args.time_window, vector_name=args.vector_name)
    test_dataset = FileListVTIDataset(test_files, time_window=args.time_window, vector_name=args.vector_name)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    sample_batch = next(iter(train_loader))
    
    # Depending on 'time_window', shape might be (B, C, D, H, W) or (B, T, C, D, H, W)
    if len(sample_batch.shape) == 5:
        B, C, D, H, W = sample_batch.shape
    elif len(sample_batch.shape) == 6:
        B, T, C, D, H, W = sample_batch.shape
        # For MAE3D currently designed for (B, C, D, H, W), we might need to fold T into C or batch it
        # Assuming spatial Swin3D without T, we take the sequence and fold T into B
        # sample_batch = sample_batch.view(B*T, C, D, H, W)
        print(f"Warning: 6D tensor loaded. Taking only the first timestep for demonstration.")
        sample_batch = sample_batch[:, 0]
        B, C, D, H, W = sample_batch.shape
    else:
        raise ValueError(f"Unexpected batch shape: {sample_batch.shape}")
        
    print(f"Data shape loaded: {C}x{D}x{H}x{W} (C x D x H x W)")

    # 2. Model Initialization
    print(f"Initializing Fused 3D MAE with mask ratio {args.mask_ratio}...")
    from pipeline import FlowVortexFusionPipeline
    pipeline = FlowVortexFusionPipeline(
        mode='pretrain',
        patch_size=(2, 4, 4),
        in_chans=C,
        embed_dim=48, 
        depths=[2, 2, 6, 2], 
        num_heads=[3, 6, 12, 24], 
        window_size=(4, 4, 4),
        mask_ratio=args.mask_ratio
    )
    
    pipeline = pipeline.to(device)
    
    # 开启 TPU/BF16 混合精度优化
    if is_tpu:
        print("Enabling bfloat16 precision for TPU optimization...")
        pipeline = pipeline.to(torch.bfloat16)

    optimizer = AdamW(pipeline.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 3. Training Loop
    best_test_loss = float('inf')
    print("Starting Pre-training...")
    
    for epoch in range(1, args.epochs + 1):
        pipeline.train()
        total_train_loss, total_mse, total_div = 0.0, 0.0, 0.0
        
        for batch_idx, volume in enumerate(train_loader):
            if len(volume.shape) == 6:
                volume = volume.view(-1, *volume.shape[2:])
            
            volume = volume.to(device)
            if is_tpu:
                volume = volume.to(torch.bfloat16)
                
            optimizer.zero_grad()
            
            # Forward pass
            x_rec, mask, ivd_pred = pipeline(volume)
            
            # Calculate loss (pi_mae_loss internal handles targets)
            # Ensure target matches volume's dtype
            loss, mse_loss, div_loss = pi_mae_loss(x_rec, volume, mask, lambda_div=args.lambda_div)
            
            loss.backward()
            
            if is_tpu:
                xm.optimizer_step(optimizer) # TPU specific step
            else:
                optimizer.step()
            
            total_train_loss += loss.item()
            total_mse += mse_loss.item()
            total_div += div_loss.item()
            
        scheduler.step()
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation
        pipeline.eval()
        total_test_loss = 0.0
        with torch.no_grad():
            for volume in test_loader:
                if len(volume.shape) == 6:
                    volume = volume.view(-1, *volume.shape[2:])
                volume = volume.to(device)
                if is_tpu:
                    volume = volume.to(torch.bfloat16)
                x_rec, mask, _ = pipeline(volume)
                loss, _, _ = pi_mae_loss(x_rec, volume, mask, lambda_div=args.lambda_div)
                total_test_loss += loss.item()
                
        avg_test_loss = total_test_loss / len(test_loader)
        
        print(f"Epoch [{epoch}/{args.epochs}] | "
              f"LR: {scheduler.get_last_lr()[0]:.2e} | "
              f"Train Loss: {avg_train_loss:.4f} (MSE: {total_mse/len(train_loader):.4f}, Div: {total_div/len(train_loader):.4f}) | "
              f"Test Loss: {avg_test_loss:.4f}")
              
        # Save Checkpoint and Visualization
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            ckpt_path = os.path.join(args.save_dir, "mae_best_checkpoint.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': pipeline.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_test_loss,
                'mean': torch.from_numpy(train_dataset.mean),
                'std': torch.from_numpy(train_dataset.std),
            }, ckpt_path)
            print(f"  -> Saved new best checkpoint to {ckpt_path}")
            
        # Optional: Save a visual reconstruction of the FIRST batch in test_loader
        # We save it every 10 epochs or at the very last epoch to save disk space
        if epoch % 10 == 0 or epoch == args.epochs:
            # We already have the last `x_rec` and `volume` from the test loop
            # x_rec shape: (B, 3, D, H, W)
            # Take the first sample in the batch
            sample_rec = x_rec[0].cpu().numpy() # (3, D, H, W)
            sample_true = volume[0].cpu().numpy() # (3, D, H, W)
            
            # Reverse normalization if standard deviation/mean was stored
            # (Assuming test_dataset has mean and std accessible)
            try:
                mean = test_dataset.mean.squeeze(0).squeeze(0) # (1, 3, 1, 1, 1) to (3, 1, 1) usually, or 0.0
                std = test_dataset.std.squeeze(0).squeeze(0)
                sample_rec = sample_rec * std + mean
                sample_true = sample_true * std + mean
            except:
                pass # If it fails, just save the normalized version
            
            # Save using PyVista (We create an ImageData block)
            vis_mesh = pv.ImageData()
            vis_mesh.dimensions = (W, H, D)
            vis_mesh.spacing = (1.0, 1.0, 1.0)
            vis_mesh.origin = (0.0, 0.0, 0.0)
            
            # Flatten with order='C' (Z-fastest locally, matching VTK x-fastest spatial array interpretation usually)
            u_rec = sample_rec[0].flatten(order='C')
            v_rec = sample_rec[1].flatten(order='C')
            w_rec = sample_rec[2].flatten(order='C')
            vec_rec = np.stack([u_rec, v_rec, w_rec], axis=1)
            
            u_true = sample_true[0].flatten(order='C')
            v_true = sample_true[1].flatten(order='C')
            w_true = sample_true[2].flatten(order='C')
            vec_true = np.stack([u_true, v_true, w_true], axis=1)
            
            vis_mesh.point_data["Reconstructed_Velocity"] = vec_rec
            vis_mesh.point_data["GroundTruth_Velocity"] = vec_true
            
            out_vti = os.path.join(args.save_dir, f"epoch_{epoch}_test_reconstruction.vti")
            vis_mesh.save(out_vti)
            print(f"  -> Saved reconstructed VTI for visual inspection to {out_vti}")

    print("\nPre-training complete!")
    print("==========================================")

if __name__ == "__main__":
    main()
