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

class SingleVTITimeSequenceDataset(Dataset):
    """
    Dataset to handle a SINGLE .vti file that contains a time dimension T.
    Splits the T sequence into sliding windows of shape (T_window, C, D, H, W).
    """
    def __init__(self, filepath, time_window=4, stride=1, 
                 start_idx=0, end_idx=None, 
                 velocity_names=("u", "v", "w"), vector_name=None, normalize=True):
        self.filepath = filepath
        self.time_window = time_window
        self.stride = stride
        self.normalize = normalize
        
        print(f"Loading single VTI file into memory: {filepath}")
        if vector_name is not None:
            # (C, D, H, W) if it's a single snapshot, OR (T, C, D, H, W) if it's a 4D sequence
            vel = read_vti_with_vector(filepath, vector_name)
        else:
            vel = read_single_vti(filepath, velocity_names)
            
        # Handling the shape.
        # If the input is (C, D, H, W) and actually represents (T=1, C, D, H, W)
        # We need to make sure the VTI actually contains a time dimension.
        # Often VTIs are just (C, D, H, W). If your VTI is 4D (T, X, Y, Z), PyVista 
        # reading might flatten T into the spatial components or scalar arrays.
        # ASSUMPTION: The user's VTI has shape (T, C, D, H, W) after some custom reshape,
        # OR the "C" dimension actually holds (T * 3) components.
        
        # Let's assume the read function returns (T, C, D, H, W) OR we need to load a directory of snapshots.
        # Wait, the user said "这个vti文件有151个时间步" (This single VTI file has 151 timesteps).
        # In VTK, a single VTI with time steps usually stores them as separate arrays (e.g., "Velocity_t0", "Velocity_t1"...)
        # OR it's a multi-block dataset. 
        # Let's adjust the logic to accept a pre-loaded tensor if passed, or load it appropriately.

        # To be safe and flexible, let's assume `data_tensor` is passed directly in:
        pass

class TensorTimeSequenceDataset(Dataset):
    """
    Takes a pre-loaded PyTorch Tensor of shape (T, C, D, H, W)
    and creates sliding windows for training/testing.
    """
    def __init__(self, data_tensor: torch.Tensor, time_window=4, stride=1, normalize=True):
        """
        data_tensor: (T, C, D, H, W)
        """
        self.data = data_tensor.float()
        self.time_window = time_window
        self.stride = stride
        self.normalize = normalize
        
        if self.normalize:
            self.mean = self.data.mean(dim=(0, 2, 3, 4), keepdim=True)
            self.std = self.data.std(dim=(0, 2, 3, 4), keepdim=True) + 1e-8
            self.data = (self.data - self.mean) / self.std

        self.num_sequences = max(1, (self.data.shape[0] - self.time_window) // self.stride + 1)

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.time_window
        
        # If we reach the end and don't have enough frames, pad with the last frame
        seq = self.data[start:end]
        if seq.shape[0] < self.time_window:
            pad_size = self.time_window - seq.shape[0]
            pad_frames = seq[-1:].repeat(pad_size, 1, 1, 1, 1)
            seq = torch.cat([seq, pad_frames], dim=0)
            
        return seq

def main():
    parser = argparse.ArgumentParser(description="Pre-train 3D MAE on a VTI directory sequence.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the directory containing .vti files")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--mask_ratio", type=float, default=0.75, help="Masking ratio (0.0 to 1.0)")
    parser.add_argument("--lambda_div", type=float, default=0.1, help="Weight for Divergence penalty loss")
    parser.add_argument("--time_window", type=int, default=1, help="Time window size (set to 1 for 3D spatial MAE, >1 for Spatio-temporal)")
    parser.add_argument("--vector_name", type=str, default=None, help="Name of vector array in VTI")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"==========================================")
    print(f"Using device: {device}")
    
    # 1. Dataset Splitting
    all_vti_files = sorted(glob.glob(os.path.join(args.data_dir, "*.vti")))
    if len(all_vti_files) == 0:
        raise ValueError(f"No .vti files found in {args.data_dir}")
        
    print(f"Found {len(all_vti_files)} total .vti files in directory.")
    
    # 按照 7:3 (70% 训练集, 30% 测试集) 划分
    num_total = len(all_vti_files)
    num_train = int(num_total * 0.7)
    num_test = num_total - num_train
    
    print(f"Splitting {num_total} timesteps (7:3 ratio): Train: {num_train} | Test: {num_test}")
    
    train_files = all_vti_files[:num_train]
    test_files = all_vti_files[num_train:]
    
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
                    vel, sp = read_vti_with_vector(f, self.vector_name)
                else:
                    from data_loader import read_single_vti
                    vel, sp = read_single_vti(f, self.velocity_names)
                self.cache.append(vel)
                if hasattr(self, 'spacing') == False:
                    self.spacing = sp # (dx, dy, dz)
                
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
            optimizer.zero_grad()
            
            # Forward pass: x_rec, mask, ivd_pred
            x_rec, mask, ivd_pred = pipeline(volume)
            
            # Calculate PI-MAE loss
            from mae3d import pi_mae_loss
            dx, dy, dz = train_dataset.spacing
            loss, mse_loss, div_loss = pi_mae_loss(x_rec, volume, mask, dx=dx, dy=dy, dz=dz, lambda_div=args.lambda_div)
            
            loss.backward()
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
                x_rec, mask, _ = pipeline(volume)
                dx, dy, dz = test_dataset.spacing
                loss, _, _ = pi_mae_loss(x_rec, volume, mask, dx=dx, dy=dy, dz=dz, lambda_div=args.lambda_div)
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
            vis_mesh.dimensions = (W, H, D) # VTK uses (x, y, z)
            
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
