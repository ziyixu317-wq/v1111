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

class FileListVTIDataset(Dataset):
    def __init__(self, file_paths, velocity_names=("u", "v", "w"), vector_name=None):
        self.file_paths = file_paths
        self.velocity_names = velocity_names
        self.vector_name = vector_name
        
        print(f"Loading {len(file_paths)} VTI files into memory for fine-tuning...")
        self.cache = []
        for f in file_paths:
            if self.vector_name:
                vel = read_vti_with_vector(f, self.vector_name)
            else:
                vel = read_single_vti(f, self.velocity_names)
            self.cache.append(vel)
            
        self.cache = np.stack(self.cache, axis=0)
        self.mean = self.cache.mean(axis=(0,2,3,4), keepdims=True)
        self.std = self.cache.std(axis=(0,2,3,4), keepdims=True) + 1e-8
        self.cache = (self.cache - self.mean) / self.std
        
    def __len__(self):
        return len(self.cache)
        
    def __getitem__(self, idx):
        return torch.from_numpy(self.cache[idx])

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Fused VortexMAE on IVD Masks.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--pretrained_ckpt", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_dir", type=str, default="./checkpoints_fused_finetune")
    
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # 1. Dataset
    # 使用总数据的 35% 到 40% 进行微调 (即 5% 的数据)
    all_files = sorted(glob.glob(os.path.join(args.data_dir, "*.vti")))
    num_total = len(all_files)
    idx_start = int(num_total * 0.35)
    idx_end = int(num_total * 0.40)
    finetune_files = all_files[idx_start:idx_end]
    
    print(f"Total files: {num_total}, using {len(finetune_files)} for fine-tuning (indices {idx_start}:{idx_end})")
    
    if not finetune_files:
        print("Not enough files for fine-tuning.")
        return

    train_dataset = FileListVTIDataset(finetune_files)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    in_chans = train_dataset[0].shape[0]
    
    # 2. Model
    pipeline = FlowVortexFusionPipeline(mode='segmentation', in_chans=in_chans)
    pipeline.to(device)
    
    print(f"Loading pretrained weights from {args.pretrained_ckpt}...")
    ckpt = torch.load(args.pretrained_ckpt, map_location=device)
    pipeline.load_state_dict(ckpt['model_state_dict'], strict=False)
    
    optimizer = AdamW(pipeline.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 3. Fine-tuning Loop
    for epoch in range(1, args.epochs + 1):
        pipeline.train()
        epoch_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # GT IVD (Uses unit spacing 1.0)
            with torch.no_grad():
                ivd = calculate_ivd(batch)
                gt_mask = (ivd > 0).float().unsqueeze(1)
            
            pred_mask = pipeline(batch)
            
            # Hybrid Loss
            bce = nn.functional.binary_cross_entropy(pred_mask, gt_mask)
            l2 = nn.functional.mse_loss(pred_mask, gt_mask)
            loss = 0.5 * bce + 0.5 * l2
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        scheduler.step()
        print(f"Epoch {epoch} | Loss: {epoch_loss/len(train_loader):.6f}")
        
    # Save with normalization stats
    torch.save({
        'model_state_dict': pipeline.state_dict(),
        'mean': torch.from_numpy(train_dataset.mean),
        'std': torch.from_numpy(train_dataset.std),
    }, os.path.join(args.save_dir, "fused_best.pth"))
    print(f"Saved fine-tuned model to {args.save_dir}/fused_best.pth")

if __name__ == "__main__":
    main()
