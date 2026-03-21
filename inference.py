import os
import argparse
import glob
import numpy as np
import torch
import pyvista as pv
from tqdm import tqdm

try:
    from scipy.ndimage import label as ccl_label
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from pipeline import FlowVortexFusionPipeline
from vortex import calculate_ivd
from data_loader import load_single_vti_as_tensor

def main():
    parser = argparse.ArgumentParser(description="Full-Field Inference for Fused FlowVortexNet.")
    parser.add_argument("data_path", type=str, help="Path to .vti file or directory.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best checkpoint.")
    parser.add_argument("--save_dir", type=str, default="./results_fused")
    parser.add_argument("--mode", type=str, default="segmentation", choices=["pretrain", "segmentation"])
    parser.add_argument("--threshold", type=float, default=0.5, help="Binary threshold")
    parser.add_argument("--no_ccl", action="store_true", help="Disable CCL")
    
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Prepare files
    all_files = sorted(glob.glob(os.path.join(args.data_path, "*.vti"))) if os.path.isdir(args.data_path) else [args.data_path]
    if not all_files: return
    
    # Process only first 3 as requested to prevent huge storage usage
    files = all_files[:3]
    print(f"Found {len(all_files)} files. Limiting inference to the first {len(files)} files as requested.")
    
    # 2. Load Checkpoint (Min-Max support)
    print(f"Loading Checkpoint from {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location=device)
    # Check if uses mean/std or min/max
    has_min_max = 'min' in ckpt
    v_min = ckpt['min'].to(device) if has_min_max else None
    v_max = ckpt['max'].to(device) if has_min_max else None
    mean = ckpt.get('mean', torch.zeros(1)).to(device)
    std = ckpt.get('std', torch.ones(1)).to(device)
    
    sample_tensor = load_single_vti_as_tensor(files[0])
    pipeline = FlowVortexFusionPipeline(mode=args.mode, in_chans=sample_tensor.shape[1])
    pipeline.load_state_dict(ckpt['model_state_dict'])
    pipeline.to(device)
    pipeline.eval()

    # Window configuration
    win_size = 128
    stride = 64

    # 3. Inference Loop with Sliding Window
    for f in tqdm(files, desc="Inference"):
        fname = os.path.basename(f)
        tensor_in = load_single_vti_as_tensor(f).to(device)
        C, D, H, W = tensor_in.shape[1:]
        
        # Norm
        if has_min_max:
            t_norm = (tensor_in - v_min) / (v_max - v_min + 1e-8)
        else:
            t_norm = (tensor_in - mean) / std

        # Accumulators
        full_logits = torch.zeros((1, 1, D, H, W), device=device)
        full_count = torch.zeros((1, 1, D, H, W), device=device)
        full_rec = torch.zeros((1, 3, D, H, W), device=device) if args.mode == 'pretrain' else None

        # Tiling
        for d in range(0, D, stride):
            for h in range(0, H, stride):
                for w in range(0, W, stride):
                    d_e, h_e, w_e = min(d + win_size, D), min(h + win_size, H), min(w + win_size, W)
                    d_s, h_s, w_s = d_e - win_size, h_e - win_size, w_e - win_size
                    d_s, h_s, w_s = max(0, d_s), max(0, h_s), max(0, w_s)
                    
                    patch = t_norm[:, :, d_s:d_e, h_s:h_e, w_s:w_e]
                    p_d, p_h, p_w = patch.shape[2], patch.shape[3], patch.shape[4]
                    
                    # Pad to win_size if needed
                    pad_d, pad_h, pad_w = win_size - p_d, win_size - p_h, win_size - p_w
                    if pad_d > 0 or pad_h > 0 or pad_w > 0:
                        patch = torch.nn.functional.pad(patch, (0, pad_w, 0, pad_h, 0, pad_d))
                    
                    with torch.no_grad():
                        if args.mode == 'pretrain':
                            rec, _, _ = pipeline(patch)
                            full_rec[:, :, d_s:d_e, h_s:h_e, w_s:w_e] += rec[:, :, :p_d, :p_h, :p_w]
                        else:
                            logits, rec = pipeline(patch)
                            full_logits[:, :, d_s:d_e, h_s:h_e, w_s:w_e] += logits[:, :, :p_d, :p_h, :p_w]
                            if full_rec is not None:
                                full_rec[:, :, d_s:d_e, h_s:h_e, w_s:w_e] += rec[:, :, :p_d, :p_h, :p_w]
                        full_count[:, :, d_s:d_e, h_s:h_e, w_s:w_e] += 1
        
        # 4. Post-process & Save
        pred_prob = (full_logits / torch.clamp(full_count, min=1.0)).sigmoid()
        p_mask = pred_prob[0, 0].cpu().numpy()
        
        if HAS_SCIPY and not args.no_ccl:
            labeled, num = ccl_label(p_mask > args.threshold)
            if num > 0:
                sizes = np.bincount(labeled.ravel())
                min_s = max(20, int(0.01 * sizes[1:].max()))
                mask_clean = np.zeros_like(p_mask)
                for i in range(1, num + 1):
                    if sizes[i] >= min_s: mask_clean[labeled == i] = 1
                p_mask = mask_clean

        mesh = pv.read(f)
        mesh.spacing, mesh.origin = (1.0, 1.0, 1.0), (0.0, 0.0, 0.0)
        
        if full_rec is not None:
            rec_v = (full_rec / torch.clamp(full_count, min=1.0))[0].cpu().numpy()
            # De-norm
            if has_min_max:
                rec_v = rec_v * (v_max[0].cpu().numpy() - v_min[0].cpu().numpy()) + v_min[0].cpu().numpy()
            else:
                rec_v = rec_v * std[0].cpu().numpy() + mean[0].cpu().numpy()
            mesh.point_data["Reconstructed_Velocity"] = np.stack([rec_v[0].flatten(order='C'), rec_v[1].flatten(order='C'), rec_v[2].flatten(order='C')], axis=1)
            
        mesh.point_data["Pred_Prob_Map"] = (full_logits[0,0]/full_count[0,0]).sigmoid().cpu().numpy().flatten(order='C')
        mesh.point_data["Binary_Selection"] = p_mask.flatten(order='C')
        
        gt_ivd = calculate_ivd(tensor_in).squeeze(0).cpu().numpy()
        mesh.point_data["GT_IVD_Mask"] = (gt_ivd > 0).astype(np.float32).flatten(order='C')
        
        mesh.save(os.path.join(args.save_dir, fname.replace(".vti", "_fused.vti")))

if __name__ == "__main__":
    main()
