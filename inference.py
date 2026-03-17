import os
import argparse
import glob
import numpy as np
import torch
import pyvista as pv
from pipeline import FlowVortexFusionPipeline
from vortex import calculate_ivd
from data_loader import load_single_vti_as_tensor

def main():
    parser = argparse.ArgumentParser(description="Inference for Fused VortexMAE.")
    parser.add_argument("data_path", type=str, help="Path to a .vti file or directory of .vti files.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to fused_best.pth.")
    parser.add_argument("--save_dir", type=str, default="./results_fused")
    parser.add_argument("--mode", type=str, default="segmentation", choices=["pretrain", "segmentation"])
    
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Prepare files
    if os.path.isdir(args.data_path):
        all_files = sorted(glob.glob(os.path.join(args.data_path, "*.vti")))
        num_total = len(all_files)
        idx_start = int(num_total * 0.40)
        idx_end = int(num_total * 0.70)
        files = all_files[idx_start:idx_end]
        print(f"Total files in dir: {num_total}, using {len(files)} for inference (indices {idx_start}:{idx_end})")
    else:
        files = [args.data_path]
    
    if not files:
        print("No .vti files found.")
        return

    # 2. Load Checkpoint and Initialize Model
    print("Loading Checkpoint...")
    ckpt = torch.load(args.checkpoint, map_location=device)
    mean = ckpt['mean'].to(device)
    std = ckpt['std'].to(device)
    
    sample_tensor = load_single_vti_as_tensor(files[0])
    in_chans = sample_tensor.shape[1]
    
    pipeline = FlowVortexFusionPipeline(mode=args.mode, in_chans=in_chans)
    pipeline.load_state_dict(ckpt['model_state_dict'])
    pipeline.to(device)
    pipeline.eval()

    # 3. Inference Loop
    print(f"Starting inference on {len(files)} files...")
    for f in files:
        fname = os.path.basename(f)
        print(f"Processing: {fname}")
        
        # Load and Normalize
        tensor_input = load_single_vti_as_tensor(f).to(device)
        tensor_input_norm = (tensor_input - mean) / std
        
        # Ensure batch dimension
        if len(tensor_input_norm.shape) == 4:
            tensor_input_norm = tensor_input_norm.unsqueeze(0)
            
        with torch.no_grad():
            if args.mode == 'pretrain':
                # Reconstruct and Pred IVD (Probability)
                x_rec, mask, ivd_pred = pipeline(tensor_input_norm)
                out_v = x_rec.squeeze(0).cpu().numpy()
                out_ivd = ivd_pred.squeeze(0).cpu().numpy()
            else:
                # Direct Binary Segmentation (Sigmoid output)
                seg_mask = pipeline(tensor_input_norm)
                out_ivd = seg_mask.squeeze(0).squeeze(0).cpu().numpy()
                out_v = None

        # 4. Save result to VTI
        mesh = pv.read(f)
        
        # RESET METADATA: Match unit spacing for ParaView visualization (0-639 bounds)
        mesh.spacing = (1.0, 1.0, 1.0)
        mesh.origin = (0.0, 0.0, 0.0)
        
        if out_v is not None:
            # Reverse normalization for visualization
            v_mean = mean.cpu().numpy().reshape(3, 1, 1, 1)
            v_std = std.cpu().numpy().reshape(3, 1, 1, 1)
            out_v = out_v * v_std + v_mean
            
            u, v, w = out_v[0].flatten(order='C'), out_v[1].flatten(order='C'), out_v[2].flatten(order='C')
            mesh.point_data["Reconstructed_Velocity"] = np.stack([u, v, w], axis=1)
        
        # Standardized naming for comparison
        mesh.point_data["Pred_Prob_Map"] = out_ivd.flatten(order='C')
        mesh.point_data["Binary_Selection"] = (out_ivd > 0.5).astype(np.float32).flatten(order='C')
        
        # GT calculation (Unit spacing)
        gt_ivd = calculate_ivd(tensor_input).squeeze(0).cpu().numpy()
        mesh.point_data["GT_IVD_Field"] = gt_ivd.flatten(order='C')
        mesh.point_data["GT_IVD_Mask"] = (gt_ivd > 0).astype(np.float32).flatten(order='C')
        
        out_name = fname.replace(".vti", "_fused.vti")
        out_path = os.path.join(args.save_dir, out_name)
        mesh.save(out_path)
        print(f"Saved: {out_name}")

if __name__ == "__main__":
    main()
