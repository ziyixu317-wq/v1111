
import argparse
import torch
import pyvista as pv
import numpy as np
import os
import glob
from data_loader import load_single_vti_as_tensor
from pipeline import FlowVortexFusionPipeline
from vortex import calculate_ivd

def main():
    parser = argparse.ArgumentParser(description="Inference for Fused FlowVortexMAE")
    parser.add_argument("vti_file", type=str)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--mode", type=str, default='segmentation', choices=['pretrain', 'segmentation'])
    parser.add_argument("--save_dir", type=str, default="./inference_results")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Handle Input (File or Directory)
    if os.path.isdir(args.vti_file):
        files = sorted(glob.glob(os.path.join(args.vti_file, "*.vti")))
        print(f"Found {len(files)} .vti files in directory.")
    else:
        files = [args.vti_file]

    if not files:
        print("No .vti files found to process.")
        return

    # 2. Model Initialization
    print("Initializing Model...")
    sample_tensor = load_single_vti_as_tensor(files[0])
    pipeline = FlowVortexFusionPipeline(mode=args.mode, in_chans=sample_tensor.shape[1])
    ckpt = torch.load(args.checkpoint, map_location=device)
    pipeline.load_state_dict(ckpt['model_state_dict'])
    pipeline.to(device)
    pipeline.eval()

    # Load Normalization Stats
    mean = ckpt.get('mean', torch.zeros(1, sample_tensor.shape[1], 1, 1, 1)).to(device)
    std = ckpt.get('std', torch.ones(1, sample_tensor.shape[1], 1, 1, 1)).to(device)

    # 3. Inference Loop
    for f in files:
        print(f"Processing: {os.path.basename(f)}")
        tensor_input, spacing_tensor = load_single_vti_as_tensor(f)
        tensor_input = tensor_input.to(device)
        dx, dy, dz = spacing_tensor.cpu().numpy()
        
        # Apply Normalization
        tensor_input_norm = (tensor_input - mean) / std
        
        if len(tensor_input_norm.shape) == 4:
            tensor_input_norm = tensor_input_norm.unsqueeze(0)
            
        with torch.no_grad():
            if args.mode == 'pretrain':
                x_rec, mask, ivd_pred = pipeline(tensor_input_norm)
                # ...
                out_v = x_rec.squeeze(0).cpu().numpy()
                out_ivd = ivd_pred.squeeze(0).cpu().numpy()
            else:
                seg_mask = pipeline(tensor_input_norm)
                out_ivd = seg_mask.squeeze(0).squeeze(0).cpu().numpy()
                out_v = None

        # 4. Save
        mesh = pv.read(f)
        if out_v is not None:
            # Reverse normalization for visual velocity saving
            v_mean = mean.cpu().numpy().reshape(3, 1, 1, 1)
            v_std = std.cpu().numpy().reshape(3, 1, 1, 1)
            out_v = out_v * v_std + v_mean
            
            u, v, w = out_v[0].flatten(order='C'), out_v[1].flatten(order='C'), out_v[2].flatten(order='C')
            mesh.point_data["Reconstructed_Velocity"] = np.stack([u, v, w], axis=1)
        
        # Unify naming with vortex_mae_repro for comparison
        mesh.point_data["Pred_Prob_Map"] = out_ivd.flatten(order='C')
        mesh.point_data["Binary_Selection"] = (out_ivd > 0.5).astype(np.float32).flatten(order='C')
        
        # Also add GT IVD for reference (calculated with correct spacing)
        gt_ivd = calculate_ivd(tensor_input, dx=dx, dy=dy, dz=dz).squeeze(0).cpu().numpy()
        mesh.point_data["GT_IVD_Field"] = gt_ivd.flatten(order='C')
        mesh.point_data["GT_IVD_Mask"] = (gt_ivd > 0).astype(np.float32).flatten(order='C')
        
        out_path = os.path.join(args.save_dir, os.path.basename(f).replace(".vti", "_fused.vti"))
        mesh.save(out_path)
        print(f"  -> Saved to {out_path}")

if __name__ == "__main__":
    main()
